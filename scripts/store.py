#!/usr/bin/env python3
"""
store.py — Pure numpy vector store with SQLite metadata + scipy cosine similarity.

No sqlite-vec extension needed. Vectors stored as binary blobs in SQLite.
Search uses scipy.spatial.distance.cosine (fast enough for <100k chunks).

Tables:
  files:   id, path, repo, branch, file_hash, indexed_at
  chunks:  id, file_id, path, chunk_text, chunk_hash, vec_bytes, created_at
  metadata: key = value pairs
  vectors: id, chunk_id, vec_bytes  (numpy-persisted .npy sidecar)

Usage:
  init:      python3 store.py --init --db foo.db
  upsert:    python3 store.py --db foo.db --upsert-file path --chunks '[["text", [0.1,...]], ...]'
  delete:    python3 store.py --db foo.db --delete-file path
  query:     python3 store.py --db foo.db --query-vec "[0.1,...]" --top-k 8
  stats:     python3 store.py --db foo.db --stats
"""
import os, sys, json, hashlib, sqlite3, argparse, struct
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine
from typing import List

DB_PATH = os.environ.get("REVIEW_DB", os.path.expanduser("~/.hermes/pr-review/index.db"))


def norm(vec) -> float:
    return float(np.linalg.norm(vec))


def vec_to_bytes(vec) -> bytes:
    """Pack float32 numpy array into binary blob."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def bytes_to_vec(data: bytes) -> np.ndarray:
    """Unpack binary blob to float32 numpy array."""
    return np.frombuffer(data, dtype=np.float32)


def _get_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str):
    """Create schema: files + chunks + metadata + vectors tables."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = _get_conn(db_path)
    
    # File registry
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            repo TEXT,
            branch TEXT,
            file_hash TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Chunk table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            vec_bytes BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Vector sidecar path (matmul index)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            vec_bytes BLOB NOT NULL
        )
    """)
    
    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    
    conn.commit()
    conn.close()
    print(f"Initialized: {db_path}")


def upsert_file(db_path: str, file_path: str, chunks: List, repo: str = "", branch: str = ""):
    """
    Upsert a file's chunks. chunks: list of (chunk_text: str, vec: List[float])
    Vectors stored as blobs in SQLite + memory-mapped numpy for fast search.
    """
    if not chunks:
        return
    
    conn = _get_conn(db_path)
    try:
        file_hash = hashlib.sha256("|".join(c[0] for c in chunks).encode()).hexdigest()[:16]
        
        # Upsert file record
        cur = conn.execute(
            """INSERT INTO files (path, repo, branch, file_hash) VALUES (?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
               file_hash=excluded.file_hash, repo=excluded.repo, branch=excluded.branch,
               indexed_at=CURRENT_TIMESTAMP""",
            (file_path, repo, branch, file_hash)
        )
        file_id = cur.lastrowid or conn.execute(
            "SELECT id FROM files WHERE path=?", (file_path,)
        ).fetchone()[0]
        
        # Delete old chunks
        old_chunk_ids = [r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE file_id=?", (file_id,)
        ).fetchall()]
        
        conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        for cid in old_chunk_ids:
            conn.execute("DELETE FROM vectors WHERE chunk_id=?", (cid,))
        
        # Insert new chunks
        for chunk_text, vec in chunks:
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
            vec_bytes = vec_to_bytes(vec)
            cur = conn.execute(
                """INSERT INTO chunks (file_id, path, chunk_text, chunk_hash, vec_bytes)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_id, file_path, chunk_text, chunk_hash, vec_bytes)
            )
            conn.execute(
                "INSERT INTO vectors (chunk_id, vec_bytes) VALUES (?, ?)",
                (cur.lastrowid, vec_bytes)
            )
        
        conn.commit()
        print(f"  upserted {len(chunks)} chunks for {file_path}")
    finally:
        conn.close()


def delete_file(db_path: str, file_path: str):
    """Remove all chunks and vectors for a file."""
    conn = _get_conn(db_path)
    try:
        file_row = conn.execute("SELECT id FROM files WHERE path=?", (file_path,)).fetchone()
        if file_row:
            file_id = file_row[0]
            chunk_ids = [r[0] for r in conn.execute(
                "SELECT id FROM chunks WHERE file_id=?", (file_id,)
            ).fetchall()]
            for cid in chunk_ids:
                conn.execute("DELETE FROM vectors WHERE chunk_id=?", (cid,))
            conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            conn.execute("DELETE FROM files WHERE id=?", (file_id,))
            conn.commit()
            print(f"  deleted: {file_path}")
        else:
            print(f"  not found: {file_path}")
    finally:
        conn.close()


def load_vector_index(db_path: str) -> tuple:
    """
    Load all vectors + metadata into memory for fast search.
    Returns (ids, matrix, chunks_meta).
    """
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT c.id, c.chunk_text, c.path, v.vec_bytes FROM chunks c JOIN vectors v ON c.id=v.chunk_id"
    ).fetchall()
    conn.close()
    
    if not rows:
        return [], np.empty((0, 768), dtype=np.float32), []
    
    ids = []
    texts = []
    paths = []
    mat = np.zeros((len(rows), 768), dtype=np.float32)
    
    for i, (cid, text, path, vb) in enumerate(rows):
        ids.append(cid)
        texts.append(text)
        paths.append(path)
        mat[i] = bytes_to_vec(vb)
    
    return ids, mat, list(zip(texts, paths))


def cosine_search(query_vec: np.ndarray, top_k: int = 8,
                  ids=None, matrix=None, texts_paths=None):
    """
    Pure-numpy cosine similarity top-K.
    Returns list of (chunk_text, path, score).
    """
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    
    # Cosine similarity = 1 - cosine distance
    scores = np.dot(m_norm, q_norm.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    
    return [
        (texts_paths[i][0], texts_paths[i][1], float(scores[i]))
        for i in top_idx
    ]


def search(db_path: str, query_vec: List[float], top_k: int = 8):
    """Top-K vector search. Returns (chunk_text, path, score)."""
    ids, matrix, texts_paths = load_vector_index(db_path)
    if not ids:
        return []
    return cosine_search(np.asarray(query_vec, dtype=np.float32), top_k, ids, matrix, texts_paths)


def set_metadata(db_path: str, key: str, value: str):
    conn = _get_conn(db_path)
    conn.execute("INSERT INTO metadata VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))
    conn.commit()
    conn.close()


def get_metadata(db_path: str, key: str) -> str:
    conn = _get_conn(db_path)
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="numpy-vector store with SQLite metadata")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--upsert-file")
    parser.add_argument("--chunks", help='JSON array of ["text", [vec], ...]')
    parser.add_argument("--delete-file")
    parser.add_argument("--set-meta", nargs=2, metavar=("KEY", "VALUE"))
    parser.add_argument("--get-meta", metavar="KEY")
    parser.add_argument("--query-vec", help="JSON float array")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--repo", default="")
    parser.add_argument("--branch", default="")
    args = parser.parse_args()

    if args.init:
        init_db(args.db)
    elif args.set_meta:
        set_metadata(args.db, args.set_meta[0], args.set_meta[1])
        print(f"set {args.set_meta[0]}={args.set_meta[1]}")
    elif args.get_meta:
        print(get_metadata(args.db, args.get_meta) or "")
    elif args.stats:
        conn = _get_conn(args.db)
        n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        print(f"files: {n_files}, chunks: {n_chunks}")
    elif args.upsert_file and args.chunks:
        import json as _json
        pairs = _json.loads(args.chunks)  # [[text, vec], ...]
        upsert_file(args.db, args.upsert_file, pairs, repo=args.repo, branch=args.branch)
    elif args.delete_file:
        delete_file(args.db, args.delete_file)
    elif args.query_vec:
        import json as _json
        qvec = _json.loads(args.query_vec)
        results = search(args.db, qvec, top_k=args.top_k)
        for text, path, score in results:
            print(json.dumps({"text": text[:200], "path": path, "score": round(score, 4)}))
    else:
        parser.print_help()
