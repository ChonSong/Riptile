#!/usr/bin/env python3
"""
init_index.py — One-time repo indexing: walk files → chunk → embed → store in sqlite-vec.

Usage:
  python3 scripts/init_index.py --repo ChonSong/hermes-webui --db ~/.hermes/pr-review/hermes-webui.db

Supports local repos (--local) or git clone (--repo).
"""
import os, sys, json, hashlib, subprocess, argparse, tempfile, shutil

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed as _embed
import store as _store

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt", ".sql", ".html", ".css", ".scss", ".sass",
}

SKIP_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", ".cache", ".pytest_cache",
    "vendor", "pkg", "target", "site", "static", "public", "coverage",
    ".tox", ".mypy_cache", ".ruff_cache", ".Cursor", ".vscode", ".idea",
}


def is_indexable(path, extensions=None, skip_dirs=None):
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    if skip_dirs is None:
        skip_dirs = SKIP_DIRS
    
    parts = path.split(os.sep)
    if any(d in parts for d in skip_dirs):
        return False
    _, ext = os.path.splitext(path)
    return ext.lower() in extensions


def walk_local_repo(local_path, extensions=None, skip_dirs=None):
    """Yield (rel_path, abs_path) for all indexable files."""
    for root, dirs, files in os.walk(local_path):
        # Prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in (skip_dirs or SKIP_DIRS)]
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, local_path)
            if is_indexable(rel, extensions, skip_dirs):
                yield rel, fpath


def clone_repo(repo, temp_dir):
    """Clone a GitHub repo to temp_dir. Returns the repo name."""
    owner, name = repo.split("/", 1)
    clone_url = f"https://github.com/{repo}.git"
    target = os.path.join(temp_dir, name)
    print(f"Cloning {repo}...")
    subprocess.run(["git", "clone", "--depth=1", clone_url, target], check=True, capture_output=True)
    return name, target


def index_file(rel_path, abs_path, db_path, repo, branch, stats):
    """Chunk + embed + store for one file."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        return
    
    if not text.strip():
        return
    
    chunks = _embed.chunk_text(text)
    if not chunks:
        return
    
    vectors = _embed.embed_with_ollama(chunks)
    
    pairs = []
    for chunk, vec in zip(chunks, vectors):
        if vec:
            pairs.append((chunk, vec))
    
    if pairs:
        _store.upsert_file(db_path, rel_path, pairs, repo=repo, branch=branch)
        stats["files_indexed"] += 1
        stats["chunks_indexed"] += len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Index a repo into sqlite-vec")
    parser.add_argument("--repo", help="GitHub repo (owner/name) to clone and index")
    parser.add_argument("--local", help="Local repo path to index directly")
    parser.add_argument("--db", required=True, help="Output .db path")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--extensions", help="Comma-separated extensions (default: all code files)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already in DB")
    args = parser.parse_args()

    if not args.repo and not args.local:
        print("--repo or --local required", file=sys.stderr)
        sys.exit(1)

    # Extensions filter
    extensions = None
    if args.extensions:
        extensions = {f".{e.strip().lstrip('.')}" for e in args.extensions.split(",")}

    # Init DB
    _store.init_db(args.db)
    _store.set_metadata(args.db, "repo", args.repo or args.local or "local")
    _store.set_metadata(args.db, "branch", args.branch)
    _store.set_metadata(args.db, "indexed_at", "")
    from datetime import datetime
    _store.set_metadata(args.db, "indexed_at", datetime.utcnow().isoformat())

    temp_dir = None
    repo_root = None
    repo_name = args.repo or args.local or "local"

    try:
        if args.repo:
            temp_dir = tempfile.mkdtemp(prefix="hermes_pr_review_")
            repo_name, repo_root = clone_repo(args.repo, temp_dir)
            print(f"Cloned to {repo_root}")
        else:
            repo_root = args.local

        stats = {"files_indexed": 0, "chunks_indexed": 0, "skipped": 0}
        
        # Check if skip_existing
        skip_paths = set()
        if args.skip_existing:
            conn = __import__("sqlite3").connect(args.db)
            rows = conn.execute("SELECT path FROM files").fetchall()
            skip_paths = {r[0] for r in rows}
            conn.close()
            print(f"Skipping {len(skip_paths)} already-indexed files")

        file_iter = walk_local_repo(repo_root, extensions=extensions)
        
        for rel_path, abs_path in file_iter:
            if rel_path in skip_paths:
                stats["skipped"] += 1
                continue
            
            print(f"  {rel_path}...", end=" ", flush=True)
            index_file(rel_path, abs_path, args.db, repo_name, args.branch, stats)
        
        print(f"\nDone: {stats['files_indexed']} files, {stats['chunks_indexed']} chunks indexed, {stats['skipped']} skipped")
        
        # Verify sqlite-vec has data
        conn = __import__("sqlite3").connect(args.db)
        vec_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        print(f"sqlite-vec rows: {vec_count}")
        conn.close()
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
