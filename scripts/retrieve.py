#!/usr/bin/env python3
"""
retrieve.py — Retrieve relevant code context using numpy vector search.

Usage:
  python3 retrieve.py --db index.db --query "what does auth.py do" --top-k 8
"""
import os, sys, json, argparse

# Add scripts dir to path so we can import store
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL",    "http://localhost:43311")
OLLAMA_EMBED = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_DB   = os.environ.get("REVIEW_DB", os.path.expanduser("~/.hermes/pr-review/index.db"))


def embed_query(text: str) -> list:
    import requests
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": OLLAMA_EMBED, "input": text},
            timeout=60
        )
        if resp.status_code != 200:
            return [0.0] * 768
        data = resp.json()
        emb = data.get("embeddings", [[]])
        if isinstance(emb[0], list) and len(emb[0]) == 768:
            return emb[0]
        return [0.0] * 768
    except Exception:
        return [0.0] * 768


def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant code context")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    from store import search
    vec = embed_query(args.query)
    if all(v == 0 for v in vec):
        print(json.dumps({"error": "embed failed"}))
        sys.exit(1)

    results = search(args.db, vec, top_k=args.top_k)
    for text, path, score in results:
        print(json.dumps({
            "text": text[:300],
            "path": path,
            "score": round(float(score), 4)
        }))


if __name__ == "__main__":
    main()
