#!/usr/bin/env python3
"""
embed.py — Ollama embeddings via /api/embed with token-aware chunking.

Fixes the Octopus bug: nomic-embed-text runtime context = 2048 tokens (~1500 chars).
We cap chunks at EMBED_CHUNK_SIZE (default 1500 chars) to stay well within limit.
"""
import os
import sys
import json
import time
import argparse
import requests
from typing import List

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_CHUNK_SIZE = int(os.environ.get("EMBED_CHUNK_SIZE", "1500"))


def chunk_text(text: str, chunk_size: int = EMBED_CHUNK_SIZE, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks, respecting code structure."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    # Try to split on code boundaries first (function, class, newline blocks)
    separators = ["\n\n", "\n", "## ", "// ", "/* ", "class ", "def ", "function ", "; ", "}"]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Prefer a separator near the end of the chunk
        for sep in separators:
            split_point = text.rfind(sep, start + chunk_size // 2, end)
            if split_point > start:
                end = split_point + len(sep)
                break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks


def embed_with_ollama(chunks: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Send chunks to Ollama /api/embed endpoint.
    Returns list of embedding vectors (768-dim for nomic-embed-text).
    Retries on transient failures.
    """
    url = f"{OLLAMA_BASE}/api/embed"
    vectors = []
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            vectors.append([])
            continue
        
        payload = {"model": model, "input": chunk}
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    emb = data.get("embeddings", [[]])
                    if isinstance(emb, list) and len(emb) > 0:
                        vectors.append(emb[0] if isinstance(emb[0], list) else emb)
                    else:
                        vectors.append([])
                    break
                elif resp.status_code == 400:
                    # Chunk too long — split it and recurse
                    print(f"  chunk {i} too long ({len(chunk)} chars), splitting...", file=sys.stderr)
                    sub_chunks = chunk_text(chunk, chunk_size=EMBED_CHUNK_SIZE // 2, overlap=50)
                    sub_vecs = embed_with_ollama(sub_chunks, model)
                    # Average the sub-embeddings
                    valid = [v for v in sub_vecs if v]
                    if valid:
                        import numpy as np
                        avg = np.mean(valid, axis=0).tolist()
                        vectors.append(avg)
                    else:
                        vectors.append([])
                    break
                else:
                    print(f"  attempt {attempt+1} → HTTP {resp.status_code}: {resp.text[:100]}", file=sys.stderr)
                    time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                print(f"  timeout on chunk {i}, attempt {attempt+1}/3", file=sys.stderr)
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"  error on chunk {i}: {e}", file=sys.stderr)
                time.sleep(1)
        else:
            print(f"  FAILED chunk {i} after 3 attempts, skipping", file=sys.stderr)
            vectors.append([])
    
    return vectors


def embed_file(file_path: str) -> List[tuple]:
    """
    Read a file, chunk it, embed each chunk.
    Returns list of (chunk_text, embedding_vector).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"  skip {file_path}: {e}")
        return []
    
    chunks = chunk_text(text)
    if not chunks:
        return []
    
    # Batch embed all chunks
    vectors = embed_with_ollama(chunks)
    
    results = []
    for chunk, vec in zip(chunks, vectors):
        if vec:  # Only keep chunks that embedded successfully
            results.append((chunk, vec))
    
    return results


if __name__ == "__main__":
    # CLI: embed a list of text strings (one per line from stdin)
    parser = argparse.ArgumentParser(description="Embed text chunks via Ollama")
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--text", help="Single text to embed")
    args = parser.parse_args()
    
    if args.text:
        chunks = [args.text]
    else:
        import sys
        chunks = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    
    vectors = embed_with_ollama(chunks, model=args.model)
    for vec in vectors:
        if vec:
            print(json.dumps(vec))
        else:
            print()
