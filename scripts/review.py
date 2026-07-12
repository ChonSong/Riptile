#!/usr/bin/env python3
"""
review.py — LLM code review using Ollama + retrieved context.

Reads diff from stdin or --diff-file, retrieves relevant context from numpy-vector store,
builds prompt, calls Ollama (qwen2.5-coder:7b), parses findings into structured GitHub comment.

Usage:
  gh pr diff 42 | python3 review.py --repo owner/repo --pr 42 --db index.db
  python3 review.py --diff-file /tmp/pr.diff --db index.db --repo owner/repo --pr 42
"""
import os, sys, json, argparse, tempfile, requests, time
from pathlib import Path

OLLAMA_BASE    = os.environ.get("OLLAMA_BASE_URL",    "http://localhost:43311")
OLLAMA_EMBED   = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
REVIEW_MODEL   = os.environ.get("OLLAMA_REVIEW_MODEL", "qwen2.5-coder:7b")
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "8"))
REVIEW_LOG     = os.environ.get("REVIEW_LOG", os.path.expanduser("~/.hermes/pr-review/review.log"))


# ── Ollama calls ────────────────────────────────────────────────────────────

def embed_query(text: str) -> list:
    """Embed text using Ollama /api/embed (singular, v0.31)."""
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


def llm_review(prompt: str, model=REVIEW_MODEL) -> str:
    """Call Ollama /api/generate for code review."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        }
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=300
        )
        if resp.status_code != 200:
            return f"[HTTP {resp.status_code}] {resp.text[:200]}"
        return resp.json().get("response", "[no response]")
    except requests.exceptions.Timeout:
        return "[timeout after 120s]"
    except Exception as e:
        return f"[error: {e}]"


# ── Context retrieval ───────────────────────────────────────────────────────

def retrieve_context(query: str, db_path: str, top_k: int = RETRIEVE_TOP_K):
    """Top-K vector search using the numpy store."""
    sys.path.insert(0, str(Path(__file__).parent))
    from store import search
    vec = embed_query(query)
    if all(v == 0 for v in vec):
        return []
    return search(db_path, vec, top_k=top_k)


# ── Review logic ────────────────────────────────────────────────────────────

def build_prompt(diff_content: str, repo: str, pr_num: int,
                 context_results: list) -> str:
    """Build the code-review prompt."""
    ctx_block = ""
    if context_results:
        ctx_lines = [
            "## Relevant codebase context:"
        ]
        for i, (text, path, score) in enumerate(context_results, 1):
            ctx_lines.append(f"\n### [{i}] {path} (score={score:.3f})")
            ctx_lines.append(f"```\n{text[:500]}\n```")
        ctx_block = "\n".join(ctx_lines)
    else:
        ctx_block = "(no relevant context found in index)"

    return f"""You are reviewing PR #{pr_num} in {repo}.

## Changed files diff
```diff
{diff_content[:8000]}
```

{ctx_block}

## Instructions
Review the diff for:
1. Bugs or logic errors
2. Security issues (injection, auth bypass, data exposure)
3. Performance problems
4. Missing error handling
5. Breaking changes or regressions

Be concise. If the change looks good, say so. Format findings as:
**Finding**: description
**Severity**: Low/Medium/High
**Location**: file:line or general area
**Suggestion**: optional fix
"""


def parse_review(response: str) -> dict:
    """Parse LLM review into structured format."""
    findings = []
    current = {}
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("**Finding**"):
            if current:
                findings.append(current)
            current = {"finding": line.split("**Finding**", 1)[1].strip(), "severity": "Medium"}
        elif line.startswith("**Severity**"):
            current["severity"] = line.split("**Severity**", 1)[1].strip()
        elif line.startswith("**Location**"):
            current["location"] = line.split("**Location**", 1)[1].strip()
        elif line.startswith("**Suggestion**"):
            current["suggestion"] = line.split("**Suggestion**", 1)[1].strip()
    if current:
        findings.append(current)

    return {
        "raw": response,
        "findings": findings,
        "summary": "N findings" if findings else "No issues found — LGTM!"
    }


def run_review(diff_content: str, repo: str, pr_num: int,
               db_path: str, model=REVIEW_MODEL):
    """Full review pipeline."""
    print(f"[review] retrieving context for {repo} PR #{pr_num}...", file=sys.stderr)
    t0 = time.time()
    ctx = retrieve_context(diff_content, db_path)
    print(f"[review] retrieved {len(ctx)} context chunks in {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"[review] building prompt...", file=sys.stderr)
    prompt = build_prompt(diff_content, repo, pr_num, ctx)
    print(f"[review] calling LLM ({model})...", file=sys.stderr)
    t1 = time.time()
    response = llm_review(prompt, model=model)
    print(f"[review] LLM responded in {time.time()-t1:.1f}s", file=sys.stderr)

    result = parse_review(response)

    # Log
    os.makedirs(os.path.dirname(REVIEW_LOG) or ".", exist_ok=True)
    with open(REVIEW_LOG, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"PR: {repo}#{pr_num}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Context chunks: {len(ctx)}\n")
        f.write(f"Response time: {time.time()-t1:.1f}s\n")
        f.write(f"\n{response}\n")

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM code review")
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--db", default=os.environ.get("REVIEW_DB", "index.db"))
    parser.add_argument("--diff-file", help="File containing diff (default: stdin)")
    parser.add_argument("--model", default=REVIEW_MODEL)
    parser.add_argument("--context-k", type=int, default=RETRIEVE_TOP_K)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    # Read diff
    if args.diff_file:
        with open(args.diff_file) as f:
            diff_content = f.read()
    else:
        diff_content = sys.stdin.read()

    if not diff_content.strip():
        print("No diff content. Nothing to review.", file=sys.stderr)
        sys.exit(0)

    print(f"[review] diff size: {len(diff_content)} chars", file=sys.stderr)
    result = run_review(diff_content, args.repo, args.pr, args.db, model=args.model)

    # Output
    print("\n## Review Summary")
    print(result["summary"])
    if result["findings"]:
        print(f"\n### Findings ({len(result['findings'])})")
        for i, f in enumerate(result["findings"], 1):
            print(f"\n{i}. **{f['finding']}**")
            print(f"   Severity: {f.get('severity','Medium')}")
            if f.get("location"):
                print(f"   Location: {f['location']}")
            if f.get("suggestion"):
                print(f"   Suggestion: {f['suggestion']}")
    else:
        print("\nNo issues found.")
    print(f"\n(Full response logged to {REVIEW_LOG})")


if __name__ == "__main__":
    main()
