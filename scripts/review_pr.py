#!/usr/bin/env python3
"""
review_pr.py — Orchestrator: fetch PR diff → call review.py.

Usage:
  python3 scripts/review_pr.py --repo ChonSong/hermes-webui --pr 42 --db index.db
"""
import os, sys, subprocess, argparse, tempfile

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def gh(args, capture=True):
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"gh {' '.join(args)} failed: {result.stderr}", file=sys.stderr)
        return None if capture else False
    return result.stdout if capture else True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    parser.add_argument("--db", required=True, help="sqlite-vec DB path")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_REVIEW_MODEL", "qwen2.5-coder:7b"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--index-on-missing", action="store_true", help="Run init_index if DB is empty")
    args = parser.parse_args()

    print(f"Fetching diff for {args.repo} PR #{args.pr}...")

    # Get diff
    diff_text = gh(["pr", "diff", args.repo, str(args.pr)])
    if diff_text is None:
        sys.exit(1)
    if not diff_text.strip():
        print("Empty diff — nothing to review.")
        sys.exit(0)

    # Check DB exists and has data
    if not os.path.exists(args.db):
        if args.index_on_missing:
            print(f"DB {args.db} not found. Running init_index...")
            init_result = subprocess.run(
                ["python3", os.path.join(os.path.dirname(__file__), "init_index.py"),
                 "--local", os.getcwd(), "--db", args.db],
                capture_output=True, text=True
            )
            print(init_result.stdout)
            if init_result.returncode != 0:
                print(f"init_index failed: {init_result.stderr}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"DB {args.db} not found. Run init_index first or use --index-on-missing.", file=sys.stderr)
            sys.exit(1)

    # Save diff to temp file (review.py reads it)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
        f.write(diff_text)
        diff_file = f.name

    try:
        # Call review.py
        review_py = os.path.join(os.path.dirname(__file__), "review.py")
        cmd = [
            "python3", review_py,
            "--diff-file", diff_file,
            "--db", args.db,
            "--repo", args.repo,
            "--pr", str(args.pr),
            "--model", args.model,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            sys.exit(result.returncode)
    finally:
        os.unlink(diff_file)


if __name__ == "__main__":
    main()
