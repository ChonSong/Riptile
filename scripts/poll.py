#!/usr/bin/env python3
"""
poll.py — Cron entry point: find open PRs opened by the authenticated user and trigger reviews.

Usage:
  python3 scripts/poll.py --repo ChonSong/hermes-webui --db ~/.hermes/pr-review/hermes-webui.db

This script is designed to run as a Hermes cron job every 5 minutes.
"""
import os, sys, subprocess, argparse, json, time
from datetime import datetime, timedelta

def gh(args, capture=True):
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"gh {' '.join(args)}: {result.stderr.strip()}", file=sys.stderr)
        return None if capture else False
    return result.stdout if capture else True


def get_open_prs(repo):
    """Return list of open PRs by the authenticated user."""
    output = gh(["pr", "list", "--repo", repo, "--state=open", "--author=@me", "--json=number,title,updatedAt"])
    if output is None:
        return []
    try:
        prs = json.loads(output)
        return prs
    except:
        return []


def is_recent_updated(updated_at):
    """True if PR was updated in the last 30 minutes (avoid re-reviewing stable PRs)."""
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        return datetime.now(updated.tzinfo) - updated < timedelta(minutes=30)
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--review-script", default=None)
    args = parser.parse_args()

    prs = get_open_prs(args.repo)
    if not prs:
        print(f"No open PRs found for {args.repo}")
        sys.exit(0)

    review_script = args.review_script or os.path.join(os.path.dirname(__file__), "review_pr.py")
    reviewed = 0
    errors = 0

    for pr in prs:
        if is_recent_updated(pr["updatedAt"]):
            print(f"Skipping PR #{pr['number']} — updated recently: {pr['title']}")
            continue

        print(f"Reviewing PR #{pr['number']}: {pr['title']}")
        result = subprocess.run(
            ["python3", review_script, "--repo", args.repo, "--pr", str(pr["number"]), "--db", args.db],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            reviewed += 1
        else:
            errors += 1
            print(f"  Error: {result.stderr[:200]}", file=sys.stderr)
        time.sleep(2)  # rate limit

    print(f"\nPoll complete: {reviewed} reviewed, {errors} errors")


if __name__ == "__main__":
    main()
