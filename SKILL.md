---
name: pr-review
description: "Local-first PR reviewer: GitHub PR → numpy-vector SQLite store → Ollama embed → retrieve context → LLM review → GitHub comments. Zero infra, polling trigger via gh pr list."
version: 0.2.0
author: Hermes Agent for ChonSong
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [code-review, github, numpy-vector, ollama, local-embeddings, pr-agent]
    related_skills: [codebase-ingestion, github-code-review, requesting-code-review]
prerequisites:
  python_packages: [numpy, scipy, requests]
  system_requirements:
    - Ollama running with nomic-embed-text and qwen2.5-coder (or other review model)
    - gh CLI authenticated (gh auth status)
  environment_variables:
    OLLAMA_BASE_URL: "http://localhost:43311"
    OLLAMA_EMBED_MODEL: "nomic-embed-text"
    OLLAMA_REVIEW_MODEL: "qwen2.5-coder:7b"
    REVIEW_DB: "~/.hermes/pr-review/index.db"
    REVIEW_LOG: "~/.hermes/pr-review/review.log"
    RETRIEVE_TOP_K: "8"
---

# PR Review — Local-First Code Review Plugin

Reviews GitHub PRs using a local Ollama LLM with sqlite-vec semantic search over
your codebase. No Docker Compose, no external vector DB, no cloud API keys.

## Architecture

```
PR opened/updated
  → cron job (gh pr list polling every N minutes)
  → review_pr.py (per PR)
      1. gh pr diff  → get changed files + patch
      2. index_repo.py (first time or on demand) → chunk + embed + store in sqlite-vec
      3. retrieve_context.py → top-K relevant chunks from sqlite-vec for each changed file
      4. review.py → build LLM prompt (diff + context) → call Ollama → parse findings
      5. gh pr comment → post structured inline comments
```

## Trigger Modes

### Mode A: Cron Polling (default — zero infra)
```bash
# Run every 5 minutes via Hermes cron
cronjob --action=create \
  --prompt="$(cat scripts/review_pr.py)" \
  --schedule="*/5 * * * *" \
  --name="pr-review" \
  --deliver=local
```
Uses `gh pr list --state=open --author=@me` to find PRs opened by the authenticated user.

### Mode B: On-Demand (per PR)
```bash
python3 scripts/review_pr.py --repo owner/repo --pr 123
```

### Mode C: Webhook Relay (near-real-time)
GitHub App webhook → Cloudflare Worker → Hermes API → triggers review.
Requires a Cloudflare Worker as relay (not built into this skill).

## Components

| Script | Purpose |
|--------|---------|
| `scripts/init_index.py` | One-time: clone/walk repo → chunk → embed → store in sqlite-vec |
| `scripts/embed.py` | Embed a list of text chunks via Ollama nomic-embed-text |
| `scripts/store.py` | Upsert/delete vectors in sqlite-vec |
| `scripts/retrieve.py` | Top-K cosine similarity search in sqlite-vec |
| `scripts/review.py` | LLM review: diff + context → structured findings → gh pr comment |
| `scripts/review_pr.py` | Orchestrator: tie all pieces together |
| `scripts/poll.py` | Cron entry point: find open PRs, call review_pr.py |

## Setup

### 1. Verify prerequisites
```bash
gh auth status
curl -s http://localhost:11434/api/tags  # Ollama running
python3 -c "import sqlite_vec; print('sqlite-vec OK')"
```

### 2. Configure environment
```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBED_MODEL=nomic-embed-text
export OLLAMA_REVIEW_MODEL=qwen2.5-coder:7b
export REVIEW_DB=~/.hermes/pr-review/index.db
```

### 3. Index a repository (one-time)
```bash
python3 scripts/init_index.py \
  --repo ChonSong/hermes-webui \
  --branch main \
  --db ~/.hermes/pr-review/hermes-webui.db
```

### 4. Test a review
```bash
python3 scripts/review_pr.py \
  --repo ChonSong/hermes-webui \
  --pr 42 \
  --db ~/.hermes/pr-review/hermes-webui.db
```

### 5. Set up polling cron
```bash
hermes cron create \
  --name "pr-review" \
  --prompt "Run python3 /home/sc/.hermes/skills/pr-review/scripts/poll.py --repo ChonSong/hermes-webui --db /home/sc/.hermes/pr-review/hermes-webui.db" \
  --schedule "*/5 * * * *"
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_REVIEW_MODEL` | `qwen2.5-coder:7b` | LLM for review |
| `EMBED_CHUNK_SIZE` | `1500` | Max chars per chunk (~375-500 tokens) |
| `RETRIEVE_TOP_K` | `8` | Context chunks to retrieve per changed file |
| `POLL_INTERVAL_MINUTES` | `5` | Cron polling interval |

## Chunk Size Rationale

nomic-embed-text runtime context window is **2048 tokens** (not 8192 as sometimes
reported — the model card says 8192 but Ollama's runtime caps at 2048 for this model).

`EMBED_CHUNK_SIZE=1500` chars ≈ 375-500 tokens, well within the 2048 limit.
Octopus used 24,000 chars — that caused 400 errors from Ollama truncation.

## Severity Ratings

Findings are posted as GitHub PR comments with emoji severity:

| Rating | Emoji | Meaning |
|--------|-------|---------|
| Critical | 🔴 | Security vuln, data loss risk, broken functionality |
| High | 🟠 | Significant bug, major design issue |
| Medium | 🟡 | Code quality, maintainability, style |
| Low | 🔵 | Suggestions, improvements, best practices |

## Output Format (PR Comment)

```markdown
## 🔍 Hermes Review

**Model:** qwen2.5-coder:7b via Ollama  
**Review:** 3 files changed, 47 lines diff

### 🔴 Critical
- **`apps/auth.py:45`** — SQL injection: `id` interpolated directly into query
  ```python
  cursor.execute(f"SELECT * FROM users WHERE id = {id}")  # DANGEROUS
  ```
  → Use parameterized: `cursor.execute("...WHERE id = ?", (id,))`

### 🟡 Medium
- **`apps/utils.py:12`** — `requests` call without timeout — will hang on slow upstream

---

*Reviewed by Hermes Agent (sqlite-vec + nomic-embed-text + Ollama)*
```

## Comparison to Octopus

| | Octopus | pr-review skill |
|---|---|---|
| Vector DB | Qdrant (separate container) | sqlite-vec (single .db file) |
| Indexing | Full reindex every time | Incremental (index once, diff retrieves) |
| Webhook | FastAPI server required | Polling (no server) |
| DB for metadata | PostgreSQL | None (gh CLI only) |
| Infrastructure | Docker Compose (5+ containers) | Just Ollama |
| Context per file | 8 top-K chunks | 8 top-K chunks |
| Chunk size | 24,000 chars (broken) | 1,500 chars (fixed) |

## Comparison to PR-Agent (Codium)

| | PR-Agent | pr-review skill |
|---|---|---|
| Model flexibility | Any via litellm | Ollama only |
| Deployment | GitHub App / Action / CLI | CLI + cron |
| Infrastructure | GitHub App server | None |
| Vector search | None (prompt-based) | sqlite-vec RAG |
| Self-hosted | Via GitHub Action | Full local |
