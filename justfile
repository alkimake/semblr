# ── flashback ─────────────────────────────────────────────────────────────────
# See VISION.md for architecture and AGENTS.md for project context.

# Index all new turns from pi session files
# Skips turns already indexed (by MD5 content hash)
index:
    OPENROUTER_API_KEY="$(pass show ai/openrouter)" npx tsx scripts/digest-all.ts

# Digest a specific session file
# Usage: just digest-session <path-to-jsonl>
digest-session path:
    npx tsx scripts/digest-session.ts {{path}}

# Query the index with a natural language query
# Usage: just query <query> [--k 10] [--budget 4000]
query query *args:
    npx tsx scripts/query-index.ts "{{query}}" {{args}}
