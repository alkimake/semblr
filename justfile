# ── flashback ─────────────────────────────────────────────────────────────────
# See VISION.md for architecture and AGENTS.md for project context.

# Override via: just --set FLASHBACK_ROUNDS_DIR /custom/path index
# or set FLASHBACK_ROUNDS_DIR in the environment.
FLASHBACK_ROUNDS_DIR := env_var_or_default("FLASHBACK_ROUNDS_DIR", "")

# Index all new turns from pi session files
# Skips turns already indexed (by MD5 content hash)
index:
    OPENROUTER_API_KEY="$(pass show ai/openrouter)" \
        FLASHBACK_ROUNDS_DIR="{{FLASHBACK_ROUNDS_DIR}}" \
        npx tsx scripts/digest-all.ts

# Digest a specific session file
# Usage: just digest-session <path-to-jsonl>
digest-session path:
    FLASHBACK_ROUNDS_DIR="{{FLASHBACK_ROUNDS_DIR}}" \
        npx tsx scripts/digest-session.ts {{path}}

# Query the index with a natural language query (via pi + flashback extension)
# Usage: just query "<your question>"
query query *args:
    echo "search interactions for '{{query}}'" | pi --print --no-builtin-tools -e .pi/extensions/flashback.ts
