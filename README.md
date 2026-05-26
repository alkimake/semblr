# Flashback

**Semantic context assembly for AI agents.** Instead of lossy summarization, Flashback stores every conversation round permanently, embeds it, and retrieves the most relevant rounds on each user prompt — by meaning, not by recency.

Runs as a [pi coding agent](https://pi.dev) extension at `.pi/extensions/flashback.ts`.

## Premise

Current AI agent sessions degrade as they accumulate context. Pi's compaction mechanism summarises past rounds to free memory, but the summaries lose detail. Flashback replaces this with a different approach:

1. **Save every round permanently.** Each user prompt + full assistant response sequence (tool calls, thinking, final answer) is saved as an individual JSON file.
2. **Embed prompt and response.** Both are sent to an embedding API (`text-embedding-3-small`) and stored as vectors in an append-only CSV index.
3. **Retrieve by relevance.** On every user prompt, the prompt is embedded and compared against all stored vectors via cosine similarity. The closest rounds are injected into context — up to a dynamic token budget.
4. **Drill-down via tools.** By default, rounds are shown as a compact numbered index. The LLM uses `get_round_details()` to expand a round and `get_tool_details()` to inspect individual tool calls within it.

The result: context that is **always roughly the same size, always the most relevant, and never lossy** — even across sessions.

## Cost

Flashback has two sources of API cost:

| Operation | Cost per invocation |
|---|---|
| **Saving a round** | 2 embedding API calls (prompt + response) |
| **Context assembly** | 1 embedding API call (the current prompt) |
| **Index search** (via `search_interactions` tool) | 1 embedding API call per search |

All embeddings go to OpenRouter → `text-embedding-3-small`. At current pricing (~$0.13/1M tokens for input, ~0.26/1M for output for text-embedding-3-small, but OpenRouter may add a small markup), the cost per embedding is on the order of fractions of a cent.

The ongoing cost is ~1 embedding per user prompt.

## Index & Session Management

Flashback stores conversation data in two areas, both outside the project tree so they survive repository moves:

### Round Storage (`FLASHBACK_ROUNDS_DIR`)

| File | Purpose |
|---|---|
| `<id>.json` | A single round: user prompt, full assistant response, tool call metadata |
| `index.csv` | Append-only vector index — one line per embedding: `base64(vector),<filepath>:prompt\|response` |

Round IDs are content-addressed (MD5 of `userPrompt + responseSequence`), so re-indexing is idempotent — same content produces the same file.

### Digest Scripts

Two scripts parse historical pi session files (JSONL format) into flashback rounds:

| Script | What it does |
|---|---|
| `scripts/digest-all.ts` | Iterates all pi session files, deduplicates against already-indexed rounds, embeds new ones in parallel (concurrency: 5) |
| `scripts/digest-session.ts` | Parses a single session file, embeds each round, appends to the vector index |

Both parse the pi session JSONL into `Round` objects containing:
- `userPrompt` — the user's text
- `responseSequence` — the assistant's full text response
- `toolCalls` — structured list of tool invocations (name, arguments, result summary)
- `turnIndex` — position within the session
- `sessionLabel` — source session directory name

### Deduplication

When `digest-all.ts` runs:
1. Loads all existing entries from `index.csv`
2. Computes the expected file path for each parsed round via content hash
3. Skips any round whose file path already appears in the index
4. Only new rounds are sent to the embedding API

This makes it safe to run repeatedly — only unindexed session data gets embedded.

### Session file format

Pi stores session data as JSONL files in its session directory. Each line is a JSON event with a `type` field. Flashback filters for `type: "message"` entries and pairs `user` messages with subsequent `assistant` and `toolResult` messages to reconstruct full rounds.

### Utility commands

```bash
# Bulk-index all historical sessions
just index

# Index a single session file
just digest-session path/to/session.jsonl

# Search the index from the command line
just query "what did we discuss about caching"
```

### Index format

The index is an append-only CSV with no schema header:

```
<base64url(JSON vector)>,<round_id>.json:prompt
<base64url(JSON vector)>,<round_id>.json:response
```

Each round produces two rows: one for the user prompt embedding, one for the assistant response embedding. The vector dimensions match the embedding model (1536 for `text-embedding-3-small`). Cosine similarity is used for retrieval.

## Known Problems

### Most-recent-round context loss (collapsed mode)
In collapsed mode (the default), every retrieved round — including the immediately preceding conversation turn — appears as a compact numbered entry. The LLM cannot directly resolve "those changes" or "it" from the previous round without calling `get_round_details()`. A mitigation injects the last round's full prompt + response (minus tool calls) as plain text, but this doesn't extend to multi-turn chains.

**Future fix:** A recency buffer that keeps the last 3–5 rounds in full, outside the indexed retrieval logic.

### Embedding API dependency
Flashback requires a working OpenRouter API key (or an alternative embedding endpoint) to function. If the API is unreachable, context assembly falls back to a no-op (no historical context injected). The extension degrades gracefully but silently.

### No local embedding fallback
Currently only one embedding model (`text-embedding-3-small` via OpenRouter) is wired. There is no local embedding option (e.g., `sentence-transformers` → ONNX → TypeScript). Adding one would eliminate the API dependency and cost for index queries.

## Quick Start

Flashback runs automatically when the extension is loaded:

```bash
# Verify it's working
pi -e .pi/extensions/flashback.ts
```

Check that the status bar shows `🧠 flashback loaded — N rounds indexed`.

### Bulk-indexing historical sessions

```bash
# Index all historical pi sessions into flashback
just index
```

This parses every JSONL session file in `~/.pi/agent/sessions/`, deduplicates against already-indexed rounds, and embeds new ones in parallel (concurrency: 5).

### Query the index

```bash
# Search the index from the command line
just query "what did we discuss about caching"
```

## Project Structure

```
.pipermanent_extension/
├── .pi/
│   └── extensions/
│       └── flashback.ts          # The extension (~1240 lines)
├── scripts/
│   ├── digest-all.ts             # Bulk-embed all historical sessions
│   ├── digest-session.ts         # Embed a single session file
│   └── test-register-tool.ts     # Tool registration test harness
├── docs/                         # (empty — extended docs not yet written)
├── VISION.md                     # Full project vision, architecture, roadmap
├── AGENTS.md                     # Project context for AI agents
├── README.md                     # This file
├── justfile                      # just command recipes
└── package.json                  # Project metadata
```

Rounds are stored in a global directory (outside the project tree) so they survive repository moves and clones.

## VISION.md

For the full vision, design principles, architecture docs, and roadmap (with implemented items checked off), see [VISION.md](VISION.md).

## License

MIT
