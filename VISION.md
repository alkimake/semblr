# Flashback — Project Vision

## Elevator Pitch

Current AI agent sessions are reborn every time you start one. Every chat, every context, every insight — gone. Within a session, context decays through lossy summarization the moment you exceed the window. Flashback replaces this with **semantic context assembly**: rounds are stored permanently as individual files, embedded, and retrieved by relevance — not recency. The context is always roughly the same size, always the most relevant to what you're working on.

## Core Problems Solved

### 1. Session Amnesia
Every new session is tabula rasa. Prior work, decisions, dead ends — all lost.

**Solution:** A persistent repository of rounds (each user prompt + model response sequence). Embedded and retrievable by semantic similarity.

### 2. Context Decay
Within a session, as context grows past the window, summarization compresses it. What survives is a rough sketch, not the details.

**Solution:** Dynamic context assembly from the round repository. Each round gets a fixed budget. Always the most relevant rounds, never lossy summarization.

## Architecture

### Round Repository
- Each round is saved as an individual file
- A "round" = one user message → full model response sequence (thinking, tool calls, tool results, final answer, etc.)
- Files are append-only, never modified after creation

### Embedding Index
- Each round gets two embeddings: prompt vector + response vector
- Stored as an append-only mapping: `vector → filepath`
- Future: more embedding strategies for different retrieval needs

### Context Assembly (on each prompt)
1. Embed the incoming user prompt
2. Compute numpy distance against all stored vectors
3. Sort by distance (ascending — closest first)
4. Pull in rounds until the token budget is reached
5. Construct context: system prompt + retrieved rounds + current prompt
6. Send to the model

### Context Budget
- A percentage of the model's max context size (e.g., 50%)
- Room reserved for system prompt, current prompt, and model response
- This mirrors the approximate approach used by current LLM agents

### Debug/Quality Logging
- Each context construction is logged: which files were selected, in what order, how close they were
- Enables review and improvement of retrieval quality over time

## Technology

- **Platform:** [pi coding agent](https://pi.dev) — extensions
- **Agent loop, tools, TUI, model abstraction:** Inherited from pi
- **Round repository:** Flat files on disk
- **Embeddings:** Any embedding API (tbd)
- **Vector index:** Flat file + numpy
- **Context assembly:** Extension hooks (`agent_start`, `message_end`, `agent_end`, `context`, `session_before_compact`, `session_compact`)

## Design Principles

1. **Semantic over sequential.** Relevance beats recency.
2. **Fixed-size context.** Predictable, cache-friendly windows.
3. **Append-only repository.** Never modify a saved round. Build better retrieval instead.
4. **Observable retrieval.** Log every context construction for quality iteration.
5. **Pluggable embeddings.** Multiple vector strategies over time.
6. **Framework-lean.** Own the context logic. Borrow the agent loop.

## Roadmap

### Phase 1 — MVP (Proof that extension hooks work)
- [ ] Pi extension that wipes context clean on every round ("total amnesia")
- [ ] Verify: model only sees current prompt, no prior conversation
- [ ] Verify: no compaction fires
- [ ] Verify: tools still work
- [ ] Verify: TUI still works

### Phase 2 — Round Repository
- [ ] Save each completed round to a file on disk
- [ ] Structure: `rounds/<id>.json` with prompt, response sequence, timestamps
- [ ] Embed prompt and response separately
- [ ] Maintain vector index file

### Phase 3 — Retrieval
- [ ] On `context` hook, embed incoming prompt
- [ ] Query index by distance
- [ ] Assemble context from closest rounds up to token budget
- [ ] Inject into the agent as replaced messages

### Phase 4 — Quality & Iteration
- [ ] Log context construction decisions
- [ ] Experiment with different embedding models
- [ ] Experiment with prompt vs response vector weighting
- [ ] Measure retrieval quality (precision/recall against manual ideal)

### Phase 5 — Advanced
- [ ] Multiple embedding strategies per round
- [ ] Hybrid retrieval (semantic + keyword/BM25)
- [ ] User-directed context curation
- [ ] Cross-project round repository sharing
