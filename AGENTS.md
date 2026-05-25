# Contexter

Next-generation context management for AI agents. See VISION.md for full details.

## MVP: Total Amnesia Testing

The extension lives at `.pi/extensions/contexter-amnesia.ts`. It wipes all conversation
context on every turn — the LLM sees only the system prompt + current user prompt.

To test:
```bash
pi -e .pi/extensions/contexter-amnesia.ts
```

## Project Structure

- `VISION.md` — project vision, architecture, roadmap
- `.pi/extensions/contexter-amnesia.ts` — MVP extension (total amnesia)
- `.pi/turns/` — turn repository (Phase 2+)
