/**
 * contexter — MVP: Total Amnesia
 *
 * Wipes ALL accumulated context between turns.
 * The LLM sees: system prompt + ALL messages from the current turn
 * (user prompt, assistant responses, tool calls, tool results).
 *
 * This proves that pi's extension hooks give us full control
 * over context assembly. If this works, we layer on retrieval.
 *
 * Approach:
 *   In `context`, we find the last user message (current prompt),
 *   keep the system prompt + everything from that user message onward.
 *   This preserves the current turn's tool calls and results
 *   while wiping all prior turns.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  // ──────────────────────────────────────────────
  // 1. WIPE CROSS-TURN CONTEXT, KEEP CURRENT TURN
  // ──────────────────────────────────────────────
  pi.on("context", async (event, ctx) => {
    const { messages } = event;

    const systemMsg = messages.find((m) => m.role === "system");

    // Find the last user message — that's the current prompt
    const lastUserIdx = messages.reduce((last, m, i) =>
      m.role === "user" ? i : last, -1);

    const kept: typeof messages = [];

    // Keep system prompt
    if (systemMsg) kept.push(systemMsg);

    // Keep everything from the current user message onward
    // (preserves tool calls, tool results, assistant responses from this turn)
    if (lastUserIdx >= 0) {
      kept.push(...messages.slice(lastUserIdx));
    }

    const wiped = messages.length - kept.length;
    ctx.ui.setStatus(
      "contexter",
      `🧠 amnesia: wiped ${wiped} msgs from prior turns, kept ${kept.length}`,
    );

    return { messages: kept };
  });

  // ──────────────────────────────────────────────
  // 2. CANCEL COMPACTION
  // ──────────────────────────────────────────────
  pi.on("session_before_compact", async (_event, _ctx) => {
    return { cancel: true };
  });

  // ──────────────────────────────────────────────
  // 3. LOG TURN COMPLETION (placeholder for repository)
  // ──────────────────────────────────────────────
  pi.on("turn_end", async (event, ctx) => {
    ctx.ui.setStatus(
      "contexter",
      `✅ turn ${event.turnIndex + 1} complete`,
    );
  });

  // ──────────────────────────────────────────────
  // 4. STARTUP NOTIFICATION
  // ──────────────────────────────────────────────
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify("🧠 contexter loaded — total amnesia mode", "info");
  });
}
