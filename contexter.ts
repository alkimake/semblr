/**
 * contexter — Retrieval-Augmented Context Assembly
 *
 * At turn_end: save the completed turn to .pi/turns/ and embed it.
 * At context: embed the current user prompt, query the vector index,
 *             inject the top-matching turns as context for the LLM.
 *
 * Replaces contexter-amnesia.ts — no wiping, just smart retrieval.
 */

import type { ExtensionAPI, ContextEvent } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import * as fs from "node:fs";
import * as crypto from "node:crypto";
import { spawnSync } from "node:child_process";

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const PROJECT_ROOT = "/home/vedat/work/personal/contexter";
const TURNS_DIR = `${PROJECT_ROOT}/.pi/turns`;
const INDEX_PATH = `${TURNS_DIR}/index.csv`;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";

const CONTEXT_BUDGET_RATIO = 0.5; // 50% of model context window for historical turns

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function normalize(v: number[]): number[] {
  const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return mag === 0 ? v : v.map((x) => x / mag);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function extractText(content: Array<{ type: string; text?: string }>): string {
  return content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text!)
    .join(" ");
}

// ─────────────────────────────────────────────
// Index CSV format:
//   filePath,vectorElement1,vectorElement2,...
//   (no header row)
//   filePath includes :prompt or :response suffix
// ─────────────────────────────────────────────

interface IndexEntry {
  filePath: string;
  vector: number[];
}

function loadIndex(): IndexEntry[] {
  if (!fs.existsSync(INDEX_PATH)) return [];
  const raw = fs.readFileSync(INDEX_PATH, "utf-8").trim();
  if (!raw) return [];
  return raw.split("\n").map((line) => {
    const comma = line.indexOf(",");
    const b64 = line.slice(0, comma);
    const filePath = line.slice(comma + 1);
    let vector: number[];
    try {
      // Format: base64url,fileName  where base64 is JSON.stringify of the vector
      const decoded = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
      vector = Array.isArray(decoded) ? decoded : [];
    } catch {
      // Fallback: try parsing as comma-separated numbers (old format)
      const vecStr = line.slice(comma + 1);
      const parts = vecStr.split(",");
      // If the first element looks like a float, it's the old format (filePath,vec1,vec2,...)
      if (parts.length > 1 && !isNaN(Number(parts[0]))) {
        vector = parts.map(Number);
      } else {
        vector = [];
      }
    }
    return { filePath, vector };
  });
}

function readTurnFile(
  filePath: string,
): { userPrompt: string; responseSequence: string; turnIndex: number; userTimestamp?: number } | null {
  // filePath may be "turn-xxx.json:prompt" or "turn-xxx.json:response"
  // strip the :prompt/:response suffix to get the actual file
  const actualFile = filePath.replace(/:prompt$|:response$/, "");
  const fullPath = `${TURNS_DIR}/${actualFile}`;
  if (!fs.existsSync(fullPath)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(fullPath, "utf-8"));
    return {
      userPrompt: data.userPrompt ?? "",
      responseSequence: data.responseSequence ?? "",
      turnIndex: data.turnIndex ?? 0,
      userTimestamp: data.userTimestamp,
    };
  } catch {
    return null;
  }
}

let lastTurnFileName: string | null = null; // tracks the most recent saved turn (process-local)
let currentUserPrompt: string | null = null; // cached from context hook for turn_end
let currentUserMessageId: string | null = null;
let currentTurnAccumulatedText: string[] = []; // accumulated assistant text across a single turn

// Compaction integration: track turns per user message ID for matching compacted messages
// to our turn files. Persisted to disk as a JSON map: userMessageId → turnFileName
const COMPACTION_CHAIN_PATH = `${TURNS_DIR}/compaction-chain.json`;
let compactionChain: Array<{ userMessageId: string; turnFileName: string; userTimestamp: number }> = [];
// Stash between session_before_compact and session_compact
let pendingCompactionTurnFiles: string[] | null = null;

async function getApiKey(): Promise<string | null> {
  // 1. Environment variable
  const envKey = process.env.OPENROUTER_API_KEY;
  if (envKey) return envKey;

  // 2. Pass store
  try {
    const result = spawnSync("pass", ["show", "ai/openrouter"], {
      timeout: 5000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    if (result.status === 0) {
      const key = result.stdout.toString().trim();
      if (key) return key;
    }
  } catch {
    // pass not available, fall through
  }

  return null;
}

async function embedText(text: string, apiKey: string): Promise<number[]> {
  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: text,
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Embedding API error ${response.status}: ${errText}`);
  }

  const data = (await response.json()) as {
    data: Array<{ embedding: number[] }>;
  };
  return data.data[0].embedding;
}

function createTurnFilePath(userPrompt: string, responseText: string): string {
  const content = userPrompt + responseText;
  const hash = crypto.createHash("md5").update(content).digest("hex");
  return `turn-${hash}.json`;
}

function appendToIndex(filePath: string, vector: number[]) {
  const line = `${filePath},${vector.join(",")}\n`;
  fs.mkdirSync(TURNS_DIR, { recursive: true });
  fs.appendFileSync(INDEX_PATH, line);
}

// ─────────────────────────────────────────────
// Compaction chain — persistent mapping of userMessageId → turnFileName
// Used by session_before_compact / session_compact to build summary turns
// that reference the original turn files being compacted.
// ─────────────────────────────────────────────

function loadCompactionChain(): Array<{ userMessageId: string; turnFileName: string; userTimestamp: number }> {
  try {
    if (fs.existsSync(COMPACTION_CHAIN_PATH)) {
      return JSON.parse(fs.readFileSync(COMPACTION_CHAIN_PATH, "utf-8"));
    }
  } catch { /* ignore */ }
  return [];
}

function saveCompactionChain() {
  try {
    fs.mkdirSync(TURNS_DIR, { recursive: true });
    fs.writeFileSync(COMPACTION_CHAIN_PATH, JSON.stringify(compactionChain, null, 2));
  } catch { /* ignore */ }
}

// ─────────────────────────────────────────────
// Extension
// ─────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // ────────────────────────────────────────────
  // 1. context — assemble context from turn repository
  // ────────────────────────────────────────────
  pi.on("context", async (event: ContextEvent, ctx) => {
    const { messages } = event;

    // --- Extract system prompt + current turn messages ---
    // We strip all prior turns to prevent conversation bloat.
    // The retrieved historical context replaces the prior conversation.
    // Current turn = everything from the last user message onward
    // (includes assistant responses, tool calls, tool results in-flight).
    const systemMsg = messages.find(
      (m) => m.role === "system" || m.role === "developer",
    ) ?? null;
    const lastUserIdx = messages.reduce((last, m, i) =>
      m.role === "user" ? i : last, -1);
    const currentMessages = lastUserIdx >= 0
      ? [...messages].slice(lastUserIdx)
      : [...messages];

    // --- Get the current user prompt (last user message) ---
    const userMessages = currentMessages.filter((m) => m.role === "user");
    if (userMessages.length === 0) return { messages };

    // Extract user prompt text — content may be a string or an array of content blocks
    const lastUserContent = userMessages[userMessages.length - 1].content;
    let userPrompt: string;
    if (typeof lastUserContent === "string") {
      userPrompt = lastUserContent.split(" ").slice(0, 200).join(" ");
    } else if (Array.isArray(lastUserContent)) {
      userPrompt = extractText(lastUserContent);
    } else {
      return { messages };
    }

    // Cache for turn_end — the full (untruncated) prompt
    currentUserPrompt = typeof lastUserContent === "string"
      ? lastUserContent
      : Array.isArray(lastUserContent)
        ? extractText(lastUserContent)
        : null;
    currentUserMessageId = (userMessages[userMessages.length - 1] as Record<string, unknown>).id as string ?? null;

    try {
      const apiKey = await getApiKey();
      if (!apiKey) return { messages };

      // Embed the user prompt
      const queryVec = normalize(await embedText(userPrompt, apiKey));

      // Load and score the index
      const index = loadIndex();
      if (index.length === 0) return { messages };

      const scored = index
        .map((entry) => ({
          ...entry,
          similarity: cosineSimilarity(queryVec, entry.vector),
        }))
        .sort((a, b) => b.similarity - a.similarity);
      const bestScore = scored.length > 0 ? scored[0].similarity : 0;

      // --- Dynamic budget ---
      const MIN_SIMILARITY = 0.30;
      const minBudget = 2000;
      const MAX_BUDGET = Math.floor(
        CONTEXT_BUDGET_RATIO *
          (event.contextWindowSize ?? 128_000),
      );
      // Linear interpolation: at MIN_SIMILARITY → minBudget, at 1.0 → MAX_BUDGET
      const t = Math.max(
        0,
        Math.min(1, (bestScore - MIN_SIMILARITY) / (1 - MIN_SIMILARITY)),
      );
      const budgetTokens = Math.floor(
        minBudget + t * (MAX_BUDGET - minBudget),
      );

      // Group by turn file, take best score per turn
      interface TurnScore {
        data: {
          userPrompt: string;
          responseSequence: string;
          turnIndex: number;
          userTimestamp?: number;
        };
        bestScore: number;
      }
      const turnScores = new Map<string, TurnScore>();
      for (const entry of scored) {
        const turnFile = entry.filePath.replace(/:prompt$|:response$/, "");
        if (!turnFile.match(/^turn-/)) continue;
        if (turnScores.has(turnFile)) continue;
        const turnData = readTurnFile(entry.filePath);
        if (!turnData) continue;
        turnScores.set(turnFile, {
          data: turnData,
          bestScore: entry.similarity,
        });
      }

      const uniqueTurns = new Set(
        index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
      ).size;

      const scoredTurns = Array.from(turnScores.values()).sort(
        (a, b) => b.bestScore - a.bestScore,
      );

      // Select turns within budget
      const selectedTurns: TurnScore[] = [];
      let usedTokens = 0;
      const addTurn = (turn: TurnScore) => {
        selectedTurns.push(turn);
      };

      // 1. Score-based selection (below threshold stops)
      for (const turn of scoredTurns) {
        if (turn.bestScore < MIN_SIMILARITY) break;
        const turnTokens = estimateTokens(
          turn.data.userPrompt + turn.data.responseSequence,
        );
        if (usedTokens + turnTokens > budgetTokens) break;
        addTurn(turn);
        usedTokens += turnTokens;
      }

      // 2. Always add the last turn (if not already there)
      if (lastTurnFileName) {
        const lastData = readTurnFile(lastTurnFileName);
        if (lastData) {
          addTurn({ data: lastData, bestScore: 0 });
        }
      }

      if (selectedTurns.length === 0) {
        ctx.ui.setStatus(
          "contexter",
          `🧠 no relevant context (best: ${bestScore.toFixed(3)})`,
        );
        return { messages };
      }

      // Build context messages — chronological order (oldest first) for coherence
      // Chronological: prefer turnIndex within session, then userTimestamp as tiebreaker
      selectedTurns.sort((a, b) => {
        const ti = a.data.turnIndex - b.data.turnIndex;
        if (ti !== 0) return ti;
        const tsA = (a.data as Record<string, unknown>).userTimestamp as number ?? 0;
        const tsB = (b.data as Record<string, unknown>).userTimestamp as number ?? 0;
        return tsA - tsB;
      });

      // Dedup by MD5 content hash — last turn may duplicate a scored turn
      const seenHashes = new Set<string>();
      const dedupedTurns: typeof selectedTurns = [];
      for (const turn of selectedTurns) {
        const hash = crypto.createHash("md5").update(turn.data.userPrompt + turn.data.responseSequence).digest("hex");
        if (seenHashes.has(hash)) continue;
        seenHashes.add(hash);
        dedupedTurns.push(turn);
      }

      // Rebuild contextMessages from deduped turns
      // Each turn is wrapped in a clear delimiter so the model knows these
      // are historical records — not the current conversation. This prevents
      // the model from mimicking tool calls or phrasing from past turns.
      const contextMessages: Array<{ role: string; content: Array<{ type: string; text: string }> }> = [];
      for (const turn of dedupedTurns) {
        contextMessages.push({
          role: "user",
          content: [{ type: "text", text: `[HISTORICAL TURN from a past conversation — similarity: ${turn.bestScore.toFixed(3)}]
User asked: ${turn.data.userPrompt}` }],
        });
        contextMessages.push({
          role: "assistant",
          content: [{ type: "text", text: `[END OF HISTORICAL TURN — this was a past response, not the current one. Stay focused on the user's most recent message above.]
${turn.data.responseSequence}` }],
        });
      }

      ctx.ui.setStatus(
        "contexter",
        `🧠 retrieved ${dedupedTurns.length} turns (${usedTokens} tok) from ${uniqueTurns} indexed`,
      );

      return {
        messages: [
          ...(systemMsg ? [systemMsg] : []),
          ...contextMessages,
          ...currentMessages,
        ],
      };
    } catch (err) {
      ctx.ui.setStatus("contexter", `🧠 error: ${(err as Error).message}`);
    }
  });

  // ────────────────────────────────────────────
  // 2. turn_end — Save turn + embed it
  // ────────────────────────────────────────────
  pi.on("turn_end", async (event, ctx) => {
    const { turnIndex, message } = event;

    // Use the prompt cached from the context hook (avoids session file parsing)
    let userPrompt = currentUserPrompt ?? "";
    let userMessageId = currentUserMessageId ?? "";

    if (!userPrompt && !message?.content) {
      ctx.ui.setStatus("contexter", "🧠 no content to save");
      return;
    }

    // Build response text from the accumulated assistant text across the turn.
    // The `message` from `turn_end` only contains the final assistant message,
    // which may be just a tool call (no text). We use currentTurnAccumulatedText
    // which was built from ALL `message_end` assistant events in this turn.
    let responseText: string;
    if (currentTurnAccumulatedText.length > 0) {
      responseText = currentTurnAccumulatedText.join("\n\n").trim();
    } else {
      // Fallback: extract text from the turn_end message
      responseText = extractText(
        (message?.content ?? []) as Array<{ type: string; text?: string }>,
      );
    }

    // Ensure turns directory
    fs.mkdirSync(TURNS_DIR, { recursive: true });

    // Generate filename from content hash
    const turnFileName = createTurnFilePath(userPrompt, responseText);
    const turnPath = `${TURNS_DIR}/${turnFileName}`;

    // Skip if already saved (deduplication)
    if (fs.existsSync(turnPath)) {
      ctx.ui.setStatus("contexter", `🧠 turn ${turnIndex} already saved (${turnFileName})`);
      lastTurnFileName = turnFileName;
      return;
    }

    // Write turn file
    const turnData = {
      id: crypto.createHash("md5").update(userPrompt + responseText).digest("hex"),
      userPrompt,
      responseSequence: responseText,
      turnIndex,
      userTimestamp: Date.now(),
      userMessageId,
    };

    try {
      fs.writeFileSync(turnPath, JSON.stringify(turnData, null, 2));
    } catch (err) {
      ctx.ui.setStatus("contexter", `🧠 write error: ${(err as Error).message}`);
      return;
    }

    // Reset accumulated text for next turn
    currentTurnAccumulatedText = [];

    // Embed prompt and response separately
    const apiKey = await getApiKey();
    if (!apiKey) {
      ctx.ui.setStatus("contexter", "🧠 saved but not embedded (no API key)");
      lastTurnFileName = turnFileName;
      return;
    }

    try {
      const [promptVec, responseVec] = await Promise.all([
        embedText(userPrompt, apiKey),
        embedText(responseText, apiKey),
      ]);
      appendToIndex(`${turnFileName}:prompt`, promptVec);
      appendToIndex(`${turnFileName}:response`, responseVec);
      ctx.ui.setStatus(
        "contexter",
        `🧠 saved + embedded turn ${turnIndex} (${turnFileName})`,
      );
    } catch (err) {
      ctx.ui.setStatus("contexter", `🧠 embedding error: ${(err as Error).message}`);
    }

    lastTurnFileName = turnFileName;

    // Track in compaction chain for summary-turn indexing
    if (userMessageId) {
      compactionChain.push({ userMessageId, turnFileName, userTimestamp: Date.now() });
      saveCompactionChain();
    }
  });

  // ────────────────────────────────────────────
  // 3. message_end & turn_start — accumulate assistant text across a turn
  // ────────────────────────────────────────────
  // We need to track ALL assistant text in a turn, not just the final message.
  // When a turn includes tool calls, the assistant sends multiple messages:
  //   text → tool call → tool result → text → tool call → ...
  // Each assistant message is a separate `message_end` event.
  // We accumulate the text from each one into currentTurnAccumulatedText.
  pi.on("turn_start", async () => {
    currentTurnAccumulatedText = [];
  });

  pi.on("message_end", async (event, _ctx) => {
    const msg = event.message;
    if (!msg) return;

    if (msg.role === "user") {
      // New turn starting — reset accumulator for the *next* assistant response
      currentTurnAccumulatedText = [];
    } else if (msg.role === "assistant") {
      // Extract text from this assistant message
      const content = msg.content as Array<{ type: string; text?: string }> | undefined;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === "text" && block.text) {
            currentTurnAccumulatedText.push(block.text);
          }
        }
      }
    }
  });

  // ────────────────────────────────────────────
  // 4. Compaction integration — capture summary as a turn
  // ────────────────────────────────────────────
  // Instead of cancelling compaction (causes unbounded memory growth in pi's
  // internal entry chain), we let it proceed. The compacted summary becomes a
  // special turn in our index with references to original turn files.
  // This turns compaction into index compression:
  //   - pi frees memory internally
  //   - the summary is retrievable by semantic similarity
  //   - when retrieved, the LLM can drill down via search_interactions(turns=[])
  //
  // In session_before_compact we identify which turn files are being compacted
  // by matching userMessageIds. In session_compact we save+embed the summary.
  pi.on("session_before_compact", async (event, _ctx) => {
    const { preparation } = event;
    // Extract user message IDs from messages being compacted
    const userMsgIds: string[] = [];
    for (const msg of preparation.messagesToSummarize) {
      if (msg.role === "user" && (msg as Record<string, unknown>).id) {
        userMsgIds.push((msg as Record<string, unknown>).id as string);
      }
    }
    // Look up our turn files that match these message IDs
    const matched: string[] = [];
    for (const entry of compactionChain) {
      if (userMsgIds.includes(entry.userMessageId)) {
        matched.push(entry.turnFileName);
      }
    }
    pendingCompactionTurnFiles = matched;
    // Don't cancel — let compaction proceed normally
  });

  pi.on("session_compact", async (event, ctx) => {
    const { compactionEntry } = event;
    if (!compactionEntry || !compactionEntry.summary) return;

    const referencedTurns = pendingCompactionTurnFiles ?? [];
    pendingCompactionTurnFiles = null;

    // Build the summary turn with references
    const summaryText = `[COMPACTION SUMMARY]\nreferenced_turns: ${referencedTurns.join(", ")}\nsummary: ${compactionEntry.summary}`;
    const turnFileName = createTurnFilePath("compaction-summary", summaryText);
    const turnPath = `${TURNS_DIR}/${turnFileName}`;

    if (fs.existsSync(turnPath)) return; // dedup

    const turnData = {
      id: crypto.createHash("md5").update(summaryText).digest("hex"),
      userPrompt: "compaction-summary",
      responseSequence: summaryText,
      turnIndex: -1,
      userTimestamp: Date.now(),
      type: "compaction_summary",
      referencedTurns,
      originalSummary: compactionEntry.summary,
    };
    fs.mkdirSync(TURNS_DIR, { recursive: true });
    fs.writeFileSync(turnPath, JSON.stringify(turnData, null, 2));

    // Embed the summary
    try {
      const apiKey = await getApiKey();
      if (apiKey) {
        const vec = await embedText(summaryText, apiKey);
        appendToIndex(`${turnFileName}:prompt`, vec);
        appendToIndex(`${turnFileName}:response`, vec);
        ctx.ui.setStatus("contexter", `📚 compaction summary saved (${referencedTurns.length} turns referenced)`);
      }
    } catch (err) {
      ctx.ui.setStatus("contexter", `🧠 compaction embed error: ${(err as Error).message}`);
    }
  });

  // ─────────────────────────────────────────────
  // 4. Startup — register tool + show status
  // ─────────────────────────────────────────────
  // registerTool is called inside session_start because factory-level
  // registration doesn't reliably make tools visible to the LLM.
  pi.on("session_start", async (_event, ctx) => {
    const indexExists = fs.existsSync(INDEX_PATH);
    const index = indexExists ? loadIndex() : [];
    const uniqueTurns = new Set(
      index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
    ).size;
    ctx.ui.notify(
      `🧠 contexter loaded — ${uniqueTurns} turns indexed`,
      "info",
    );

    // Load persistent compaction chain
    compactionChain = loadCompactionChain();

    // Register the search_interactions tool here, not at factory level
    pi.registerTool({
      name: "search_interactions",
      label: "Search Interactions",
      description: "Search all past user interactions for topics, questions, or discussions. Unlike the built-in search_memory (which searches within the current session), this searches across ALL sessions the user has ever had — every conversation turn ever indexed. Use this when you need to find something from a past session, recall prior discussions, or reconnect with knowledge that was established a long time ago.\n\nYou can optionally scope the search to specific turn files by passing the `turns` parameter. This is useful when you find a compaction summary turn (type: compaction_summary) — it will contain referenced_turns that you can pass here to drill down into the original detail within that compacted section.",
      promptSnippet: "Search past interactions for relevant context",
      parameters: Type.Object({
        query: Type.String({ description: "The search query — what you want to find in past conversations" }),
        minSimilarity: Type.Optional(Type.Number({ description: "Minimum similarity threshold (0.0 to 1.0). Default 0.25. Lower to get broader matches." })),
        turns: Type.Optional(Type.Array(Type.String(), { description: "Optional list of turn filenames to scope the search to (e.g., ['turn-abc.json', 'turn-def.json']). When provided, only these turn files are searched — useful for drilling into compaction summary references." })),
      }),
      async execute(toolCallId, params, signal, onUpdate, ctx2) {
        const p = params as { query: string; minSimilarity?: number; turns?: string[] };
        const query = p.query;
        if (!query) {
          return {
            content: [{ type: "text", text: "No query provided." }],
            details: {},
          };
        }
        const threshold = p.minSimilarity ?? 0.25;
        const scopeTurns = p.turns ?? null;

        const apiKey = await getApiKey();
        if (!apiKey) {
          return {
            content: [{ type: "text", text: "No API key available for embedding. Skipping search." }],
            details: {},
          };
        }

        // Embed the query
        const queryVec = normalize(await embedText(query, apiKey));

        // Load index and score
        let index = loadIndex();
        if (index.length === 0) {
          return {
            content: [{ type: "text", text: "The turn index is empty. No conversations have been saved yet." }],
            details: {},
          };
        }

        // If turns[] is provided, scope the search to only those turn files
        if (scopeTurns && scopeTurns.length > 0) {
          const scopeSet = new Set(scopeTurns);
          index = index.filter((entry) => {
            const turnFile = entry.filePath.replace(/:prompt$|:response$/, "");
            return scopeSet.has(turnFile);
          });
          if (index.length === 0) {
            return {
              content: [{ type: "text", text: `No indexed vectors found for the specified turns: ${scopeTurns.join(", ")}. They may not be embedded yet.` }],
              details: {},
            };
          }
        }

        const scored = index
          .map((entry) => ({ ...entry, similarity: cosineSimilarity(queryVec, entry.vector) }))
          .sort((a, b) => b.similarity - a.similarity);

        // Group by turn file, take best score per turn
        const turnScores = new Map<string, { data: { userPrompt: string; responseSequence: string; turnIndex: number }; bestScore: number }>();
        for (const entry of scored) {
          const turnFile = entry.filePath.replace(/:prompt$|:response$/, "");
          if (!turnFile.match(/^turn-/)) continue;
          if (turnScores.has(turnFile)) continue;
          const turnData = readTurnFile(entry.filePath);
          if (!turnData) continue;
          turnScores.set(turnFile, { data: turnData, bestScore: entry.similarity });
        }

        const sorted = Array.from(turnScores.values())
          .sort((a, b) => b.bestScore - a.bestScore);

        if (sorted.length === 0) {
          return {
            content: [{ type: "text", text: "No matching turns found in the index." }],
            details: {},
          };
        }

        // Build result text — top 5 turns with score
        const MIN_SIMILARITY = threshold;
        const lines: string[] = [];
        let count = 0;
        for (const turn of sorted) {
          if (turn.bestScore < MIN_SIMILARITY) break;
          if (count >= 5) break;
          count++;
          lines.push(`--- Turn (score: ${turn.bestScore.toFixed(3)}) ---`);
          lines.push(`User: ${turn.data.userPrompt.slice(0, 500)}`);
          lines.push(`Assistant: ${turn.data.responseSequence.slice(0, 1000)}`);
          lines.push("");
        }

        if (count === 0) {
          return {
            content: [{ type: "text", text: `No relevant turns found (best score: ${sorted[0].bestScore.toFixed(3)}).` }],
            details: {},
          };
        }

        return {
          content: [{ type: "text", text: `Found ${count} relevant turns:\n\n${lines.join("\n")}` }],
          details: { matched: count, topScore: sorted[0].bestScore },
        };
      },
    });
  });
}
