/**
 * contexter — Retrieval-Augmented Context Assembly
 *
 * At agent_end: save the completed round to .pi/rounds/ and embed it.
 * At context: embed the current user prompt, query the vector index,
 *             inject the top-matching rounds as context for the LLM.
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
const ROUNDS_DIR = `${PROJECT_ROOT}/.pi/rounds`;
const INDEX_PATH = `${ROUNDS_DIR}/index.csv`;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";

const CONTEXT_BUDGET_RATIO = 0.5; // 50% of model context window for historical rounds

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

function readRoundFile(
  filePath: string,
): { userPrompt: string; responseSequence: string; turnIndex: number; userTimestamp?: number } | null {
  // filePath may be "xxx.json:prompt" or "xxx.json:response"
  // strip the :prompt/:response suffix to get the actual file
  const actualFile = filePath.replace(/:prompt$|:response$/, "");
  const fullPath = `${ROUNDS_DIR}/${actualFile}`;
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

let lastRoundFileName: string | null = null; // tracks the most recent saved round (process-local)

// Per-agent accumulation (reset in agent_start, saved in agent_end)
let agentUserPrompt: string | null = null;
let agentTurnIndex: number | null = null;
let agentAccumulatedText: string[] = []

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

function createRoundFilePath(userPrompt: string, responseText: string): string {
  const content = userPrompt + responseText;
  const hash = crypto.createHash("md5").update(content).digest("hex");
  return `${hash}.json`;
}

function appendToIndex(filePath: string, vector: number[]) {
  const line = `${filePath},${vector.join(",")}\n`;
  fs.mkdirSync(ROUNDS_DIR, { recursive: true });
  fs.appendFileSync(INDEX_PATH, line);
}

// ─────────────────────────────────────────────
// Extension
// ─────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // ────────────────────────────────────────────
  // 1. context — assemble context from round repository
  // ────────────────────────────────────────────
  pi.on("context", async (event: ContextEvent, ctx) => {
    const { messages } = event;

    // --- Extract system prompt + current round messages ---
    // We strip all prior rounds to prevent conversation bloat.
    // The retrieved historical context replaces the prior conversation.
    // Current round = everything from the last user message onward
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

    // No longer caching for agent_end -- agent_start handles it cleanly


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

      // Group by round file, take best score per round
      interface RoundScore {
        data: {
          userPrompt: string;
          responseSequence: string;
          turnIndex: number;
          userTimestamp?: number;
        };
        bestScore: number;
      }
      const roundScores = new Map<string, RoundScore>();
      for (const entry of scored) {
        const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
        if (!roundFile.endsWith(".json")) continue;
        if (roundScores.has(roundFile)) continue;
        const roundData = readRoundFile(entry.filePath);
        if (!roundData) continue;
        roundScores.set(roundFile, {
          data: roundData,
          bestScore: entry.similarity,
        });
      }

      const uniqueRounds = new Set(
        index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
      ).size;

      const scoredRounds = Array.from(roundScores.values()).sort(
        (a, b) => b.bestScore - a.bestScore,
      );

      // Select rounds within budget
      const selectedRounds: RoundScore[] = [];
      let usedTokens = 0;
      const addRound = (round: RoundScore) => {
        selectedRounds.push(round);
      };

      // 1. Score-based selection (below threshold stops)
      for (const round of scoredRounds) {
        if (round.bestScore < MIN_SIMILARITY) break;
        const roundTokens = estimateTokens(
          round.data.userPrompt + round.data.responseSequence,
        );
        if (usedTokens + roundTokens > budgetTokens) break;
        addRound(round);
        usedTokens += roundTokens;
      }

      // 2. Always add the last round (if not already there)
      // Content hash used later to move it to last position after dedup.
      let lastRoundContentHash: string | null = null;
      if (lastRoundFileName) {
        const lastData = readRoundFile(lastRoundFileName);
        if (lastData) {
          addRound({ data: lastData, bestScore: 0 });
          lastRoundContentHash = crypto.createHash("md5")
            .update(lastData.userPrompt + lastData.responseSequence)
            .digest("hex");
        }
      }

      if (selectedRounds.length === 0) {
        ctx.ui.setStatus(
          "contexter",
          `🧠 no relevant context (best: ${bestScore.toFixed(3)})`,
        );
        return { messages };
      }

      // Build context messages — chronological order (oldest first) for coherence
      // Chronological: prefer turnIndex within session, then userTimestamp as tiebreaker
      selectedRounds.sort((a, b) => {
        const ti = a.data.turnIndex - b.data.turnIndex;
        if (ti !== 0) return ti;
        const tsA = (a.data as Record<string, unknown>).userTimestamp as number ?? 0;
        const tsB = (b.data as Record<string, unknown>).userTimestamp as number ?? 0;
        return tsA - tsB;
      });

      // Dedup by MD5 content hash — last round may duplicate a scored round
      const seenHashes = new Set<string>();
      const dedupedRounds: typeof selectedRounds = [];
      for (const round of selectedRounds) {
        const hash = crypto.createHash("md5").update(round.data.userPrompt + round.data.responseSequence).digest("hex");
        if (seenHashes.has(hash)) continue;
        seenHashes.add(hash);
        dedupedRounds.push(round);
      }

      // Post-sort: ensure last round is always the LAST element in the injected context.
      // This bridges the referential gap — the model sees the preceding round's response
      // immediately before the current prompt, making "these", "it", "that" resolvable.
      if (lastRoundContentHash) {
        const lastIdx = dedupedRounds.findIndex(r => {
          const h = crypto.createHash("md5").update(r.data.userPrompt + r.data.responseSequence).digest("hex");
          return h === lastRoundContentHash;
        });
        if (lastIdx !== -1 && lastIdx !== dedupedRounds.length - 1) {
          const [lastRound] = dedupedRounds.splice(lastIdx, 1);
          dedupedRounds.push(lastRound);
        }
      }

      // Rebuild contextMessages from deduped rounds
      // Each round is wrapped in a clear delimiter so the model knows these
      // are historical records — not the current conversation. This prevents
      // the model from mimicking tool calls or phrasing from past rounds.
      const contextMessages: Array<{ role: string; content: Array<{ type: string; text: string }> }> = [];
      for (const round of dedupedRounds) {
        contextMessages.push({
          role: "user",
          content: [{ type: "text", text: `[HISTORICAL TURN from a past conversation — similarity: ${round.bestScore.toFixed(3)}]
User asked: ${round.data.userPrompt}` }],
        });
        contextMessages.push({
          role: "assistant",
          content: [{ type: "text", text: `[END OF HISTORICAL TURN — this was a past response, not the current one. Stay focused on the user's most recent message above.]
${round.data.responseSequence}` }],
        });
      }

      ctx.ui.setStatus(
        "contexter",
        `🧠 retrieved ${dedupedRounds.length} rounds (${usedTokens} tok) from ${uniqueRounds} indexed`,
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
  // 2. agent_start + message_end + agent_end — Save round + embed it
  // ────────────────────────────────────────────
  // agent_start/agent_end fire once per user prompt (unlike turn_start/turn_end
  // which fire per inner LLM call within a tool-calling loop). By saving at
  // agent_end we capture the FULL assistant response across all tool iterations.
  pi.on("agent_start", async (event, _ctx) => {
    const { messages } = event;
    // Extract the first user message content as the prompt for this agent cycle
    const firstUser = messages?.find((m: { role: string }) => m.role === "user");
    if (firstUser) {
      const content = firstUser.content;
      if (typeof content === "string") {
        agentUserPrompt = content;
      } else if (Array.isArray(content)) {
        agentUserPrompt = extractText(content as Array<{ type: string; text?: string }>);
      }
    }
    agentTurnIndex = event.turnIndex ?? null;
    agentAccumulatedText = [];
  });

  pi.on("message_end", async (event, _ctx) => {
    const msg = event.message;
    if (!msg) return;

    if (msg.role === "user") {
      // User sent something -- don't reset the accumulator, this is a new agent
      // cycle (agent_start will reset it). Keep safe.
    } else if (msg.role === "assistant") {
      // Extract text from this assistant message
      const content = msg.content as Array<{ type: string; text?: string }> | undefined;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === "text" && block.text) {
            agentAccumulatedText.push(block.text);
          }
        }
      }
    }
  });

  pi.on("agent_end", async (event, ctx) => {
    const { messages } = event;

    // Get user prompt -- prefer agent_start cached value, fall back to messages
    let userPrompt = agentUserPrompt ?? "";
    if (!userPrompt && messages) {
      const lastUser = [...messages].reverse().find((m: { role: string }) => m.role === "user");
      if (lastUser) {
        const content = lastUser.content;
        if (typeof content === "string") {
          userPrompt = content;
        } else if (Array.isArray(content)) {
          userPrompt = extractText(content as Array<{ type: string; text?: string }>);
        }
      }
    }

    if (!userPrompt) {
      ctx.ui.setStatus("contexter", "\u{1f9e0} agent_end: no user prompt to save");
      return;
    }

    // Build response text from accumulated assistant text across all tool iterations
    let responseText = agentAccumulatedText.join("\n\n").trim();
    if (!responseText) {
      // Fallback: extract text from messages (last assistant message)
      const lastAssistant = messages ? [...messages].reverse().find((m: { role: string }) => m.role === "assistant") : null;
      if (lastAssistant) {
        const content = lastAssistant.content;
        if (typeof content === "string") {
          responseText = content;
        } else if (Array.isArray(content)) {
          responseText = extractText(content as Array<{ type: string; text?: string }>);
        }
      }
    }

    if (!responseText) {
      ctx.ui.setStatus("contexter", "\u{1f9e0} agent_end: no response text");
      return;
    }

    fs.mkdirSync(ROUNDS_DIR, { recursive: true });

    const roundFileName = createRoundFilePath(userPrompt, responseText);
    const roundPath = `${ROUNDS_DIR}/${roundFileName}`;

    // Skip if already saved (deduplication by content hash)
    if (fs.existsSync(roundPath)) {
      ctx.ui.setStatus("contexter", `\u{1f9e0} round already saved (${roundFileName})`);
      lastRoundFileName = roundFileName;
      agentAccumulatedText = [];
      agentUserPrompt = null;
      agentTurnIndex = null;
      return;
    }

    // Write round file
    const roundData = {
      id: crypto.createHash("md5").update(userPrompt + responseText).digest("hex"),
      userPrompt,
      responseSequence: responseText,
      turnIndex: agentTurnIndex ?? 0,
      userTimestamp: Date.now(),
    };

    try {
      fs.writeFileSync(roundPath, JSON.stringify(roundData, null, 2));
    } catch (err) {
      ctx.ui.setStatus("contexter", `\u{1f9e0} write error: ${(err as Error).message}`);
      agentAccumulatedText = [];
      agentUserPrompt = null;
      agentTurnIndex = null;
      return;
    }

    // Embed prompt and response separately
    const apiKey = await getApiKey();
    if (!apiKey) {
      ctx.ui.setStatus("contexter", "\u{1f9e0} saved but not embedded (no API key)");
      lastRoundFileName = roundFileName;
      agentAccumulatedText = [];
      agentUserPrompt = null;
      agentTurnIndex = null;
      return;
    }

    try {
      const [promptVec, responseVec] = await Promise.all([
        embedText(userPrompt, apiKey),
        embedText(responseText, apiKey),
      ]);
      appendToIndex(`${roundFileName}:prompt`, promptVec);
      appendToIndex(`${roundFileName}:response`, responseVec);
      ctx.ui.setStatus(
        "contexter",
        `\u{1f9e0} saved + embedded round (${roundFileName})`,
      );
    } catch (err) {
      ctx.ui.setStatus("contexter", `\u{1f9e0} embedding error: ${(err as Error).message}`);
    }

    lastRoundFileName = roundFileName;
    agentAccumulatedText = [];
    agentUserPrompt = null;
    agentTurnIndex = null;
  });

  // 4. Compaction integration — capture summary as a round
  // ────────────────────────────────────────────
  // Instead of cancelling compaction (causes unbounded memory growth in pi's
  // internal entry chain), we let it proceed. The compacted summary becomes a
  // special round in our index with references to original round files.
  // This turns compaction into index compression:
  //   - pi frees memory internally
  //   - the summary is retrievable by semantic similarity
  //   - when retrieved, the LLM can drill down via search_interactions(rounds=[])
  //
  // In session_before_compact we identify which round files are being compacted
  // by matching user message content via MD5 hash. This works regardless of
  // whether the round was saved by our agent_end handler or by digest-all.ts.
  pi.on("session_before_compact", async (event, _ctx) => {
    const { preparation } = event;

    // Build a lookup of user prompt MD5 hash → roundFileName from ALL round files
    // Matches by message content, not by message ID — so it works for rounds
    // created by digest-all.ts and by the agent_end handler alike.
    const promptHashToFile = new Map<string, string>();
    try {
      const files = fs.readdirSync(ROUNDS_DIR).filter(f => f.endsWith(".json") && !f.startsWith("index"));
      for (const file of files) {
        const raw = fs.readFileSync(ROUNDS_DIR + "/" + file, "utf-8");
        const data = JSON.parse(raw);
        if (data.userPrompt && data.userPrompt !== "compaction-summary") {
          const hash = crypto.createHash("md5").update(data.userPrompt).digest("hex");
          promptHashToFile.set(hash, file);
        }
      }
    } catch { /* ignore read errors */ }

    // Extract user message content from messages being compacted,
    // hash it, and look up the matching round file.
    // IMPORTANT: use extractText() here (same as digest-all.ts) so the hash
    // matches what was stored in the round file's userPrompt field.
    const matched = new Set<string>();
    for (const msg of preparation.messagesToSummarize) {
      if (msg.role !== "user") continue;
      let text = "";
      const content = msg.content;
      if (typeof content === "string") {
        text = content;
      } else if (Array.isArray(content)) {
        text = extractText(content as Array<{ type: string; text?: string }>);
      }
      if (!text) continue;
      const hash = crypto.createHash("md5").update(text).digest("hex");
      const file = promptHashToFile.get(hash);
      if (file) matched.add(file);
    }

    pendingCompactionTurnFiles = Array.from(matched);
    // Don't cancel — let compaction proceed normally
  });

  pi.on("session_compact", async (event, ctx) => {
    const { compactionEntry } = event;
    if (!compactionEntry || !compactionEntry.summary) return;

    const referencedTurns = pendingCompactionTurnFiles ?? [];
    pendingCompactionTurnFiles = null;

    // Build the summary round with references
    const summaryText = `[COMPACTION SUMMARY]\nreferenced_rounds: ${referencedTurns.join(", ")}\nsummary: ${compactionEntry.summary}`;
    const roundFileName = createRoundFilePath("compaction-summary", summaryText);
    const roundPath = `${ROUNDS_DIR}/${roundFileName}`;

    if (fs.existsSync(roundPath)) return; // dedup

    const roundData = {
      id: crypto.createHash("md5").update(summaryText).digest("hex"),
      userPrompt: "compaction-summary",
      responseSequence: summaryText,
      turnIndex: -1,
      userTimestamp: Date.now(),
      type: "compaction_summary",
      referencedTurns,
      originalSummary: compactionEntry.summary,
    };
    fs.mkdirSync(ROUNDS_DIR, { recursive: true });
    fs.writeFileSync(roundPath, JSON.stringify(roundData, null, 2));

    // Embed the summary
    try {
      const apiKey = await getApiKey();
      if (apiKey) {
        const vec = await embedText(summaryText, apiKey);
        appendToIndex(`${roundFileName}:prompt`, vec);
        appendToIndex(`${roundFileName}:response`, vec);
        ctx.ui.setStatus("contexter", `📚 compaction summary saved (${referencedTurns.length} rounds referenced)`);
      }
    } catch (err) {
      ctx.ui.setStatus("contexter", `🧠 compaction embed error: ${(err as Error).message}`);
    }
  });

  // ─────────────────────────────────────────────
  // 5. Startup — register tool + show status
  // ─────────────────────────────────────────────
  // registerTool is called inside session_start because factory-level
  // registration doesn't reliably make tools visible to the LLM.
  pi.on("session_start", async (_event, ctx) => {
    const indexExists = fs.existsSync(INDEX_PATH);
    const index = indexExists ? loadIndex() : [];
    const uniqueRounds = new Set(
      index.map((e: { filePath: string }) => e.filePath.replace(/:prompt$|:response$/, ""))
    ).size;
    ctx.ui.notify(
      `🧠 contexter loaded — ${uniqueRounds} rounds indexed`,
      "info",
    );

    // Register the search_interactions tool here, not at factory level
    pi.registerTool({
      name: "search_interactions",
      label: "Search Interactions",
      description: "Search all past user interactions for topics, questions, or discussions. Unlike the built-in search_memory (which searches within the current session), this searches across ALL sessions the user has ever had — every conversation round ever indexed. Use this when you need to find something from a past session, recall prior discussions, or reconnect with knowledge that was established a long time ago.\n\nYou can optionally scope the search to specific round files by passing the `turns` parameter. This is useful when you find a compaction summary round (type: compaction_summary) — it will contain referenced_rounds (stored as the field referenced_turns for backward compatibility) that you can pass here to drill down into the original detail within that compacted section.",
      promptSnippet: "Search past interactions for relevant context",
      parameters: Type.Object({
        query: Type.String({ description: "The search query — what you want to find in past conversations" }),
        minSimilarity: Type.Optional(Type.Number({ description: "Minimum similarity threshold (0.0 to 1.0). Default 0.25. Lower to get broader matches." })),
        turns: Type.Optional(Type.Array(Type.String(), { description: "Optional list of round filenames to scope the search to (e.g., ['abc.json', 'def.json']). When provided, only these round files are searched — useful for drilling into compaction summary references." })),
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
        const scopeRounds = p.rounds ?? null;

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
            content: [{ type: "text", text: "The round index is empty. No conversations have been saved yet." }],
            details: {},
          };
        }

        // If rounds[] is provided, scope the search to only those round files
        if (scopeRounds && scopeRounds.length > 0) {
          const scopeSet = new Set(scopeRounds);
          index = index.filter((entry) => {
            const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
            return scopeSet.has(roundFile);
          });
          if (index.length === 0) {
            return {
              content: [{ type: "text", text: `No indexed vectors found for the specified rounds: ${scopeRounds.join(", ")}. They may not be embedded yet.` }],
              details: {},
            };
          }
        }

        const scored = index
          .map((entry) => ({ ...entry, similarity: cosineSimilarity(queryVec, entry.vector) }))
          .sort((a, b) => b.similarity - a.similarity);

        // Group by round file, take best score per round
        const roundScores = new Map<string, { data: { userPrompt: string; responseSequence: string; turnIndex: number }; bestScore: number }>();
        for (const entry of scored) {
          const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
          if (!roundFile.endsWith(".json")) continue;
          if (roundScores.has(roundFile)) continue;
          const roundData = readRoundFile(entry.filePath);
          if (!roundData) continue;
          roundScores.set(roundFile, { data: roundData, bestScore: entry.similarity });
        }

        const sorted = Array.from(roundScores.values())
          .sort((a, b) => b.bestScore - a.bestScore);

        if (sorted.length === 0) {
          return {
            content: [{ type: "text", text: "No matching turns found in the index." }],
            details: {},
          };
        }

        // Build result text — top 5 rounds with score
        const MIN_SIMILARITY = threshold;
        const lines: string[] = [];
        let count = 0;
        for (const round of sorted) {
          if (round.bestScore < MIN_SIMILARITY) break;
          if (count >= 5) break;
          count++;
          lines.push(`--- Round (score: ${round.bestScore.toFixed(3)}) ---`);
          lines.push(`User: ${round.data.userPrompt.slice(0, 500)}`);
          lines.push(`Assistant: ${round.data.responseSequence.slice(0, 1000)}`);
          lines.push("");
        }

        if (count === 0) {
          return {
            content: [{ type: "text", text: `No relevant rounds found (best score: ${sorted[0].bestScore.toFixed(3)}).` }],
            details: {},
          };
        }

        return {
          content: [{ type: "text", text: `Found ${count} relevant rounds:\n\n${lines.join("\n")}` }],
          details: { matched: count, topScore: sorted[0].bestScore },
        };
      },
    });
  });
}
