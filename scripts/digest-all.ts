/**
 * digest-all.ts — Bulk-embed all pi session JSONL files into the contexter index.
 *
 * Iterates every session in ~/.pi/agent/sessions/, skips already-indexed turns,
 * parallelizes embedding via OpenRouter.
 *
 * Usage:
 *   OPENROUTER_API_KEY="$(pass show ai/openrouter)" npx tsx scripts/digest-all.ts
 *
 * Safe to run while pi is using the extension — the index is append-only.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// ─────────────────────────────────────────────
// Config (matches digest-session.ts)
// ─────────────────────────────────────────────

const SESSIONS_DIR = path.resolve(os.homedir(), ".pi", "agent", "sessions");
const CONVERTER_DIR = path.resolve(import.meta.dirname, "..", ".pi", "turns");
const TURNS_DIR = CONVERTER_DIR;
const INDEX_PATH = path.resolve(TURNS_DIR, "index.csv");

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";
const CONCURRENCY = 5; // parallel embedding requests
const MAX_RESPONSE_CHARS = 8000;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface Turn {
  id: string;
  userPrompt: string;
  responseSequence: string;
  userTimestamp: number;
  responseEndTimestamp: number;
  turnIndex: number;
  sessionLabel: string; // human-readable label for this session file
}

// ─────────────────────────────────────────────
// Parse a single JSONL file into turns
// ─────────────────────────────────────────────

function parseSessionFile(filePath: string, sessionLabel: string): Turn[] {
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.trim().split("\n").filter(Boolean);
  const entries: Array<Record<string, unknown>> = lines.map((l) => JSON.parse(l));

  const turns: Turn[] = [];
  let currentUserMsg: Record<string, unknown> | null = null;
  let responseParts: string[] = [];
  let turnIndex = 0;

  for (const entry of entries) {
    if (entry.type !== "message") continue;
    const msg = entry.message as Record<string, unknown> | undefined;
    if (!msg) continue;

    const role = msg.role as string;
    const content = msg.content as Array<{ type: string; text?: string }> | undefined;
    const timestamp = msg.timestamp as number | undefined;

    if (role === "user") {
      // Save previous turn if exists
      if (currentUserMsg) {
        turns.push({
          id: currentUserMsg.id as string,
          userPrompt: extractText(
            (currentUserMsg.message as Record<string, unknown>)?.content as
              | Array<{ type: string; text?: string }>
              | undefined,
          ),
          responseSequence: responseParts.join("\n\n").trim(),
          userTimestamp: (currentUserMsg.message as Record<string, unknown>)?.timestamp as number ?? 0,
          responseEndTimestamp: timestamp ?? Date.now(),
          turnIndex,
          sessionLabel,
        });
        turnIndex++;
      }
      currentUserMsg = entry;
      responseParts = [];
    } else if (role === "assistant" && currentUserMsg && content) {
      const text = extractText(content);
      if (text) responseParts.push(text);
    }
  }

  // Save last turn — only if it has a non-empty response or we have multiple turns.
  // Skip turns whose response is trivially short (< 20 chars) — these are usually
  // session files that ended mid-stream (truncated assistant response).
  const finalResponse = responseParts.join("\n\n").trim();
  if (currentUserMsg && (finalResponse.length >= 20 || turnIndex > 0)) {
    turns.push({
      id: currentUserMsg.id as string,
      userPrompt: extractText(
        (currentUserMsg.message as Record<string, unknown>)?.content as
          | Array<{ type: string; text?: string }>
          | undefined,
      ),
      responseSequence: finalResponse,
      userTimestamp: (currentUserMsg.message as Record<string, unknown>)?.timestamp as number ?? 0,
      responseEndTimestamp: Date.now(),
      turnIndex,
      sessionLabel,
    });
  }

  return turns;
}

function extractText(content?: Array<{ type: string; text?: string }>): string {
  if (!content || !Array.isArray(content)) return "";
  return content
    .filter((c): c is { type: string; text: string } => c.type === "text" && typeof c.text === "string")
    .map((c) => c.text)
    .join(" ")
    .trim();
}

// ─────────────────────────────────────────────
// Embedding (single call)
// ─────────────────────────────────────────────

async function embed(text: string): Promise<number[]> {
  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: text,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`OpenRouter embedding error (${response.status}): ${err}`);
  }

  const data = (await response.json()) as {
    data: Array<{ embedding: number[] }>;
  };
  return data.data[0].embedding;
}

// ─────────────────────────────────────────────
// Vector helpers
// ─────────────────────────────────────────────

function normalize(v: number[]): number[] {
  const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return mag === 0 ? v : v.map((x) => x / mag);
}

// ─────────────────────────────────────────────
// Index I/O
// ─────────────────────────────────────────────

function loadIndexFilePaths(): Set<string> {
  if (!fs.existsSync(INDEX_PATH)) return new Set();
  const lines = fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean);
  return new Set(lines.map((line) => {
    const [, filePath] = line.split(",", 2);
    return filePath.replace(/:prompt$|:response$/, "");
  }));
}

function appendToIndex(vector: number[], filePath: string): void {
  const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
  fs.appendFileSync(INDEX_PATH, `${b64},${filePath}\n`);
}

// ─────────────────────────────────────────────
// Count user messages in a JSONL (for progress)
// ─────────────────────────────────────────────

function countUserMessages(filePath: string): number {
  const content = fs.readFileSync(filePath, "utf-8");
  let count = 0;
  for (const line of content.trim().split("\n").filter(Boolean)) {
    try {
      const obj = JSON.parse(line);
      if (obj.type === "message" && obj.message?.role === "user") count++;
    } catch {}
  }
  return count;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

async function main() {
  if (!OPENROUTER_API_KEY) {
    console.error("❌ OPENROUTER_API_KEY environment variable required");
    process.exit(1);
  }

  // Gather all session JSONL files
  const sessionDirs = fs.readdirSync(SESSIONS_DIR)
    .filter((d) => d.startsWith("--"))
    .map((d) => path.join(SESSIONS_DIR, d));

  const jsonlFiles: Array<{ filePath: string; label: string }> = [];
  for (const dir of sessionDirs) {
    const label = path.basename(dir);
    const files = fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
    for (const f of files) {
      jsonlFiles.push({ filePath: path.join(dir, f), label });
    }
  }

  jsonlFiles.sort((a, b) => a.filePath.localeCompare(b.filePath));

  console.log(`📂 Found ${jsonlFiles.length} session files across ${sessionDirs.length} directories\n`);

  // Ensure the .pi/turns dir
  fs.mkdirSync(TURNS_DIR, { recursive: true });

  // Load existing index dedup set
  const existingTurns = loadIndexFilePaths();
  console.log(`📊 Already indexed: ${existingTurns.size} turns\n`);

  // Parse all sessions into a flat list of turns (skipping already-indexed)
  const allTurns: Turn[] = [];
  let skippedTotal = 0;

  for (const { filePath, label } of jsonlFiles) {
    const turns = parseSessionFile(filePath, label);
    const newTurns = turns.filter((t) => {
      const key = `turn-${require("node:crypto").createHash("md5").update(t.userPrompt + t.responseSequence).digest("hex")}.json`;
      return !existingTurns.has(key);
    });
    skippedTotal += turns.length - newTurns.length;
    allTurns.push(...newTurns);
  }

  const totalNew = allTurns.length;
  console.log(`📊 New turns to embed: ${totalNew} (${skippedTotal} already indexed)\n`);

  if (totalNew === 0) {
    console.log("✨ Nothing to do — all sessions already indexed!");
    return;
  }

  // Parallel embedding with concurrency limit
  let completed = 0;
  let errors = 0;

  async function processTurn(turn: Turn): Promise<void> {
    const crypto = require("node:crypto");
    const turnFile = `turn-${crypto.createHash("md5").update(turn.userPrompt + turn.responseSequence).digest("hex")}.json`;
    const turnId = `${turn.sessionLabel}/${turnFile}`;

    // Write turn file (always)
    fs.writeFileSync(
      path.resolve(TURNS_DIR, turnFile),
      JSON.stringify(turn, null, 2),
    );

    try {
      const promptVector = await embed(turn.userPrompt);
      appendToIndex(normalize(promptVector), `${turnFile}:prompt`);

      const respText = turn.responseSequence.slice(0, MAX_RESPONSE_CHARS);
      if (respText) {
        const respVector = await embed(respText);
        appendToIndex(normalize(respVector), `${turnFile}:response`);
      }

      completed++;
      const pct = ((completed / totalNew) * 100).toFixed(1);
      process.stderr.write(
        `  ✅ [${completed}/${totalNew} ${pct}%] ${turnId}\n`,
      );
    } catch (err) {
      errors++;
      process.stderr.write(`  ❌ [ERROR] ${turnId}: ${(err as Error).message}\n`);
    }
  }

  // Run with concurrency limit
  async function runQueue(): Promise<void> {
    const queue = [...allTurns];
    const workers: Promise<void>[] = [];

    for (let i = 0; i < CONCURRENCY; i++) {
      workers.push(
        (async () => {
          while (queue.length > 0) {
            const turn = queue.shift()!;
            await processTurn(turn);
          }
        })(),
      );
    }

    await Promise.all(workers);
  }

  const startTime = Date.now();
  await runQueue();
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  const finalCount = fs.existsSync(INDEX_PATH)
    ? fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean).length
    : 0;

  console.log(`\n✅ Done in ${elapsed}s. ${completed} turns embedded, ${errors} errors.`);
  console.log(`   Index: ${finalCount} vectors at ${INDEX_PATH}`);
  console.log(`   Turns: ${fs.readdirSync(TURNS_DIR).filter((f) => f.startsWith("turn-")).length} files`);
}

main().catch((err) => {
  console.error("❌ Fatal:", err);
  process.exit(1);
});
