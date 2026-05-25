/**
 * digest-session.ts — Parse a pi session JSONL into turns, embed them via OpenRouter,
 * and build a vector index.
 *
 * Usage:
 *   npx tsx scripts/digest-session.ts <session-file>
 *
 * Output:
 *   .pi/turns/<id>.json    — each turn as a file
 *   .pi/turns/index.csv    — vector index (base64(vector),filepath)
 */

import * as fs from "node:fs";
import * as path from "node:path";

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
}

interface SessionEntry {
  type: string;
  id: string;
  parentId: string | null;
  timestamp?: string;
  message?: {
    role: string;
    content: Array<{ type: string; text?: string }>;
    timestamp?: number;
  };
}

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"; // actually embeddings endpoint
const TURNS_DIR = path.resolve(import.meta.dirname, "..", ".pi", "turns");
const INDEX_PATH = path.resolve(TURNS_DIR, "index.csv");

// ─────────────────────────────────────────────
// Parse session into turns
// ─────────────────────────────────────────────

function parseSession(filePath: string): Turn[] {
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.trim().split("\n").filter(Boolean);
  const entries: SessionEntry[] = lines.map((l) => JSON.parse(l));

  const turns: Turn[] = [];
  let currentUserMsg: SessionEntry | null = null;
  let responseParts: string[] = [];
  let turnIndex = 0;

  for (const entry of entries) {
    if (entry.type !== "message" || !entry.message) continue;
    const { role, content } = entry.message;

    if (role === "user") {
      // Save previous turn if exists
      if (currentUserMsg) {
        turns.push({
          id: currentUserMsg.id,
          userPrompt: extractText(currentUserMsg.message!.content),
          responseSequence: responseParts.join("\n\n").trim(),
          userTimestamp: currentUserMsg.message!.timestamp ?? 0,
          responseEndTimestamp: entry.message?.timestamp ?? 0,
          turnIndex,
        });
        turnIndex++;
      }
      currentUserMsg = entry;
      responseParts = [];
    } else if (role === "assistant" && currentUserMsg) {
      const text = extractText(content);
      if (text) responseParts.push(text);
    }
  }

  // Save last turn
  if (currentUserMsg) {
    turns.push({
      id: currentUserMsg.id,
      userPrompt: extractText(currentUserMsg.message!.content),
      responseSequence: responseParts.join("\n\n").trim(),
      userTimestamp: currentUserMsg.message!.timestamp ?? 0,
      responseEndTimestamp: Date.now(),
      turnIndex,
    });
  }

  return turns;
}

function extractText(content: Array<{ type: string; text?: string }>): string {
  return content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text!)
    .join(" ")
    .trim();
}

// ─────────────────────────────────────────────
// Embedding via OpenRouter
// ─────────────────────────────────────────────

async function embed(text: string): Promise<number[]> {
  if (!OPENROUTER_API_KEY) {
    throw new Error("OPENROUTER_API_KEY environment variable required");
  }

  const response = await fetch(
    "https://openrouter.ai/api/v1/embeddings",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: text,
      }),
    },
  );

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

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ─────────────────────────────────────────────
// Index I/O
// ─────────────────────────────────────────────

function loadIndex(): Array<{ vector: number[]; filePath: string }> {
  if (!fs.existsSync(INDEX_PATH)) return [];
  const lines = fs.readFileSync(INDEX_PATH, "utf-8").trim().split("\n").filter(Boolean);
  return lines.map((line) => {
    const [b64, filePath] = line.split(",", 2);
    const vector = JSON.parse(Buffer.from(b64, "base64url").toString("utf-8"));
    return { vector, filePath };
  });
}

function appendToIndex(vector: number[], filePath: string): void {
  const b64 = Buffer.from(JSON.stringify(vector)).toString("base64url");
  fs.appendFileSync(INDEX_PATH, `${b64},${filePath}\n`);
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

async function main() {
  const sessionFile = process.argv[2];
  if (!sessionFile) {
    console.error("Usage: npx tsx scripts/digest-session.ts <session-file>");
    process.exit(1);
  }

  console.log(`📂 Session: ${sessionFile}`);
  const turns = parseSession(sessionFile);
  console.log(`📊 Parsed ${turns.length} turns`);

  // Ensure turns directory
  fs.mkdirSync(TURNS_DIR, { recursive: true });

  // Check existing index to skip already-processed turns
  const existing = new Set(
    loadIndex().map((e) => path.basename(e.filePath)),
  );

  let embedded = 0;
  let skipped = 0;

  for (const turn of turns) {
    const crypto = require("node:crypto");
    const turnFile = `turn-${crypto.createHash("md5").update(turn.userPrompt + turn.responseSequence).digest("hex")}.json`;

    // Always write the turn file (idempotent)
    fs.writeFileSync(
      path.resolve(TURNS_DIR, turnFile),
      JSON.stringify(turn, null, 2),
    );

    if (existing.has(turnFile)) {
      skipped++;
      continue;
    }

    // Embed prompt
    console.log(`  🔄 Embedding turn ${turn.turnIndex + 1}/${turns.length}...`);
    const promptVector = await embed(turn.userPrompt);
    appendToIndex(normalize(promptVector), turnFile + ":prompt");

    // Embed response
    const respVector = await embed(turn.responseSequence.slice(0, 8000)); // conservative token limit
    appendToIndex(normalize(respVector), turnFile + ":response");

    embedded++;
  }

  console.log(`\n✅ Done. ${embedded} new turns embedded, ${skipped} already in index.`);
  console.log(`   Index: ${INDEX_PATH} (${loadIndex().length} vectors)`);
}

main().catch((err) => {
  console.error("❌ Error:", err);
  process.exit(1);
});
