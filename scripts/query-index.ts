/**
 * query-index.ts — Embed a prompt, query the vector index, show which rounds match.
 *
 * Usage:
 *   npx tsx scripts/query-index.ts "<your prompt>"
 *
 * Options:
 *   --top N     Show top N results (default: 10)
 *   --budget N  Show as many rounds as fit in N tokens (default: no limit)
 *   --show      Also print the round content
 *   --chrono    Order selected rounds by timestamp (chronological) instead of relevance
 */

import * as fs from "node:fs";
import * as path from "node:path";

const ROUNDS_DIR = path.resolve(import.meta.dirname, "..", ".pi", "rounds");
const INDEX_PATH = path.resolve(ROUNDS_DIR, "index.csv");

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const EMBEDDING_MODEL = "openai/text-embedding-3-small";

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

// ─────────────────────────────────────────────
// Embedding via OpenRouter
// ─────────────────────────────────────────────

async function embed(text: string): Promise<number[]> {
  const response = await fetch("https://openrouter.ai/api/v1/embeddings", {
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
    throw new Error(`OpenRouter error (${response.status}): ${err}`);
  }

  const data = (await response.json()) as { data: Array<{ embedding: number[] }> };
  return data.data[0].embedding;
}

// ─────────────────────────────────────────────
// Turn file reader
// ─────────────────────────────────────────────

interface Turn {
  id: string;
  userPrompt: string;
  responseSequence: string;
  turnIndex: number;
}

function readRound(filePath: string): Round | null {
  const fullPath = path.resolve(ROUNDS_DIR, filePath.replace(/:prompt$|:response$/, ""));
  try {
    return JSON.parse(fs.readFileSync(fullPath, "utf-8"));
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────────
// Token estimation (rough: 4 chars ≈ 1 token)
// ─────────────────────────────────────────────

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const topNIdx = args.indexOf("--top");
  const topN = topNIdx >= 0 ? parseInt(args[topNIdx + 1], 10) : 10;
  const budgetIdx = args.indexOf("--budget");
  const budget = budgetIdx >= 0 ? parseInt(args[budgetIdx + 1], 10) : null;
  const showContent = args.includes("--show");
  const chrono = args.includes("--chrono");

  // Collect prompt from remaining args (remove flags)
  const flagSet = new Set(["--top", "--budget", "--show", "--chrono"]);
  const promptArgs = args.filter((a, i) => {
    if (flagSet.has(a)) return false;
    const prev = args[i - 1];
    if (prev === "--top" || prev === "--budget") return false;
    return true;
  });
  const query = promptArgs.join(" ");

  if (!query) {
    console.error("Usage: npx tsx scripts/query-index.ts <query> [--top N] [--budget N] [--show]");
    process.exit(1);
  }

  if (!OPENROUTER_API_KEY) {
    console.error("OPENROUTER_API_KEY environment variable required");
    process.exit(1);
  }

  if (!fs.existsSync(INDEX_PATH)) {
    console.error("No index found. Run digest-session.ts first.");
    process.exit(1);
  }

  console.log(`🔍 Query: "${query}"`);
  console.log();

  // Embed query
  console.log("  Embedding query...");
  const queryVec = normalize(await embed(query));

  // Load index and compute distances
  const index = loadIndex();
  const scored = index
    .map((entry) => ({
      ...entry,
      similarity: cosineSimilarity(queryVec, entry.vector),
    }))
    .sort((a, b) => b.similarity - a.similarity);

  console.log(`  Scored ${index.length} vectors (${Math.ceil(index.length / 2)} rounds)\n`);

  // Group by round (prompt + response vectors)
  const roundScores = new Map<number, { round: Round; promptSim: number; respSim: number; sources: string[] }>();

  for (const entry of scored) {
    const roundFile = entry.filePath.replace(/:prompt$|:response$/, "");
    const round = readRound(roundFile);
    if (!round) continue;

    if (!roundScores.has(round.turnIndex)) {
      roundScores.set(round.turnIndex, {
        round,
        promptSim: 0,
        respSim: 0,
        sources: [],
      });
    }

    const data = roundScores.get(round.turnIndex)!;
    if (entry.filePath.endsWith(":prompt")) {
      data.promptSim = entry.similarity;
      data.sources.push(`prompt(sim=${entry.similarity.toFixed(4)})`);
    } else {
      data.respSim = entry.similarity;
      data.sources.push(`resp(sim=${entry.similarity.toFixed(4)})`);
    }
  }

  // Sort by best score (max of prompt or response sim)
  const ranked = Array.from(roundScores.values())
    .map((d) => ({
      ...d,
      bestScore: Math.max(d.promptSim, d.respSim),
      avgScore: (d.promptSim + d.respSim) / 2,
    }))
    .sort((a, b) => b.bestScore - a.bestScore);

  // Apply budget
  let selected: typeof ranked = [];
  if (budget) {
    let usedTokens = 0;
    const systemTokens = 500; // rough system prompt size
    const currentPromptTokens = estimateTokens(query);
    const reserveTokens = 4000; // room for model response
    const available = budget - systemTokens - currentPromptTokens - reserveTokens;

    for (const r of ranked) {
      const roundTokens = estimateTokens(r.round.userPrompt + r.round.responseSequence);
      if (usedTokens + roundTokens <= available) {
        selected.push(r);
        usedTokens += roundTokens;
      }
    }

    console.log(`📏 Budget: ${budget} tokens, available for rounds: ~${available}, used: ${usedTokens}\n`);
  } else {
    selected = ranked.slice(0, topN);
  }

  // Display results
  console.log(`📋 Selected ${selected.length} rounds:\n`);

  for (let i = 0; i < selected.length; i++) {
    const { round, bestScore, avgScore, promptSim, respSim, sources } = selected[i];
    console.log(`  #${i + 1} | round ${round.turnIndex} | best=${bestScore.toFixed(4)} avg=${avgScore.toFixed(4)}`);
    console.log(`      prompt: ${round.userPrompt.slice(0, 100)}${round.userPrompt.length > 100 ? "..." : ""}`);
    console.log(`      sources: ${sources.join(", ")}`);

    if (showContent) {
      console.log(`      response: ${round.responseSequence.slice(0, 200)}${round.responseSequence.length > 200 ? "..." : ""}`);
    }
    console.log();
  }

  // Order selected by timestamp for narrative coherence
  if (chrono) {
    selected.sort((a, b) => a.round.turnIndex - b.round.turnIndex);
    console.log(`📅 Ordered by timestamp (chronological)`);
    console.log(`    Round range: ${selected[0].round.turnIndex} → ${selected[selected.length - 1].round.turnIndex}\n`);
  }

  // Summary stats
  const meanScore = ranked.reduce((s, r) => s + r.bestScore, 0) / ranked.length;
  const medianScore = ranked.length > 0
    ? ranked[Math.floor(ranked.length / 2)].bestScore
    : 0;
  console.log(`📊 Score distribution across all ${ranked.length} rounds:`);
  console.log(`    Best:  ${ranked[0]?.bestScore.toFixed(4) ?? "N/A"}`);
  console.log(`    Top 5: ${ranked.slice(0, 5).map(r => r.bestScore.toFixed(4)).join(", ")}`);
  console.log(`    Median: ${medianScore.toFixed(4)}`);
  console.log(`    Mean:  ${meanScore.toFixed(4)}`);
  console.log(`    Worst: ${ranked[ranked.length - 1]?.bestScore.toFixed(4) ?? "N/A"}`);

}

main().catch((err) => {
  console.error("❌ Error:", err);
  process.exit(1);
});
