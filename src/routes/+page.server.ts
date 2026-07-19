import { mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { fail } from "@sveltejs/kit";

import { db } from "$lib/server/db";
import { parseCsvRowsWithRounds } from "$lib/server/ml/data";
import { resolvePendingFeedback } from "$lib/server/ml/policy";
import {
  lotteryDrawBalls,
  lotteryDraws,
  lotteryEntries,
} from "$lib/server/db/schema";
import type { Actions } from "./$types";

const LOTTERY_URL =
  "https://lottery.merseyworld.com/cgi-bin/lottery?days=2&Machine=Z&Ballset=0&order=0&show=1&year=0&display=CSV";
const OUTPUT_DIR = "lotterydata";
const OUTPUT_FILE = "LotteryData.csv";

/**
 * The lottery URL returns an HTML page with the CSV data inside a <PRE> block.
 * This function extracts that inner text, then removes the first two lines
 * (the title line and the blank separator line) to produce clean CSV text.
 */
function extractCsv(html: string): string {
  const match = html.match(/<PRE>([\s\S]*?)<\/PRE>/i);
  if (!match) {
    throw new Error("Could not find CSV data in the downloaded response.");
  }

  const lines = match[1].split(/\r?\n/);

  // Remove the first 2 rows: the title line and the blank line that follow <PRE>
  return lines.slice(2).join("\n").trimEnd();
}

function sanitizeCsv(csv: string): { cleanCsv: string; dataRows: string[] } {
  const lines = csv
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    throw new Error("CSV did not contain any data rows.");
  }

  const headerRow = lines[0];

  // Keep only real draw rows (e.g. "3188, Sat,11,Jul,2026,...") and drop footer text lines.
  const dataRows = lines.slice(1).filter((line) => /^\d+\s*,/.test(line));

  if (dataRows.length === 0) {
    throw new Error("CSV did not contain valid lottery draw rows.");
  }

  return {
    cleanCsv: [headerRow, ...dataRows].join("\n"),
    dataRows,
  };
}

function reloadLotteryDataset(dataRows: string[]): number {
  const deduplicated = parseCsvRowsWithRounds(dataRows);
  const parsedRows = deduplicated.map((item) => item.parsed);
  const rowsToInsert = deduplicated.map((item) => item.row);

  db.transaction((tx) => {
    tx.delete(lotteryDrawBalls).run();
    tx.delete(lotteryDraws).run();
    tx.delete(lotteryEntries).run();

    const insertedLegacyEntries = tx
      .insert(lotteryEntries)
      .values(rowsToInsert.map((row) => ({ numbers: row })))
      .returning({ id: lotteryEntries.id })
      .all();

    for (let i = 0; i < parsedRows.length; i += 1) {
      const parsed = parsedRows[i];
      const legacyId = insertedLegacyEntries[i]?.id;

      const insertedDraw = tx
        .insert(lotteryDraws)
        .values({
          legacyEntryId: legacyId,
          drawNumber: parsed.draw.drawNumber,
          drawDate: parsed.draw.drawDate,
          dayName: parsed.draw.dayName,
          gameName: parsed.draw.gameName,
          machine: parsed.draw.machine,
          wins: parsed.draw.wins,
          ballSet: parsed.draw.ballSet,
          drawSequence: parsed.draw.drawSequence,
          drawRound: parsed.draw.drawRound,
          jackpotAmount: parsed.draw.jackpotAmount,
          sourceRow: parsed.draw.sourceRow,
        })
        .returning({ id: lotteryDraws.id })
        .get();

      if (!insertedDraw) {
        throw new Error("Failed to insert normalized draw row.");
      }

      tx.insert(lotteryDrawBalls)
        .values(
          parsed.balls.map((ball) => ({
            drawId: insertedDraw.id,
            position: ball.position,
            value: ball.value,
            isBonus: ball.isBonus,
          })),
        )
        .run();
    }
  });

  return rowsToInsert.length;
}

export const actions: Actions = {
  default: async ({ fetch }) => {
    let response: Response;

    try {
      response = await fetch(LOTTERY_URL, {
        headers: { "User-Agent": "Mozilla/5.0" },
      });
    } catch {
      return fail(502, {
        success: false,
        message: "Network error: could not reach the lottery server.",
      });
    }

    if (!response.ok) {
      return fail(response.status, {
        success: false,
        message: `Download failed — server responded with HTTP ${response.status}.`,
      });
    }

    let cleanCsv: string;
    let dataRows: string[];
    try {
      const html = await response.text();
      const extractedCsv = extractCsv(html);
      ({ cleanCsv, dataRows } = sanitizeCsv(extractedCsv));
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to process downloaded data.";
      return fail(500, { success: false, message });
    }

    const outputDir = resolve(process.cwd(), OUTPUT_DIR);
    const outputPath = resolve(outputDir, OUTPUT_FILE);

    try {
      await mkdir(outputDir, { recursive: true });
      // Delete existing file if present, then write fresh copy
      await rm(outputPath, { force: true });
      await writeFile(outputPath, cleanCsv, "utf8");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unknown error writing file.";
      return fail(500, {
        success: false,
        message: `File write error: ${message}`,
      });
    }

    let importedRows = 0;
    let resolvedFeedback = 0;
    try {
      importedRows = reloadLotteryDataset(dataRows);
      resolvedFeedback = resolvePendingFeedback().resolvedCount;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unknown database error.";
      return fail(500, {
        success: false,
        message: `Database reload error: ${message}`,
      });
    }

    return {
      success: true,
      messages: [
        `✓ Saved ${OUTPUT_FILE} to ./${OUTPUT_DIR}/`,
        `✓ Reloaded lottery.db with ${importedRows} records using Drizzle.`,
        `✓ Resolved ${resolvedFeedback} pending prediction feedback outcomes.`,
      ],
      message: `✓ Saved ${OUTPUT_FILE} and reloaded lottery.db.`,
    };
  },
};
