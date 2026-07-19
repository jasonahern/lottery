import { desc, inArray } from "drizzle-orm";

import { db } from "$lib/server/db";
import { backfillNormalizedFromLegacyIfNeeded } from "$lib/server/ml/data";
import { lotteryDrawBalls, lotteryDraws } from "$lib/server/db/schema";

export const load = async () => {
  backfillNormalizedFromLegacyIfNeeded();

  const draws = db
    .select()
    .from(lotteryDraws)
    .orderBy(desc(lotteryDraws.drawDate), desc(lotteryDraws.id))
    .all();

  const drawIds = draws.map((d) => d.id);
  const allBalls = drawIds.length
    ? db
        .select()
        .from(lotteryDrawBalls)
        .where(inArray(lotteryDrawBalls.drawId, drawIds))
        .all()
    : [];

  const ballsByDraw = new Map<number, typeof allBalls>();
  for (const ball of allBalls) {
    const arr = ballsByDraw.get(ball.drawId) ?? [];
    arr.push(ball);
    ballsByDraw.set(ball.drawId, arr);
  }

  const entries = draws.map((draw) => {
    const ordered = [...(ballsByDraw.get(draw.id) ?? [])].sort(
      (a, b) => a.position - b.position,
    );
    const mainBalls = ordered.filter((b) => !b.isBonus).map((b) => b.value);
    const bonusBall = ordered.find((b) => b.isBonus)?.value ?? null;

    return {
      ...draw,
      mainBalls,
      bonusBall,
    };
  });

  return {
    entries,
  };
};
