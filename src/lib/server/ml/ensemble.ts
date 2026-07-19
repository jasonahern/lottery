export type EnsembleMethod = "neural" | "frequency" | "heuristic" | "random";

export type EnsembleWeights = Record<EnsembleMethod, number>;
export type EnsembleScores = Record<EnsembleMethod, number[]>;

export const DEFAULT_ENSEMBLE_WEIGHTS: EnsembleWeights = {
  neural: 0.45,
  frequency: 0.2,
  heuristic: 0.35,
  random: 0,
};

export const ENSEMBLE_LEARNING_RATE = 1;
export const MAX_ENSEMBLE_SIGNAL_WEIGHT = 0.6;

export function normalizeScores(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const ranked = scores
    .map((score, index) => ({ score: Number.isFinite(score) ? score : 0, index }))
    .sort((a, b) => a.score - b.score || b.index - a.index);
  const normalized = new Array(scores.length).fill(0);
  const denominator = Math.max(1, scores.length - 1);
  let start = 0;
  while (start < ranked.length) {
    let end = start + 1;
    while (end < ranked.length && ranked[end].score === ranked[start].score) end += 1;
    const averageRank = (start + end - 1) / 2 / denominator;
    for (let rank = start; rank < end; rank += 1) {
      normalized[ranked[rank].index] = averageRank;
    }
    start = end;
  }
  return normalized;
}

export function normalizeWeights(weights: EnsembleWeights): EnsembleWeights {
  const safe = Object.fromEntries(
    Object.entries(weights).map(([method, value]) => [method, Math.max(0, value)]),
  ) as EnsembleWeights;
  const total = Object.values(safe).reduce((sum, value) => sum + value, 0);
  if (total <= 0) return { ...DEFAULT_ENSEMBLE_WEIGHTS };
  return Object.fromEntries(
    Object.entries(safe).map(([method, value]) => [method, value / total]),
  ) as EnsembleWeights;
}

export function blendEnsembleScores(
  componentScores: EnsembleScores,
  weights: EnsembleWeights,
): number[] {
  const normalized = Object.fromEntries(
    Object.entries(componentScores).map(([method, scores]) => [method, normalizeScores(scores)]),
  ) as EnsembleScores;
  const normalizedWeights = getEffectiveEnsembleWeightsFromNormalized(normalized, weights);
  const size = normalized.neural.length;
  return Array.from({ length: size }, (_, index) =>
    (Object.keys(normalizedWeights) as EnsembleMethod[]).reduce(
      (sum, method) => sum + normalizedWeights[method] * (normalized[method][index] ?? 0),
      0,
    ),
  );
}

function scoresAreEquivalent(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every(
    (value, index) => Math.abs(value - (right[index] ?? 0)) < 1e-12,
  );
}

export function collapseDuplicateExpertWeights(
  calibrationScores: EnsembleScores[],
  weights: EnsembleWeights,
): EnsembleWeights {
  const normalizedSets = calibrationScores.map((scores) =>
    Object.fromEntries(
      Object.entries(scores).map(([method, values]) => [method, normalizeScores(values)]),
    ) as EnsembleScores,
  );
  const collapsed = { ...normalizeWeights(weights) };
  const methods = (Object.keys(collapsed) as EnsembleMethod[]).filter(
    (method) => collapsed[method] > 0,
  );

  methods.forEach((method, index) => {
    if (collapsed[method] <= 0) return;
    for (const candidate of methods.slice(index + 1)) {
      if (collapsed[candidate] <= 0) continue;
      const duplicate = normalizedSets.length > 0 && normalizedSets.every((scores) =>
        scoresAreEquivalent(scores[method], scores[candidate]),
      );
      if (duplicate) {
        collapsed[method] += collapsed[candidate];
        collapsed[candidate] = 0;
      }
    }
  });

  return normalizeWeights(collapsed);
}

function capGroupWeights(rawWeights: number[], maximum: number): number[] {
  if (rawWeights.length <= 1) return rawWeights.map(() => 1);
  const result = rawWeights.map(() => 0);
  let remaining = rawWeights.map((_, index) => index);
  let remainingMass = 1;
  while (remaining.length > 0) {
    const rawTotal = remaining.reduce((sum, index) => sum + rawWeights[index], 0);
    const proposed = remaining.map((index) => ({
      index,
      value: rawTotal > 0
        ? (rawWeights[index] / rawTotal) * remainingMass
        : remainingMass / remaining.length,
    }));
    const capped = proposed.filter((item) => item.value > maximum);
    if (capped.length === 0) {
      proposed.forEach((item) => { result[item.index] = item.value; });
      break;
    }
    capped.forEach((item) => { result[item.index] = maximum; });
    remainingMass -= maximum * capped.length;
    const cappedIndexes = new Set(capped.map((item) => item.index));
    remaining = remaining.filter((index) => !cappedIndexes.has(index));
  }
  return result;
}

function getEffectiveEnsembleWeightsFromNormalized(
  normalizedScores: EnsembleScores,
  weights: EnsembleWeights,
): EnsembleWeights {
  const methods = (Object.keys(weights) as EnsembleMethod[]).filter(
    (method) => weights[method] > 0,
  );
  const groups: EnsembleMethod[][] = [];
  for (const method of methods) {
    const group = groups.find((candidate) =>
      scoresAreEquivalent(normalizedScores[candidate[0]], normalizedScores[method]),
    );
    if (group) group.push(method);
    else groups.push([method]);
  }
  if (groups.length === 0) {
    return getEffectiveEnsembleWeightsFromNormalized(
      normalizedScores,
      DEFAULT_ENSEMBLE_WEIGHTS,
    );
  }
  const groupRawWeights = groups.map((group) =>
    group.reduce((sum, method) => sum + weights[method], 0),
  );
  const groupWeights = capGroupWeights(groupRawWeights, MAX_ENSEMBLE_SIGNAL_WEIGHT);
  const effective: EnsembleWeights = { neural: 0, frequency: 0, heuristic: 0, random: 0 };
  groups.forEach((group, groupIndex) => {
    const groupRaw = group.reduce((sum, method) => sum + weights[method], 0);
    group.forEach((method) => {
      effective[method] = groupWeights[groupIndex] *
        (groupRaw > 0 ? weights[method] / groupRaw : 1 / group.length);
    });
  });
  return effective;
}

export function getEffectiveEnsembleWeights(
  componentScores: EnsembleScores,
  weights: EnsembleWeights,
): EnsembleWeights {
  const normalized = Object.fromEntries(
    Object.entries(componentScores).map(([method, scores]) => [method, normalizeScores(scores)]),
  ) as EnsembleScores;
  return getEffectiveEnsembleWeightsFromNormalized(normalized, weights);
}

/** Multiplicative-weights update. Rewards must be in [0, 1]. */
export function updateEnsembleWeights(
  weights: EnsembleWeights,
  rewards: Record<EnsembleMethod, number>,
  learningRate = ENSEMBLE_LEARNING_RATE,
): EnsembleWeights {
  const current = normalizeWeights(weights);
  const updated = Object.fromEntries(
    (Object.keys(current) as EnsembleMethod[]).map((method) => [
      method,
      current[method] * Math.exp(learningRate * Math.max(0, Math.min(1, rewards[method]))),
    ]),
  ) as EnsembleWeights;
  return normalizeWeights(updated);
}

export function buildSeededRandomScores(size: number, seed: number): number[] {
  let state = Math.max(1, Math.floor(seed)) % 2147483647;
  return Array.from({ length: size }, () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  });
}

export function calculateCalibrationReward(
  scores: number[],
  numbers: number[],
  actualNumbers: number[],
): number {
  if (actualNumbers.length === 0 || numbers.length === 0) return 0;
  const ranked = scores
    .map((score, index) => ({ score, number: numbers[index] }))
    .sort((a, b) => b.score - a.score || a.number - b.number);
  const rankByNumber = new Map(ranked.map((item, rank) => [item.number, rank]));
  const denominator = Math.max(1, numbers.length - 1);
  return actualNumbers.reduce((sum, number) => {
    const rank = rankByNumber.get(number) ?? numbers.length - 1;
    return sum + (numbers.length - 1 - rank) / denominator;
  }, 0) / actualNumbers.length;
}
