export type RollingFold<T> = {
  fold: number;
  train: T[];
  calibration: T[];
  holdout: T[];
};

/** Builds expanding-window folds without allowing an item into two partitions. */
export function createRollingFolds<T>(
  ordered: T[],
  options: { minimumTrain: number; calibrationSize: number; holdoutSize: number; folds: number },
): RollingFold<T>[] {
  const { minimumTrain, calibrationSize, holdoutSize, folds } = options;
  if ([minimumTrain, calibrationSize, holdoutSize, folds].some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error("Rolling-fold sizes must be positive integers.");
  }
  const available = ordered.length - minimumTrain - calibrationSize;
  const possible = Math.floor(available / holdoutSize);
  const foldCount = Math.min(folds, Math.max(0, possible));
  const result: RollingFold<T>[] = [];
  for (let fold = 0; fold < foldCount; fold += 1) {
    const holdoutEnd = ordered.length - (foldCount - fold - 1) * holdoutSize;
    const holdoutStart = holdoutEnd - holdoutSize;
    const calibrationStart = holdoutStart - calibrationSize;
    result.push({
      fold: fold + 1,
      train: ordered.slice(0, calibrationStart),
      calibration: ordered.slice(calibrationStart, holdoutStart),
      holdout: ordered.slice(holdoutStart, holdoutEnd),
    });
  }
  return result;
}

export function randomExpectedMatches(picks: number, winners: number, vocabularySize: number): number {
  return (picks * winners) / vocabularySize;
}

export type PairedGateObservation = {
  groupKey: string;
  neuralMatches: number;
  ensembleMatches: number;
};

export type ReliabilityGateDiagnostics = {
  selectedMethod: "ensemble" | "neural";
  neuralAverageMatches: number;
  ensembleAverageMatches: number;
  pairedAdvantage: number;
  confidenceLow: number;
  confidenceHigh: number;
  confidenceLevel: number;
  minimumAdvantage: number;
  independentGroupCount: number;
  gateSampleCount: number;
  bootstrapIterations: number;
  bootstrapSeed: number;
  reason:
    | "ensemble_selected"
    | "ensemble_not_higher"
    | "advantage_below_margin"
    | "confidence_interval_includes_zero"
    | "insufficient_groups";
};

function seededUnitValues(seed: number): () => number {
  let state = Math.max(1, Math.floor(seed)) % 2147483647;
  return () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  };
}

function quantile(sorted: number[], probability: number): number {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function evaluateReliabilityGate(
  observations: PairedGateObservation[],
  options: {
    confidenceLevel: number;
    minimumAdvantage: number;
    minimumGroups: number;
    bootstrapIterations: number;
    seed: number;
  },
): ReliabilityGateDiagnostics {
  const groups = new Map<string, number[]>();
  for (const observation of observations) {
    const differences = groups.get(observation.groupKey) ?? [];
    differences.push(observation.ensembleMatches - observation.neuralMatches);
    groups.set(observation.groupKey, differences);
  }
  const groupAdvantages = [...groups.values()].map((values) =>
    values.reduce((sum, value) => sum + value, 0) / values.length,
  );
  const neuralAverageMatches = observations.length
    ? observations.reduce((sum, value) => sum + value.neuralMatches, 0) / observations.length
    : 0;
  const ensembleAverageMatches = observations.length
    ? observations.reduce((sum, value) => sum + value.ensembleMatches, 0) / observations.length
    : 0;
  const pairedAdvantage = groupAdvantages.length
    ? groupAdvantages.reduce((sum, value) => sum + value, 0) / groupAdvantages.length
    : 0;
  const bootstrapMeans: number[] = [];
  const random = seededUnitValues(options.seed);
  if (groupAdvantages.length > 0) {
    for (let iteration = 0; iteration < options.bootstrapIterations; iteration += 1) {
      let total = 0;
      for (let index = 0; index < groupAdvantages.length; index += 1) {
        total += groupAdvantages[Math.floor(random() * groupAdvantages.length)];
      }
      bootstrapMeans.push(total / groupAdvantages.length);
    }
  }
  bootstrapMeans.sort((a, b) => a - b);
  const alpha = 1 - options.confidenceLevel;
  const confidenceLow = quantile(bootstrapMeans, alpha / 2);
  const confidenceHigh = quantile(bootstrapMeans, 1 - alpha / 2);

  let reason: ReliabilityGateDiagnostics["reason"] = "ensemble_selected";
  if (groupAdvantages.length < options.minimumGroups) reason = "insufficient_groups";
  else if (ensembleAverageMatches <= neuralAverageMatches) reason = "ensemble_not_higher";
  else if (pairedAdvantage < options.minimumAdvantage) reason = "advantage_below_margin";
  else if (confidenceLow <= 0) reason = "confidence_interval_includes_zero";

  return {
    selectedMethod: reason === "ensemble_selected" ? "ensemble" : "neural",
    neuralAverageMatches,
    ensembleAverageMatches,
    pairedAdvantage,
    confidenceLow,
    confidenceHigh,
    confidenceLevel: options.confidenceLevel,
    minimumAdvantage: options.minimumAdvantage,
    independentGroupCount: groupAdvantages.length,
    gateSampleCount: observations.length,
    bootstrapIterations: options.bootstrapIterations,
    bootstrapSeed: options.seed,
    reason,
  };
}

export function createRollingEventFolds<T>(
  ordered: T[],
  groupKey: (item: T) => string,
  options: { minimumTrainGroups: number; calibrationGroups: number; holdoutGroups: number; folds: number },
): RollingFold<T>[] {
  const groups: T[][] = [];
  for (const item of ordered) {
    const previous = groups.at(-1);
    if (!previous || groupKey(previous[0]) !== groupKey(item)) groups.push([item]);
    else previous.push(item);
  }
  const groupFolds = createRollingFolds(groups, {
    minimumTrain: options.minimumTrainGroups,
    calibrationSize: options.calibrationGroups,
    holdoutSize: options.holdoutGroups,
    folds: options.folds,
  });
  return groupFolds.map((fold) => ({
    fold: fold.fold,
    train: fold.train.flat(),
    calibration: fold.calibration.flat(),
    holdout: fold.holdout.flat(),
  }));
}
