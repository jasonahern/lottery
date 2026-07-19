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
