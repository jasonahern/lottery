export type NnRunStatus = "queued" | "running" | "completed" | "failed";

export type TrainingPreset = "small" | "medium" | "large" | "xlarge";

export type TrainingConfig = {
  holdoutWeeks: number;
  windowSize: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  dropoutRate: number;
  activation: "relu" | "tanh" | "sigmoid";
  hiddenLayers: number[];
  modelFamily: string;
  currentEraStartDate: string | null;
  trainingSeed: number;
  positiveClassWeight: number;
  earlyStoppingPatience: number;
  earlyStoppingMinDelta: number;
};

export const DEFAULT_PROFILE_NAME = "balanced_v1";

export const PRESET_HIDDEN_LAYERS: Record<TrainingPreset, number[]> = {
  small: [64, 32],
  medium: [128, 64],
  large: [256, 128],
  xlarge: [512, 256],
};

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  holdoutWeeks: 4,
  windowSize: 10,
  epochs: 80,
  batchSize: 32,
  learningRate: 0.001,
  dropoutRate: 0.25,
  activation: "relu",
  hiddenLayers: PRESET_HIDDEN_LAYERS.medium,
  modelFamily: "mlp_v2",
  currentEraStartDate: "2015-10-09",
  trainingSeed: 42,
  positiveClassWeight: 53 / 6,
  earlyStoppingPatience: 12,
  earlyStoppingMinDelta: 0.0001,
};

export const MAX_TRAINING_LIMITS = {
  maxHiddenLayers: 4,
  maxNeuronsPerLayer: 1024,
  maxEpochs: 500,
  maxBatchSize: 1024,
};

function isFinitePositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) > 0;
}

function isFinitePositiveNumber(value: unknown): value is number {
  return Number.isFinite(value) && Number(value) > 0;
}

export function parseHiddenLayersInput(value: string): number[] {
  const parsed = value
    .split(",")
    .map((n) => Number(n.trim()))
    .filter((n) => Number.isFinite(n) && n > 0)
    .map((n) => Math.floor(n));

  return parsed;
}

export function resolveHiddenLayers(
  preset?: TrainingPreset,
  customHiddenLayers?: number[],
): number[] {
  if (customHiddenLayers && customHiddenLayers.length > 0) {
    return customHiddenLayers;
  }

  if (preset) {
    return PRESET_HIDDEN_LAYERS[preset];
  }

  return DEFAULT_TRAINING_CONFIG.hiddenLayers;
}

export function validateTrainingConfig(config: TrainingConfig): string[] {
  const errors: string[] = [];

  if (!isFinitePositiveInteger(config.holdoutWeeks)) {
    errors.push("holdoutWeeks must be a positive integer.");
  }

  if (!isFinitePositiveInteger(config.windowSize)) {
    errors.push("windowSize must be a positive integer.");
  }

  if (!isFinitePositiveInteger(config.epochs)) {
    errors.push("epochs must be a positive integer.");
  } else if (config.epochs > MAX_TRAINING_LIMITS.maxEpochs) {
    errors.push(`epochs must be <= ${MAX_TRAINING_LIMITS.maxEpochs}.`);
  }

  if (!isFinitePositiveInteger(config.batchSize)) {
    errors.push("batchSize must be a positive integer.");
  } else if (config.batchSize > MAX_TRAINING_LIMITS.maxBatchSize) {
    errors.push(`batchSize must be <= ${MAX_TRAINING_LIMITS.maxBatchSize}.`);
  }

  if (!isFinitePositiveNumber(config.learningRate)) {
    errors.push("learningRate must be a positive number.");
  }

  if (
    !Number.isFinite(config.dropoutRate) ||
    config.dropoutRate < 0 ||
    config.dropoutRate >= 1
  ) {
    errors.push("dropoutRate must be between 0 (inclusive) and 1 (exclusive).");
  }

  if (!Array.isArray(config.hiddenLayers) || config.hiddenLayers.length === 0) {
    errors.push("hiddenLayers must contain at least one layer.");
  } else {
    if (config.hiddenLayers.length > MAX_TRAINING_LIMITS.maxHiddenLayers) {
      errors.push(
        `hiddenLayers must have at most ${MAX_TRAINING_LIMITS.maxHiddenLayers} entries.`,
      );
    }

    config.hiddenLayers.forEach((neurons, idx) => {
      if (!isFinitePositiveInteger(neurons)) {
        errors.push(`hiddenLayers[${idx}] must be a positive integer.`);
        return;
      }

      if (neurons > MAX_TRAINING_LIMITS.maxNeuronsPerLayer) {
        errors.push(
          `hiddenLayers[${idx}] must be <= ${MAX_TRAINING_LIMITS.maxNeuronsPerLayer}.`,
        );
      }
    });
  }

  if (!config.modelFamily.trim()) {
    errors.push("modelFamily cannot be empty.");
  }

  if (!isFinitePositiveInteger(config.trainingSeed)) {
    errors.push("trainingSeed must be a positive integer.");
  }
  if (!isFinitePositiveNumber(config.positiveClassWeight)) {
    errors.push("positiveClassWeight must be a positive number.");
  }
  if (!isFinitePositiveInteger(config.earlyStoppingPatience)) {
    errors.push("earlyStoppingPatience must be a positive integer.");
  }
  if (!isFinitePositiveNumber(config.earlyStoppingMinDelta)) {
    errors.push("earlyStoppingMinDelta must be a positive number.");
  }

  if (
    config.currentEraStartDate !== null &&
    Number.isNaN(Date.parse(config.currentEraStartDate))
  ) {
    errors.push("currentEraStartDate must be a valid date string or null.");
  }

  return errors;
}
