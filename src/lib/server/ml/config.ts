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
  trainingSeeds: number[];
  positiveClassWeight: number;
  earlyStoppingPatience: number;
  earlyStoppingMinDelta: number;
  enableRollingBacktest: boolean;
  rollingFolds: number;
  rollingHoldoutWeeks: number;
  minimumTrainingWeeks: number;
  reliabilityConfidenceLevel: number;
  reliabilityMinimumAdvantage: number;
  reliabilityMinimumGroups: number;
  reliabilityBootstrapIterations: number;
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
  trainingSeeds: [42],
  positiveClassWeight: 53 / 6,
  earlyStoppingPatience: 12,
  earlyStoppingMinDelta: 0.0001,
  enableRollingBacktest: false,
  rollingFolds: 4,
  rollingHoldoutWeeks: 8,
  minimumTrainingWeeks: 104,
  reliabilityConfidenceLevel: 0.95,
  reliabilityMinimumAdvantage: 0.05,
  reliabilityMinimumGroups: 30,
  reliabilityBootstrapIterations: 2000,
};

export const MAX_TRAINING_LIMITS = {
  maxHiddenLayers: 4,
  maxNeuronsPerLayer: 1024,
  maxEpochs: 500,
  maxBatchSize: 1024,
  maxTrainingSeeds: 5,
};

export function parseTrainingSeedsInput(value: string): number[] {
  return [...new Set(value.split(",").map((part) => Number(part.trim())).filter((seed) => Number.isInteger(seed) && seed > 0))];
}

export function resolveTrainingSeeds(config: Pick<TrainingConfig, "trainingSeed" | "trainingSeeds">): number[] {
  return config.trainingSeeds.length > 0 ? config.trainingSeeds : [config.trainingSeed];
}

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
  if (!Array.isArray(config.trainingSeeds) || config.trainingSeeds.length === 0) {
    errors.push("trainingSeeds must contain at least one seed.");
  } else if (config.trainingSeeds.length > MAX_TRAINING_LIMITS.maxTrainingSeeds) {
    errors.push(`trainingSeeds must contain at most ${MAX_TRAINING_LIMITS.maxTrainingSeeds} seeds.`);
  } else if (new Set(config.trainingSeeds).size !== config.trainingSeeds.length || config.trainingSeeds.some((seed) => !isFinitePositiveInteger(seed))) {
    errors.push("trainingSeeds must contain unique positive integers.");
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
  if (typeof config.enableRollingBacktest !== "boolean") errors.push("enableRollingBacktest must be boolean.");
  if (!isFinitePositiveInteger(config.rollingFolds) || config.rollingFolds > 10) errors.push("rollingFolds must be between 1 and 10.");
  if (!isFinitePositiveInteger(config.rollingHoldoutWeeks)) errors.push("rollingHoldoutWeeks must be a positive integer.");
  if (!isFinitePositiveInteger(config.minimumTrainingWeeks)) errors.push("minimumTrainingWeeks must be a positive integer.");
  if (!(config.reliabilityConfidenceLevel > 0 && config.reliabilityConfidenceLevel < 1)) errors.push("reliabilityConfidenceLevel must be between 0 and 1.");
  if (!(config.reliabilityMinimumAdvantage >= 0)) errors.push("reliabilityMinimumAdvantage must be non-negative.");
  if (!isFinitePositiveInteger(config.reliabilityMinimumGroups)) errors.push("reliabilityMinimumGroups must be a positive integer.");
  if (!isFinitePositiveInteger(config.reliabilityBootstrapIterations) || config.reliabilityBootstrapIterations > 10000) errors.push("reliabilityBootstrapIterations must be between 1 and 10000.");

  if (
    config.currentEraStartDate !== null &&
    Number.isNaN(Date.parse(config.currentEraStartDate))
  ) {
    errors.push("currentEraStartDate must be a valid date string or null.");
  }

  return errors;
}
