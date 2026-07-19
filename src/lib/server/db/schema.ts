import { index, integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const lotteryEntries = sqliteTable("lottery_entries", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  numbers: text("numbers").notNull(),
  createdAt: integer("created_at", { mode: "timestamp_ms" })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const lotteryDraws = sqliteTable(
  "lottery_draws",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    legacyEntryId: integer("legacy_entry_id").references(
      () => lotteryEntries.id,
      {
        onDelete: "set null",
      },
    ),
    drawNumber: integer("draw_number").notNull(),
    drawDate: integer("draw_date", { mode: "timestamp_ms" }).notNull(),
    dayName: text("day_name"),
    gameName: text("game_name"),
    machine: text("machine"),
    wins: integer("wins"),
    ballSet: integer("ball_set"),
    drawSequence: integer("draw_sequence"),
    drawRound: integer("draw_round").notNull().default(1),
    jackpotAmount: integer("jackpot_amount"),
    sourceRow: text("source_row").notNull(),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    drawNumberIdx: index("lottery_draws_draw_number_idx").on(table.drawNumber),
    drawDateIdx: index("lottery_draws_draw_date_idx").on(table.drawDate),
    gameDrawIdx: index("lottery_draws_game_draw_idx").on(
      table.gameName,
      table.drawNumber,
    ),
    sourceRowUniqueIdx: uniqueIndex("lottery_draws_source_row_unique_idx").on(
      table.sourceRow,
    ),
    gameDrawRoundUniqueIdx: uniqueIndex("lottery_draws_game_draw_round_unique_idx").on(
      table.gameName,
      table.drawNumber,
      table.drawRound,
    ),
  }),
);

export const lotteryDrawBalls = sqliteTable(
  "lottery_draw_balls",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    drawId: integer("draw_id")
      .notNull()
      .references(() => lotteryDraws.id, { onDelete: "cascade" }),
    position: integer("position").notNull(),
    value: integer("value").notNull(),
    isBonus: integer("is_bonus", { mode: "boolean" }).notNull().default(false),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    drawPositionIdx: index("lottery_draw_balls_draw_position_idx").on(
      table.drawId,
      table.position,
    ),
    valueIdx: index("lottery_draw_balls_value_idx").on(table.value),
  }),
);

export const nnTrainingRuns = sqliteTable(
  "nn_training_runs",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    status: text("status").notNull().default("queued"),
    startedAt: integer("started_at", { mode: "timestamp_ms" }),
    endedAt: integer("ended_at", { mode: "timestamp_ms" }),
    holdoutWeeks: integer("holdout_weeks").notNull(),
    windowSize: integer("window_size").notNull(),
    totalEpochs: integer("total_epochs").notNull(),
    currentEpoch: integer("current_epoch").notNull().default(0),
    trainSamples: integer("train_samples"),
    testSamples: integer("test_samples"),
    samplesTotal: integer("samples_total"),
    samplesProcessed: integer("samples_processed").notNull().default(0),
    modelFamily: text("model_family").notNull().default("mlp_v1"),
    hiddenLayersJson: text("hidden_layers_json").notNull(),
    hyperparamsJson: text("hyperparams_json").notNull(),
    paramCount: integer("param_count"),
    finalTrainLoss: text("final_train_loss"),
    finalValLoss: text("final_val_loss"),
    holdoutScore: text("holdout_score"),
    neuralHoldoutScore: text("neural_holdout_score"),
    ensembleHoldoutScore: text("ensemble_holdout_score"),
    frequencyHoldoutScore: text("frequency_holdout_score"),
    heuristicHoldoutScore: text("heuristic_holdout_score"),
    randomHoldoutScore: text("random_holdout_score"),
    bestEpoch: integer("best_epoch"),
    inputEncoding: text("input_encoding").notNull().default("sorted_scalar_v1"),
    lossVersion: text("loss_version").notNull().default("binary_crossentropy_v1"),
    trainingSeed: integer("training_seed"),
    isValid: integer("is_valid", { mode: "boolean" }).notNull().default(true),
    invalidatedAt: integer("invalidated_at", { mode: "timestamp_ms" }),
    invalidationReason: text("invalidation_reason"),
    modelArtifactPath: text("model_artifact_path"),
    modelSha256: text("model_sha256"),
    errorMessage: text("error_message"),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    statusIdx: index("nn_training_runs_status_idx").on(table.status),
    createdAtIdx: index("nn_training_runs_created_at_idx").on(table.createdAt),
    holdoutScoreIdx: index("nn_training_runs_holdout_score_idx").on(
      table.holdoutScore,
    ),
  }),
);

export const nnEpochMetrics = sqliteTable(
  "nn_epoch_metrics",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: integer("run_id")
      .notNull()
      .references(() => nnTrainingRuns.id, { onDelete: "cascade" }),
    epoch: integer("epoch").notNull(),
    trainLoss: text("train_loss"),
    valLoss: text("val_loss"),
    elapsedMs: integer("elapsed_ms"),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    runEpochIdx: index("nn_epoch_metrics_run_epoch_idx").on(
      table.runId,
      table.epoch,
    ),
  }),
);

export const nnTestResults = sqliteTable(
  "nn_test_results",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: integer("run_id")
      .notNull()
      .references(() => nnTrainingRuns.id, { onDelete: "cascade" }),
    drawId: integer("draw_id"),
    drawDate: text("draw_date"),
    predictedNumbers: text("predicted_numbers").notNull(),
    actualNumbers: text("actual_numbers").notNull(),
    frequencyPredictedNumbers: text("frequency_predicted_numbers"),
    frequencyMatchCount: integer("frequency_match_count"),
    randomPredictedNumbers: text("random_predicted_numbers"),
    randomMatchCount: integer("random_match_count"),
    matchCount: integer("match_count").notNull().default(0),
    topKHit: integer("top_k_hit", { mode: "boolean" }).notNull().default(false),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    runDrawIdx: index("nn_test_results_run_draw_idx").on(
      table.runId,
      table.drawDate,
    ),
  }),
);

export const nnPredictionEvaluations = sqliteTable(
  "nn_prediction_evaluations",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: integer("run_id")
      .notNull()
      .references(() => nnTrainingRuns.id, { onDelete: "cascade" }),
    drawId: integer("draw_id"),
    drawDate: text("draw_date"),
    method: text("method").notNull(),
    predictedNumbers: text("predicted_numbers").notNull(),
    actualNumbers: text("actual_numbers").notNull(),
    matchCount: integer("match_count").notNull().default(0),
    topKHit: integer("top_k_hit", { mode: "boolean" }).notNull().default(false),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    runMethodIdx: index("nn_prediction_evaluations_run_method_idx").on(
      table.runId,
      table.method,
    ),
    runDrawMethodIdx: index("nn_prediction_evaluations_run_draw_method_idx").on(
      table.runId,
      table.drawId,
      table.method,
    ),
  }),
);

export const nnPolicySettings = sqliteTable(
  "nn_policy_settings",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    key: text("key").notNull(),
    value: text("value").notNull(),
    updatedAt: integer("updated_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    keyIdx: index("nn_policy_settings_key_idx").on(table.key),
    updatedAtIdx: index("nn_policy_settings_updated_at_idx").on(
      table.updatedAt,
    ),
  }),
);

export const nnPredictionDecisions = sqliteTable(
  "nn_prediction_decisions",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: integer("run_id").references(() => nnTrainingRuns.id, {
      onDelete: "set null",
    }),
    targetDrawId: integer("target_draw_id").references(() => lotteryDraws.id, {
      onDelete: "set null",
    }),
    targetDrawNumber: integer("target_draw_number").notNull(),
    basedOnDrawNumber: integer("based_on_draw_number").notNull(),
    basedOnDrawDate: text("based_on_draw_date").notNull(),
    predictedNumbers: text("predicted_numbers").notNull(),
    source: text("source").notNull().default("policy"),
    policyMode: text("policy_mode").notNull().default("shadow"),
    algorithmVersion: text("algorithm_version").notNull().default("bandit_v1"),
    contextJson: text("context_json").notNull(),
    actionJson: text("action_json").notNull(),
    usedForRunId: integer("used_for_run_id").references(
      () => nnTrainingRuns.id,
      {
        onDelete: "set null",
      },
    ),
    wasOverridden: integer("was_overridden", { mode: "boolean" })
      .notNull()
      .default(false),
    overrideConfigJson: text("override_config_json"),
    resolved: integer("resolved", { mode: "boolean" }).notNull().default(false),
    resolvedAt: integer("resolved_at", { mode: "timestamp_ms" }),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    runIdx: index("nn_prediction_decisions_run_idx").on(table.runId),
    targetNumberIdx: index("nn_prediction_decisions_target_number_idx").on(
      table.targetDrawNumber,
    ),
    unresolvedIdx: index("nn_prediction_decisions_unresolved_idx").on(
      table.resolved,
      table.targetDrawNumber,
    ),
    createdAtIdx: index("nn_prediction_decisions_created_at_idx").on(
      table.createdAt,
    ),
  }),
);

export const nnFeedbackOutcomes = sqliteTable(
  "nn_feedback_outcomes",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    decisionId: integer("decision_id")
      .notNull()
      .references(() => nnPredictionDecisions.id, { onDelete: "cascade" }),
    drawId: integer("draw_id").references(() => lotteryDraws.id, {
      onDelete: "set null",
    }),
    actualNumbers: text("actual_numbers").notNull(),
    matchCount: integer("match_count").notNull().default(0),
    topKHit: integer("top_k_hit", { mode: "boolean" }).notNull().default(false),
    rewardValue: text("reward_value").notNull(),
    rewardVersion: text("reward_version").notNull().default("v1"),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    decisionIdx: index("nn_feedback_outcomes_decision_idx").on(
      table.decisionId,
    ),
    drawIdx: index("nn_feedback_outcomes_draw_idx").on(table.drawId),
    createdAtIdx: index("nn_feedback_outcomes_created_at_idx").on(
      table.createdAt,
    ),
  }),
);

export const nnPolicyUpdates = sqliteTable(
  "nn_policy_updates",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    decisionId: integer("decision_id").references(
      () => nnPredictionDecisions.id,
      {
        onDelete: "set null",
      },
    ),
    feedbackOutcomeId: integer("feedback_outcome_id").references(
      () => nnFeedbackOutcomes.id,
      {
        onDelete: "set null",
      },
    ),
    algorithmVersion: text("algorithm_version").notNull().default("bandit_v1"),
    policyMode: text("policy_mode").notNull().default("shadow"),
    inputFeaturesJson: text("input_features_json").notNull(),
    chosenActionJson: text("chosen_action_json").notNull(),
    rewardValue: text("reward_value").notNull(),
    baselineReward: text("baseline_reward"),
    advantage: text("advantage"),
    explorationRate: text("exploration_rate").notNull().default("0.1"),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .$defaultFn(() => new Date()),
  },
  (table) => ({
    decisionIdx: index("nn_policy_updates_decision_idx").on(table.decisionId),
    outcomeIdx: index("nn_policy_updates_outcome_idx").on(
      table.feedbackOutcomeId,
    ),
    createdAtIdx: index("nn_policy_updates_created_at_idx").on(table.createdAt),
  }),
);
