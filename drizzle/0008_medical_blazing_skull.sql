-- Remove exact duplicate normalized draws, retaining the oldest canonical row.
DELETE FROM `lottery_draws` AS duplicate
WHERE EXISTS (
  SELECT 1 FROM `lottery_draws` AS canonical
  WHERE canonical.`game_name` IS duplicate.`game_name`
    AND canonical.`draw_number` = duplicate.`draw_number`
    AND canonical.`source_row` = duplicate.`source_row`
    AND canonical.`id` < duplicate.`id`
);--> statement-breakpoint
CREATE UNIQUE INDEX `lottery_draws_source_row_unique_idx` ON `lottery_draws` (`source_row`);--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `neural_holdout_score` text;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `ensemble_holdout_score` text;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `frequency_holdout_score` text;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `heuristic_holdout_score` text;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `random_holdout_score` text;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `best_epoch` integer;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `input_encoding` text DEFAULT 'sorted_scalar_v1' NOT NULL;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `loss_version` text DEFAULT 'binary_crossentropy_v1' NOT NULL;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `training_seed` integer;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `is_valid` integer DEFAULT true NOT NULL;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `invalidated_at` integer;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `invalidation_reason` text;--> statement-breakpoint
-- Historical artifacts were trained before duplicate/leakage protection and v2 encoding.
UPDATE `nn_training_runs`
SET `is_valid` = false,
    `invalidated_at` = unixepoch('now') * 1000,
    `invalidation_reason` = 'Invalidated by v2 data-integrity migration; retraining required.';
