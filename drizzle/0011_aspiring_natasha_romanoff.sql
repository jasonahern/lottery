CREATE TABLE `nn_backtest_folds` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`fold` integer NOT NULL,
	`train_start_date` text NOT NULL,
	`train_end_date` text NOT NULL,
	`calibration_start_date` text NOT NULL,
	`calibration_end_date` text NOT NULL,
	`holdout_start_date` text NOT NULL,
	`holdout_end_date` text NOT NULL,
	`train_samples` integer NOT NULL,
	`calibration_samples` integer NOT NULL,
	`holdout_samples` integer NOT NULL,
	`best_epoch` integer,
	`final_train_loss` text,
	`final_val_loss` text,
	`neural_score` text NOT NULL,
	`ensemble_score` text NOT NULL,
	`gated_score` text NOT NULL,
	`random_score` text NOT NULL,
	`selected_method` text NOT NULL,
	`diagnostics_json` text NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `nn_training_runs`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `nn_backtest_folds_run_fold_uq` ON `nn_backtest_folds` (`run_id`,`fold`);--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `current_phase` text DEFAULT 'final_training' NOT NULL;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `current_fold` integer DEFAULT 0 NOT NULL;--> statement-breakpoint
ALTER TABLE `nn_training_runs` ADD `total_folds` integer DEFAULT 0 NOT NULL;