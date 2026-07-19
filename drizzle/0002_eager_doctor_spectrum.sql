PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_nn_epoch_metrics` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`epoch` integer NOT NULL,
	`train_loss` text,
	`val_loss` text,
	`elapsed_ms` integer,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `nn_training_runs`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
INSERT INTO `__new_nn_epoch_metrics`("id", "run_id", "epoch", "train_loss", "val_loss", "elapsed_ms", "created_at") SELECT "id", "run_id", "epoch", "train_loss", "val_loss", "elapsed_ms", "created_at" FROM `nn_epoch_metrics`;--> statement-breakpoint
DROP TABLE `nn_epoch_metrics`;--> statement-breakpoint
ALTER TABLE `__new_nn_epoch_metrics` RENAME TO `nn_epoch_metrics`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
CREATE INDEX `nn_epoch_metrics_run_epoch_idx` ON `nn_epoch_metrics` (`run_id`,`epoch`);--> statement-breakpoint
CREATE TABLE `__new_nn_test_results` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`draw_id` integer,
	`draw_date` text,
	`predicted_numbers` text NOT NULL,
	`actual_numbers` text NOT NULL,
	`match_count` integer DEFAULT 0 NOT NULL,
	`top_k_hit` integer DEFAULT false NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `nn_training_runs`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
INSERT INTO `__new_nn_test_results`("id", "run_id", "draw_id", "draw_date", "predicted_numbers", "actual_numbers", "match_count", "top_k_hit", "created_at") SELECT "id", "run_id", "draw_id", "draw_date", "predicted_numbers", "actual_numbers", "match_count", "top_k_hit", "created_at" FROM `nn_test_results`;--> statement-breakpoint
DROP TABLE `nn_test_results`;--> statement-breakpoint
ALTER TABLE `__new_nn_test_results` RENAME TO `nn_test_results`;--> statement-breakpoint
CREATE INDEX `nn_test_results_run_draw_idx` ON `nn_test_results` (`run_id`,`draw_date`);--> statement-breakpoint
CREATE INDEX `nn_training_runs_status_idx` ON `nn_training_runs` (`status`);--> statement-breakpoint
CREATE INDEX `nn_training_runs_created_at_idx` ON `nn_training_runs` (`created_at`);--> statement-breakpoint
CREATE INDEX `nn_training_runs_holdout_score_idx` ON `nn_training_runs` (`holdout_score`);