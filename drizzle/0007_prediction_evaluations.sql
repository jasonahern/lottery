CREATE TABLE `nn_prediction_evaluations` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`draw_id` integer,
	`draw_date` text,
	`method` text NOT NULL,
	`predicted_numbers` text NOT NULL,
	`actual_numbers` text NOT NULL,
	`match_count` integer DEFAULT 0 NOT NULL,
	`top_k_hit` integer DEFAULT false NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `nn_training_runs`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `nn_prediction_evaluations_run_method_idx` ON `nn_prediction_evaluations` (`run_id`, `method`);
--> statement-breakpoint
CREATE INDEX `nn_prediction_evaluations_run_draw_method_idx` ON `nn_prediction_evaluations` (`run_id`, `draw_id`, `method`);
