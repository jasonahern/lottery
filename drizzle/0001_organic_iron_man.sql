CREATE TABLE `nn_epoch_metrics` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`epoch` integer NOT NULL,
	`train_loss` text,
	`val_loss` text,
	`elapsed_ms` integer,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `nn_test_results` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer NOT NULL,
	`draw_id` integer,
	`draw_date` text,
	`predicted_numbers` text NOT NULL,
	`actual_numbers` text NOT NULL,
	`match_count` integer DEFAULT 0 NOT NULL,
	`top_k_hit` integer DEFAULT false NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `nn_training_runs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`status` text DEFAULT 'queued' NOT NULL,
	`started_at` integer,
	`ended_at` integer,
	`holdout_weeks` integer NOT NULL,
	`window_size` integer NOT NULL,
	`total_epochs` integer NOT NULL,
	`current_epoch` integer DEFAULT 0 NOT NULL,
	`train_samples` integer,
	`test_samples` integer,
	`samples_total` integer,
	`samples_processed` integer DEFAULT 0 NOT NULL,
	`model_family` text DEFAULT 'mlp_v1' NOT NULL,
	`hidden_layers_json` text NOT NULL,
	`hyperparams_json` text NOT NULL,
	`param_count` integer,
	`final_train_loss` text,
	`final_val_loss` text,
	`holdout_score` text,
	`model_artifact_path` text,
	`model_sha256` text,
	`error_message` text,
	`created_at` integer NOT NULL
);
