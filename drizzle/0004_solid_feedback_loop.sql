CREATE TABLE `nn_prediction_decisions` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`run_id` integer,
	`target_draw_id` integer,
	`target_draw_number` integer NOT NULL,
	`based_on_draw_number` integer NOT NULL,
	`based_on_draw_date` text NOT NULL,
	`predicted_numbers` text NOT NULL,
	`source` text DEFAULT 'policy' NOT NULL,
	`policy_mode` text DEFAULT 'shadow' NOT NULL,
	`algorithm_version` text DEFAULT 'bandit_v1' NOT NULL,
	`context_json` text NOT NULL,
	`action_json` text NOT NULL,
	`used_for_run_id` integer,
	`was_overridden` integer DEFAULT false NOT NULL,
	`override_config_json` text,
	`resolved` integer DEFAULT false NOT NULL,
	`resolved_at` integer,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`run_id`) REFERENCES `nn_training_runs`(`id`) ON UPDATE no action ON DELETE set null,
	FOREIGN KEY (`target_draw_id`) REFERENCES `lottery_draws`(`id`) ON UPDATE no action ON DELETE set null,
	FOREIGN KEY (`used_for_run_id`) REFERENCES `nn_training_runs`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `nn_prediction_decisions_run_idx` ON `nn_prediction_decisions` (`run_id`);
--> statement-breakpoint
CREATE INDEX `nn_prediction_decisions_target_number_idx` ON `nn_prediction_decisions` (`target_draw_number`);
--> statement-breakpoint
CREATE INDEX `nn_prediction_decisions_unresolved_idx` ON `nn_prediction_decisions` (`resolved`,`target_draw_number`);
--> statement-breakpoint
CREATE INDEX `nn_prediction_decisions_created_at_idx` ON `nn_prediction_decisions` (`created_at`);
--> statement-breakpoint

CREATE TABLE `nn_feedback_outcomes` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`decision_id` integer NOT NULL,
	`draw_id` integer,
	`actual_numbers` text NOT NULL,
	`match_count` integer DEFAULT 0 NOT NULL,
	`top_k_hit` integer DEFAULT false NOT NULL,
	`reward_value` text NOT NULL,
	`reward_version` text DEFAULT 'v1' NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`decision_id`) REFERENCES `nn_prediction_decisions`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`draw_id`) REFERENCES `lottery_draws`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `nn_feedback_outcomes_decision_idx` ON `nn_feedback_outcomes` (`decision_id`);
--> statement-breakpoint
CREATE INDEX `nn_feedback_outcomes_draw_idx` ON `nn_feedback_outcomes` (`draw_id`);
--> statement-breakpoint
CREATE INDEX `nn_feedback_outcomes_created_at_idx` ON `nn_feedback_outcomes` (`created_at`);
--> statement-breakpoint

CREATE TABLE `nn_policy_updates` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`decision_id` integer,
	`feedback_outcome_id` integer,
	`algorithm_version` text DEFAULT 'bandit_v1' NOT NULL,
	`policy_mode` text DEFAULT 'shadow' NOT NULL,
	`input_features_json` text NOT NULL,
	`chosen_action_json` text NOT NULL,
	`reward_value` text NOT NULL,
	`baseline_reward` text,
	`advantage` text,
	`exploration_rate` text DEFAULT '0.1' NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`decision_id`) REFERENCES `nn_prediction_decisions`(`id`) ON UPDATE no action ON DELETE set null,
	FOREIGN KEY (`feedback_outcome_id`) REFERENCES `nn_feedback_outcomes`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `nn_policy_updates_decision_idx` ON `nn_policy_updates` (`decision_id`);
--> statement-breakpoint
CREATE INDEX `nn_policy_updates_outcome_idx` ON `nn_policy_updates` (`feedback_outcome_id`);
--> statement-breakpoint
CREATE INDEX `nn_policy_updates_created_at_idx` ON `nn_policy_updates` (`created_at`);
