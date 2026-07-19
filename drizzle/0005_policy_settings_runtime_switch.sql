CREATE TABLE `nn_policy_settings` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`key` text NOT NULL,
	`value` text NOT NULL,
	`updated_at` integer NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `nn_policy_settings_key_idx` ON `nn_policy_settings` (`key`);
--> statement-breakpoint
CREATE INDEX `nn_policy_settings_updated_at_idx` ON `nn_policy_settings` (`updated_at`);
