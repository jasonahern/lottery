CREATE TABLE `lottery_draw_balls` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`draw_id` integer NOT NULL,
	`position` integer NOT NULL,
	`value` integer NOT NULL,
	`is_bonus` integer DEFAULT false NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`draw_id`) REFERENCES `lottery_draws`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `lottery_draw_balls_draw_position_idx` ON `lottery_draw_balls` (`draw_id`,`position`);--> statement-breakpoint
CREATE INDEX `lottery_draw_balls_value_idx` ON `lottery_draw_balls` (`value`);--> statement-breakpoint
CREATE TABLE `lottery_draws` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`legacy_entry_id` integer,
	`draw_number` integer NOT NULL,
	`draw_date` integer NOT NULL,
	`day_name` text,
	`game_name` text,
	`machine` text,
	`draw_sequence` integer,
	`jackpot_amount` integer,
	`source_row` text NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`legacy_entry_id`) REFERENCES `lottery_entries`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE INDEX `lottery_draws_draw_number_idx` ON `lottery_draws` (`draw_number`);--> statement-breakpoint
CREATE INDEX `lottery_draws_draw_date_idx` ON `lottery_draws` (`draw_date`);--> statement-breakpoint
CREATE INDEX `lottery_draws_game_draw_idx` ON `lottery_draws` (`game_name`,`draw_number`);