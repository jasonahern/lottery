ALTER TABLE `lottery_draws` ADD `draw_round` integer DEFAULT 1 NOT NULL;--> statement-breakpoint
UPDATE `lottery_draws` AS current
SET `draw_round` = (
  SELECT COUNT(*) FROM `lottery_draws` AS preceding
  WHERE preceding.`game_name` IS current.`game_name`
    AND preceding.`draw_number` = current.`draw_number`
    AND preceding.`id` <= current.`id`
);--> statement-breakpoint
CREATE UNIQUE INDEX `lottery_draws_game_draw_round_unique_idx` ON `lottery_draws` (`game_name`,`draw_number`,`draw_round`);
