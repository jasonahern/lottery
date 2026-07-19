ALTER TABLE `lottery_draws` ADD `wins` integer;
--> statement-breakpoint
ALTER TABLE `lottery_draws` ADD `ball_set` integer;
--> statement-breakpoint
ALTER TABLE `nn_test_results` ADD `frequency_predicted_numbers` text;
--> statement-breakpoint
ALTER TABLE `nn_test_results` ADD `frequency_match_count` integer;
--> statement-breakpoint
ALTER TABLE `nn_test_results` ADD `random_predicted_numbers` text;
--> statement-breakpoint
ALTER TABLE `nn_test_results` ADD `random_match_count` integer;
