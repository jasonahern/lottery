import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";

import * as schema from "./schema";

const sqlite = new Database("lottery.db");

export function ensureDatabaseSchema(): void {
  const trainingRunsTable = sqlite
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'nn_training_runs'")
    .get();
  if (!trainingRunsTable) return;

  const columns = sqlite
    .prepare("PRAGMA table_info(nn_training_runs)")
    .all() as Array<{ name: string }>;
  if (!columns.some((column) => column.name === "is_favorite")) {
    sqlite.exec(
      "ALTER TABLE nn_training_runs ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0",
    );
  }
  if (!columns.some((column) => column.name === "current_phase")) {
    sqlite.exec("ALTER TABLE nn_training_runs ADD COLUMN current_phase TEXT NOT NULL DEFAULT 'final_training'");
  }
  if (!columns.some((column) => column.name === "current_fold")) {
    sqlite.exec("ALTER TABLE nn_training_runs ADD COLUMN current_fold INTEGER NOT NULL DEFAULT 0");
  }
  if (!columns.some((column) => column.name === "total_folds")) {
    sqlite.exec("ALTER TABLE nn_training_runs ADD COLUMN total_folds INTEGER NOT NULL DEFAULT 0");
  }
  sqlite.exec(`CREATE TABLE IF NOT EXISTS nn_backtest_folds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES nn_training_runs(id) ON DELETE CASCADE,
    fold INTEGER NOT NULL,
    train_start_date TEXT NOT NULL, train_end_date TEXT NOT NULL,
    calibration_start_date TEXT NOT NULL, calibration_end_date TEXT NOT NULL,
    holdout_start_date TEXT NOT NULL, holdout_end_date TEXT NOT NULL,
    train_samples INTEGER NOT NULL, calibration_samples INTEGER NOT NULL, holdout_samples INTEGER NOT NULL,
    best_epoch INTEGER, final_train_loss TEXT, final_val_loss TEXT,
    neural_score TEXT NOT NULL, ensemble_score TEXT NOT NULL, gated_score TEXT NOT NULL, random_score TEXT NOT NULL,
    selected_method TEXT NOT NULL, diagnostics_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
  )`);
  sqlite.exec("CREATE UNIQUE INDEX IF NOT EXISTS nn_backtest_folds_run_fold_uq ON nn_backtest_folds(run_id, fold)");
}

ensureDatabaseSchema();

export const db = drizzle(sqlite, { schema });

export type Db = typeof db;
