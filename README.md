# sv

Everything you need to build a Svelte project, powered by [`sv`](https://github.com/sveltejs/cli).

## Creating a project

If you're seeing this, you've probably already done this step. Congrats!

```sh
# create a new project
npx sv create my-app
```

To recreate this project with the same configuration:

```sh
# recreate this project
npx sv@0.16.2 create --template minimal --types ts --install npm lottery
```

## Developing

Once you've created a project and installed dependencies with `npm install` (or `pnpm install` or `yarn`), start a development server:

```sh
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

## Database (SQLite + Drizzle ORM)

This project is configured to use Drizzle ORM with a local SQLite database file named `lottery.db`.

```sh
# generate SQL migration files from schema changes
npm run db:generate

# apply migrations to lottery.db
npm run db:migrate

# open Drizzle Studio
npm run db:studio
```

Use the configured database client from server-side code:

```ts
import { db } from "$lib/server/db";
import { lotteryEntries } from "$lib/server/db/schema";

const entries = db.select().from(lotteryEntries).all();
```

Lottery data is now normalized into:

- `lottery_draws`: one row per draw with typed date/metadata
- `lottery_draw_balls`: one row per ball (with `position` and `is_bonus`)

`lottery_entries` is retained as legacy raw-source storage for compatibility and audit.
The app backfills normalized rows automatically from legacy data when needed.

## Building

To create a production version of your app:

```sh
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.

## Neural Network Training

The project now includes a TensorFlow.js training workflow for lottery backtesting.

1. Open `/training` in the app.
2. Create a run with a preset or custom hidden layers.
3. The run starts immediately and updates progress automatically.

Key behavior:

- Time-based split excludes the last `holdoutWeeks` from training.
- Training/inference reads normalized draw data (with automatic legacy backfill fallback).
- Run metadata, per-epoch metrics, and holdout predictions are persisted in SQLite.
- Model artifacts are saved to `artifacts/nn-runs/` and indexed in `nn_training_runs`.
- A recommendation panel suggests when to scale neurons or regularize.

Run API:

- `GET /training/runs?limit=20` returns recent runs and recommendation.
- `GET /training/runs?runId=123` returns run detail with epoch and holdout data.

## Testing

Run all unit tests:

```sh
npm run test
```
