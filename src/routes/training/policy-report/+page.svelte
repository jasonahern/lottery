<script lang="ts">
  let { data } = $props();

  const feedbackStatus = $derived(data.feedbackStatus);
  const history = $derived(data.history);
  const trend = $derived(data.trend);
  const successHistory = $derived(data.successHistory);
  const successTrend = $derived(data.successTrend);

  const chart = $derived.by(() => {
    if (!trend || trend.length === 0) {
      return null;
    }

    const width = 900;
    const height = 260;
    const padX = 40;
    const padY = 20;

    const values = trend.flatMap(
      (point: { reward: number; rollingAverage: number }) => [
        point.reward,
        point.rollingAverage,
      ],
    );
    const min = Math.min(...values, -1);
    const max = Math.max(...values, 1);
    const range = Math.max(0.001, max - min);

    const xFor = (idx: number) =>
      padX + (idx / Math.max(1, trend.length - 1)) * (width - padX * 2);
    const yFor = (v: number) =>
      padY + ((max - v) / range) * (height - padY * 2);

    const rewardPath = trend
      .map(
        (point: { reward: number }, idx: number) =>
          `${idx === 0 ? "M" : "L"}${xFor(idx)},${yFor(point.reward)}`,
      )
      .join(" ");

    const avgPath = trend
      .map(
        (point: { rollingAverage: number }, idx: number) =>
          `${idx === 0 ? "M" : "L"}${xFor(idx)},${yFor(point.rollingAverage)}`,
      )
      .join(" ");

    const zeroY = yFor(0);

    return {
      width,
      height,
      rewardPath,
      avgPath,
      zeroY,
      min,
      max,
    };
  });

  const successChart = $derived.by(() => {
    if (!successTrend || successTrend.length === 0) {
      return null;
    }

    const width = 900;
    const height = 260;
    const padX = 40;
    const padY = 20;

    const values = successTrend.flatMap(
      (point: { successPercent: number; rollingAverage: number }) => [
        point.successPercent,
        point.rollingAverage,
      ],
    );
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 100);
    const range = Math.max(0.001, max - min);

    const xFor = (idx: number) =>
      padX + (idx / Math.max(1, successTrend.length - 1)) * (width - padX * 2);
    const yFor = (v: number) =>
      padY + ((max - v) / range) * (height - padY * 2);

    const valuePath = successTrend
      .map(
        (point: { successPercent: number }, idx: number) =>
          `${idx === 0 ? "M" : "L"}${xFor(idx)},${yFor(point.successPercent)}`,
      )
      .join(" ");

    const avgPath = successTrend
      .map(
        (point: { rollingAverage: number }, idx: number) =>
          `${idx === 0 ? "M" : "L"}${xFor(idx)},${yFor(point.rollingAverage)}`,
      )
      .join(" ");

    return {
      width,
      height,
      valuePath,
      avgPath,
      min,
      max,
    };
  });

  const latestSuccessPercent = $derived.by(() => {
    if (!successTrend || successTrend.length === 0) {
      return null;
    }

    return successTrend[successTrend.length - 1]?.successPercent ?? null;
  });

  const averageSuccessPercent30 = $derived.by(() => {
    if (!successTrend || successTrend.length === 0) {
      return null;
    }

    const tail = successTrend.slice(-30);
    const average =
      tail.reduce((acc: number, point: { successPercent: number }) => {
        return acc + point.successPercent;
      }, 0) / tail.length;

    return Number(average.toFixed(2));
  });
</script>

<svelte:head>
  <title>Policy Report</title>
</svelte:head>

<div
  class="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8"
>
  <header
    class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between"
  >
    <div>
      <h1 class="text-3xl font-bold tracking-tight text-zinc-900 sm:text-4xl">
        Policy Update Report
      </h1>
      <p class="mt-1 text-sm text-zinc-600">
        Review policy update history and reward trends used by the
        reinforcement-style feedback loop.
      </p>
    </div>
    <div class="flex gap-2">
      <a
        href="/training"
        class="inline-flex cursor-pointer items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition-[transform,colors] duration-100 hover:bg-zinc-50 active:translate-y-px active:scale-[0.98] active:bg-zinc-100"
      >
        Back to Training
      </a>
      <a
        href="/training/policy-report/data?limit=120"
        class="inline-flex cursor-pointer items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition-[transform,colors] duration-100 hover:bg-zinc-50 active:translate-y-px active:scale-[0.98] active:bg-zinc-100"
      >
        JSON Endpoint
      </a>
    </div>
  </header>

  <section class="grid gap-4 md:grid-cols-5">
    <div class="rounded-lg border border-zinc-200 bg-white p-4">
      <p class="text-xs uppercase tracking-wide text-zinc-500">Policy Mode</p>
      <p class="mt-2 text-2xl font-semibold text-zinc-900">
        {feedbackStatus.mode}
      </p>
    </div>
    <div class="rounded-lg border border-zinc-200 bg-white p-4">
      <p class="text-xs uppercase tracking-wide text-zinc-500">
        Recent Avg Reward
      </p>
      <p class="mt-2 text-2xl font-semibold text-zinc-900">
        {feedbackStatus.recentAverageReward}
      </p>
    </div>
    <div class="rounded-lg border border-zinc-200 bg-white p-4">
      <p class="text-xs uppercase tracking-wide text-zinc-500">
        Resolved Outcomes
      </p>
      <p class="mt-2 text-2xl font-semibold text-zinc-900">
        {feedbackStatus.resolvedOutcomes}
      </p>
    </div>
    <div class="rounded-lg border border-zinc-200 bg-white p-4">
      <p class="text-xs uppercase tracking-wide text-zinc-500">
        Latest Success %
      </p>
      <p class="mt-2 text-2xl font-semibold text-zinc-900">
        {latestSuccessPercent !== null ? `${latestSuccessPercent}%` : "n/a"}
      </p>
    </div>
    <div class="rounded-lg border border-zinc-200 bg-white p-4">
      <p class="text-xs uppercase tracking-wide text-zinc-500">
        Avg Success % (30)
      </p>
      <p class="mt-2 text-2xl font-semibold text-zinc-900">
        {averageSuccessPercent30 !== null
          ? `${averageSuccessPercent30}%`
          : "n/a"}
      </p>
    </div>
  </section>

  <section class="rounded-lg border border-zinc-200 bg-white p-4">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="text-lg font-semibold text-zinc-900">Reward Trend Chart</h2>
      <span class="text-xs text-zinc-500">Latest {trend.length} points</span>
    </div>

    {#if chart}
      <svg
        viewBox={`0 0 ${chart.width} ${chart.height}`}
        class="w-full overflow-visible rounded-md border border-zinc-100 bg-zinc-50"
      >
        <line
          x1="40"
          x2={chart.width - 40}
          y1={chart.zeroY}
          y2={chart.zeroY}
          stroke="#d4d4d8"
          stroke-width="1"
          stroke-dasharray="4 4"
        />
        <path
          d={chart.rewardPath}
          fill="none"
          stroke="#0284c7"
          stroke-width="2"
        />
        <path
          d={chart.avgPath}
          fill="none"
          stroke="#0f766e"
          stroke-width="2.5"
        />
        <text x="8" y="20" font-size="10" fill="#71717a"
          >max {chart.max.toFixed(2)}</text
        >
        <text x="8" y={chart.height - 8} font-size="10" fill="#71717a"
          >min {chart.min.toFixed(2)}</text
        >
      </svg>
      <div class="mt-2 flex gap-4 text-xs text-zinc-600">
        <span class="inline-flex items-center gap-1"
          ><span class="inline-block h-2 w-4 rounded bg-sky-600"
          ></span>reward</span
        >
        <span class="inline-flex items-center gap-1"
          ><span class="inline-block h-2 w-4 rounded bg-teal-700"></span>rolling
          avg (10)</span
        >
      </div>
    {:else}
      <p class="text-sm text-zinc-600">Not enough reward data yet.</p>
    {/if}
  </section>

  <section class="rounded-lg border border-zinc-200 bg-white p-4">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="text-lg font-semibold text-zinc-900">
        Prediction Success % Trend
      </h2>
      <span class="text-xs text-zinc-500"
        >Latest {successTrend.length} points</span
      >
    </div>

    {#if successChart}
      <svg
        viewBox={`0 0 ${successChart.width} ${successChart.height}`}
        class="w-full overflow-visible rounded-md border border-zinc-100 bg-zinc-50"
      >
        <path
          d={successChart.valuePath}
          fill="none"
          stroke="#f97316"
          stroke-width="2"
        />
        <path
          d={successChart.avgPath}
          fill="none"
          stroke="#b45309"
          stroke-width="2.5"
        />
        <text x="8" y="20" font-size="10" fill="#71717a"
          >max {successChart.max.toFixed(2)}%</text
        >
        <text x="8" y={successChart.height - 8} font-size="10" fill="#71717a"
          >min {successChart.min.toFixed(2)}%</text
        >
      </svg>
      <div class="mt-2 flex gap-4 text-xs text-zinc-600">
        <span class="inline-flex items-center gap-1"
          ><span class="inline-block h-2 w-4 rounded bg-orange-500"
          ></span>success %</span
        >
        <span class="inline-flex items-center gap-1"
          ><span class="inline-block h-2 w-4 rounded bg-amber-700"
          ></span>rolling avg % (10)</span
        >
      </div>
    {:else}
      <p class="text-sm text-zinc-600">Not enough success data yet.</p>
    {/if}
  </section>

  <section class="rounded-lg border border-zinc-200 bg-white p-4">
    <h2 class="text-lg font-semibold text-zinc-900">
      Per-Draw Success History
    </h2>
    {#if successHistory.length === 0}
      <p class="mt-2 text-sm text-zinc-600">
        No resolved prediction outcomes yet.
      </p>
    {:else}
      <div class="mt-3 overflow-x-auto">
        <table class="w-full min-w-215 border-collapse text-sm">
          <thead>
            <tr class="border-b border-zinc-200 text-left text-zinc-600">
              <th class="px-3 py-2 font-semibold">Outcome ID</th>
              <th class="px-3 py-2 font-semibold">Draw ID</th>
              <th class="px-3 py-2 font-semibold">Decision ID</th>
              <th class="px-3 py-2 font-semibold">Matches</th>
              <th class="px-3 py-2 font-semibold">Actual Count</th>
              <th class="px-3 py-2 font-semibold">Success %</th>
              <th class="px-3 py-2 font-semibold">Created</th>
            </tr>
          </thead>
          <tbody>
            {#each successHistory as item}
              <tr class="border-b border-zinc-100 text-zinc-800">
                <td class="px-3 py-2">{item.id}</td>
                <td class="px-3 py-2">{item.drawId ?? "n/a"}</td>
                <td class="px-3 py-2">{item.decisionId}</td>
                <td class="px-3 py-2">{item.matchCount}</td>
                <td class="px-3 py-2">{item.actualCount}</td>
                <td class="px-3 py-2">{item.successPercent}%</td>
                <td class="px-3 py-2">{item.createdAt}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  <section class="rounded-lg border border-zinc-200 bg-white p-4">
    <h2 class="text-lg font-semibold text-zinc-900">Policy Update History</h2>
    {#if history.length === 0}
      <p class="mt-2 text-sm text-zinc-600">No policy updates recorded yet.</p>
    {:else}
      <div class="mt-3 overflow-x-auto">
        <table class="w-full min-w-230 border-collapse text-sm">
          <thead>
            <tr class="border-b border-zinc-200 text-left text-zinc-600">
              <th class="px-3 py-2 font-semibold">Update ID</th>
              <th class="px-3 py-2 font-semibold">Mode</th>
              <th class="px-3 py-2 font-semibold">Reward</th>
              <th class="px-3 py-2 font-semibold">Baseline</th>
              <th class="px-3 py-2 font-semibold">Advantage</th>
              <th class="px-3 py-2 font-semibold">Exploration</th>
              <th class="px-3 py-2 font-semibold">Decision ID</th>
              <th class="px-3 py-2 font-semibold">Created</th>
            </tr>
          </thead>
          <tbody>
            {#each history as item}
              <tr class="border-b border-zinc-100 text-zinc-800">
                <td class="px-3 py-2">{item.id}</td>
                <td class="px-3 py-2">{item.policyMode}</td>
                <td class="px-3 py-2">{item.rewardValue}</td>
                <td class="px-3 py-2">{item.baselineReward ?? "n/a"}</td>
                <td class="px-3 py-2">{item.advantage ?? "n/a"}</td>
                <td class="px-3 py-2">{item.explorationRate}</td>
                <td class="px-3 py-2">{item.decisionId ?? "n/a"}</td>
                <td class="px-3 py-2">{item.createdAt}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
</div>
