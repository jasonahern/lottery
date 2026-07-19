<script lang="ts">
  import { enhance } from "$app/forms";
  import { invalidateAll } from "$app/navigation";
  import { onDestroy } from "svelte";
  import type { SubmitFunction } from "@sveltejs/kit";

  import { Badge } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import { Card } from "$lib/components/ui/card";

  let { data, form } = $props();

  const defaults = $derived(data.defaults);
  const latestRuns = $derived(data.latestRuns);
  const activeRun = $derived(data.activeRun);
  const holdoutRunDetail = $derived(data.holdoutRunDetail);
  const holdoutResultsRunId = $derived(data.holdoutResultsRunId);
  const nextDrawPrediction = $derived(data.nextDrawPrediction);
  const datasetStats = $derived(data.datasetStats);
  const latestRunFormConfig = $derived(data.latestRunFormConfig);
  const feedbackStatus = $derived(data.feedbackStatus);
  const predictionRunCandidates = $derived(data.predictionRunCandidates);
  const runComparisonSummaries = $derived(data.runComparisonSummaries);
  const pinnedPredictionRunId = $derived(data.pinnedPredictionRunId);
  const predictionRunMessage = $derived(data.predictionRunMessage);
  const nextPredictionMethods = $derived.by(() =>
    nextDrawPrediction
      ? [
          { label: "Ensemble", numbers: nextDrawPrediction.predictedNumbers, weight: null },
          { label: "Neural", numbers: nextDrawPrediction.neuralPredictedNumbers, weight: nextDrawPrediction.ensembleWeights.neural },
          { label: "Frequency", numbers: nextDrawPrediction.frequencyPredictedNumbers, weight: nextDrawPrediction.ensembleWeights.frequency },
          { label: "Heuristic", numbers: nextDrawPrediction.heuristicPredictedNumbers, weight: nextDrawPrediction.ensembleWeights.heuristic },
          { label: "Random control", numbers: nextDrawPrediction.randomPredictedNumbers, weight: nextDrawPrediction.ensembleWeights.random },
        ]
      : [],
  );
  const nextPredictionAgreement = $derived.by(() => {
    const agreement = new Map<number, string[]>();
    for (const method of nextPredictionMethods) {
      for (const number of method.numbers) {
        agreement.set(number, [...(agreement.get(number) ?? []), method.label]);
      }
    }
    return agreement;
  });
  const sharedNextPredictionNumbers = $derived(
    [...nextPredictionAgreement.entries()]
      .filter(([, methods]) => methods.length > 1)
      .sort((a, b) => b[1].length - a[1].length || a[0] - b[0]),
  );

  const activeProgress = $derived(form?.progress ?? activeRun);
  const createdRun = $derived(form?.run);
  const holdoutComparison = $derived.by(() => {
    const results = holdoutRunDetail?.testResults ?? [];
    if (results.length === 0) {
      return null;
    }

    const average = (values: number[]) =>
      values.length === 0
        ? null
        : values.reduce((acc, value) => acc + value, 0) / values.length;

    return {
      model: average(results.map((result) => result.matchCount)),
      frequency: average(
        results
          .map((result) => result.frequencyMatchCount)
          .filter((value): value is number => typeof value === "number"),
      ),
      random: average(
        results
          .map((result) => result.randomMatchCount)
          .filter((value): value is number => typeof value === "number"),
      ),
      heuristic: average(
        results
          .map((result) => result.heuristicMatchCount)
          .filter((value): value is number => typeof value === "number"),
      ),
      ensemble: average(
        results
          .map((result) => result.ensembleMatchCount)
          .filter((value): value is number => typeof value === "number"),
      ),
    };
  });

  let selectedPreset = $state("medium");
  let customHiddenLayersText = $state("");
  let holdoutWeeksInput = $state("");
  let windowSizeInput = $state("10");
  let epochsInput = $state("");
  let batchSizeInput = $state("");
  let learningRateInput = $state("");
  let dropoutRateInput = $state("");
  let positiveClassWeightInput = $state("");
  let earlyStoppingPatienceInput = $state("");
  let earlyStoppingMinDeltaInput = $state("");
  let trainingSeedInput = $state("");
  let policyModeInput = $state("");
  let predictionRunInput = $state("");
  let appliedFormRunId = $state<number | null>(null);
  let defaultsApplied = $state(false);

  $effect(() => {
    if (!policyModeInput) {
      policyModeInput = feedbackStatus.mode;
    }

    if (!predictionRunInput) {
      predictionRunInput = pinnedPredictionRunId
        ? String(pinnedPredictionRunId)
        : "";
    }
  });

  $effect(() => {
    const config = latestRunFormConfig;
    if (config && appliedFormRunId !== config.runId) {
      const layerSignature = config.hiddenLayers.join(",");
      const presetByLayers: Record<string, string> = {
        "64,32": "small",
        "128,64": "medium",
        "256,128": "large",
        "512,256": "xlarge",
      };
      selectedPreset = presetByLayers[layerSignature] ?? "custom";
      customHiddenLayersText = presetByLayers[layerSignature]
        ? ""
        : layerSignature;
      holdoutWeeksInput = String(config.holdoutWeeks);
      windowSizeInput = String(config.windowSize);
      epochsInput = String(config.epochs);
      batchSizeInput = String(config.batchSize);
      learningRateInput = String(config.learningRate);
      dropoutRateInput = String(config.dropoutRate);
      positiveClassWeightInput = String(config.positiveClassWeight);
      earlyStoppingPatienceInput = String(config.earlyStoppingPatience);
      earlyStoppingMinDeltaInput = String(config.earlyStoppingMinDelta);
      trainingSeedInput = String(config.trainingSeed);
      appliedFormRunId = config.runId;
      defaultsApplied = true;
    } else if (!config && !defaultsApplied) {
      holdoutWeeksInput = String(defaults.holdoutWeeks);
      windowSizeInput = String(defaults.windowSize);
      epochsInput = String(defaults.epochs);
      batchSizeInput = String(defaults.batchSize);
      learningRateInput = String(defaults.learningRate);
      dropoutRateInput = String(defaults.dropoutRate);
      positiveClassWeightInput = String(defaults.positiveClassWeight);
      earlyStoppingPatienceInput = String(defaults.earlyStoppingPatience);
      earlyStoppingMinDeltaInput = String(defaults.earlyStoppingMinDelta);
      trainingSeedInput = String(defaults.trainingSeed);
      defaultsApplied = true;
    }
  });

  const preserveCreateRunValues: SubmitFunction = () => {
    return async ({ update }) => {
      await update({ reset: false });
    };
  };

  $effect(() => {
    predictionRunInput = pinnedPredictionRunId
      ? String(pinnedPredictionRunId)
      : "";
  });

  const selectedHiddenLayers = $derived.by(() => {
    if (customHiddenLayersText.trim().length > 0) {
      return customHiddenLayersText
        .split(",")
        .map((n) => Number(n.trim()))
        .filter((n) => Number.isFinite(n) && n > 0)
        .map((n) => Math.floor(n));
    }

    if (selectedPreset === "small") {
      return [64, 32];
    }

    if (selectedPreset === "large") {
      return [256, 128];
    }

    if (selectedPreset === "xlarge") {
      return [512, 256];
    }

    if (selectedPreset === "custom") {
      return [];
    }

    return [128, 64];
  });

  const previewParamCount = $derived.by(() => {
    if (
      selectedHiddenLayers.length === 0 ||
      datasetStats.distinctNumberCount <= 0
    ) {
      return 0;
    }

    const inputSize = Number(windowSizeInput) * 6;
    if (!Number.isFinite(inputSize) || inputSize <= 0) {
      return 0;
    }

    let params = 0;
    let prev = inputSize;
    for (const width of selectedHiddenLayers) {
      params += prev * width + width;
      prev = width;
    }

    params +=
      prev * datasetStats.distinctNumberCount +
      datasetStats.distinctNumberCount;
    return params;
  });

  const progressPct = $derived(
    activeProgress?.samplesTotal
      ? Math.round(
          (activeProgress.samplesProcessed / activeProgress.samplesTotal) * 100,
        )
      : 0,
  );

  let pollTimer: ReturnType<typeof setInterval> | null = null;

  $effect(() => {
    if (!activeProgress) {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      return;
    }

    const shouldPoll =
      activeProgress.status === "queued" || activeProgress.status === "running";
    if (!shouldPoll) {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      return;
    }

    if (!pollTimer) {
      pollTimer = setInterval(() => {
        void invalidateAll();
      }, 2000);
    }

    return () => {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };
  });

  onDestroy(() => {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  });
</script>

<svelte:head>
  <title>Training Dashboard</title>
</svelte:head>

<div
  class="relative mx-auto flex w-full max-w-[120rem] flex-col gap-6 px-3 py-8 sm:px-5 lg:px-6"
>
  <div
    class="pointer-events-none absolute inset-x-0 top-0 -z-10 h-40 bg-gradient-to-r from-cyan-100 via-sky-100 to-emerald-100 blur-2xl"
  ></div>

  <header
    class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between"
  >
    <div>
      <h1 class="text-3xl font-bold tracking-tight text-zinc-900 sm:text-4xl">
        Neural Network Training
      </h1>
      <p class="mt-1 text-sm text-zinc-600">
        Run end-to-end training, track progress live, and compare holdout
        outcomes.
      </p>
    </div>

    <div class="flex gap-2">
      <a
        href="/"
        class="inline-flex items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50"
      >
        Back to Home
      </a>
      <a
        href="/database"
        class="inline-flex items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50"
      >
        View Database
      </a>
      <a
        href="/training/policy-report"
        class="inline-flex items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50"
      >
        Policy Report
      </a>
    </div>
  </header>

  <div class="grid gap-4 md:grid-cols-3">
    <Card>
      <p class="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Draws In Database
      </p>
      <p class="mt-2 text-3xl font-semibold text-zinc-900">
        {datasetStats.drawCount}
      </p>
    </Card>
    <Card class={datasetStats.conflictingDrawIdentities > 0 ? "border-red-300" : "border-emerald-300"}>
      <p class="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Data Integrity
      </p>
      <p class:text-red-700={datasetStats.conflictingDrawIdentities > 0} class:text-emerald-700={datasetStats.conflictingDrawIdentities === 0} class="mt-2 text-3xl font-semibold">
        {datasetStats.conflictingDrawIdentities === 0 ? "Clean" : `${datasetStats.conflictingDrawIdentities} conflicts`}
      </p>
      <p class="mt-1 text-xs text-zinc-500">Random expectation: {datasetStats.randomExpectedMatches.toFixed(2)} matches</p>
    </Card>
    <Card>
      <p class="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Distinct Numbers
      </p>
      <p class="mt-2 text-3xl font-semibold text-zinc-900">
        {datasetStats.distinctNumberCount}
      </p>
    </Card>
    <Card>
      <p class="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Max Observed Number
      </p>
      <p class="mt-2 text-3xl font-semibold text-zinc-900">
        {datasetStats.maxObservedNumber}
      </p>
    </Card>
  </div>

  <Card class="border-cyan-200 bg-gradient-to-br from-cyan-50 to-white">
    <div
      class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
    >
      <div>
        <p class="text-sm font-semibold text-zinc-900">Quick Start</p>
        <p class="text-sm text-zinc-600">
          Launch with default balanced settings: holdout {defaults.holdoutWeeks}w,
          epochs {defaults.epochs}, layers [{defaults.hiddenLayers.join(", ")}].
        </p>
      </div>

      <form method="POST" action="?/quickStart" use:enhance>
        <Button type="submit" size="lg">Quick Start Run</Button>
      </form>
    </div>
  </Card>

  <Card class="border-emerald-300 bg-gradient-to-br from-emerald-100 to-white">
    <div
      class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
    >
      <div>
        <p
          class="text-xs font-semibold uppercase tracking-wide text-emerald-700"
        >
          Next Draw Prediction
        </p>
        {#if nextDrawPrediction}
          <p class="mt-1 text-xl font-bold text-zinc-900">
            Ticket prediction for Draw #{nextDrawPrediction.nextDrawNumber}
          </p>
          <p class="text-sm text-zinc-700">
            Based on completed run #{nextDrawPrediction.runId}, using latest
            known draw #{nextDrawPrediction.basedOnDrawNumber} ({nextDrawPrediction.basedOnDrawDate}).
            The same six-number ticket applies to both Round 1 and Round 2.
          </p>
          {#if pinnedPredictionRunId}
            <p class="mt-1 text-xs font-semibold text-emerald-800">
              Fixed-weights mode is ON with pinned run #{pinnedPredictionRunId}.
            </p>
          {/if}
          {#if nextDrawPrediction.ensembleReliability}
            <p class="mt-1 text-xs font-semibold text-emerald-800">
              Reliability gate selected {nextDrawPrediction.ensembleReliability.selectedMethod === "neural" ? "Neural only" : "the calibrated ensemble"}
              ({nextDrawPrediction.ensembleReliability.gateSampleCount} calibration samples;
              Neural {nextDrawPrediction.ensembleReliability.neuralAverageMatches.toFixed(2)},
              Ensemble {nextDrawPrediction.ensembleReliability.ensembleAverageMatches.toFixed(2)}).
            </p>
          {/if}
        {:else}
          <p class="mt-1 text-sm text-zinc-700">
            No completed run artifact is available yet. Complete at least one
            run to generate the next-draw prediction.
          </p>
        {/if}
      </div>
    </div>

    {#if nextDrawPrediction}
      <div class="mt-4 flex flex-wrap items-center gap-4 text-xs text-zinc-600">
        <span class="font-semibold text-zinc-700">Agreement key:</span>
        <span class="inline-flex items-center gap-1.5"><span class="h-4 w-4 rounded-full bg-zinc-700"></span>1 method</span>
        <span class="inline-flex items-center gap-1.5"><span class="h-4 w-4 rounded-full bg-cyan-700"></span>2 methods</span>
        <span class="inline-flex items-center gap-1.5"><span class="h-4 w-4 rounded-full bg-emerald-700"></span>3+ methods</span>
      </div>
      <div class="mt-4 grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
        {#each nextPredictionMethods as prediction}
          <div class:border-emerald-400={prediction.label === "Ensemble"} class="rounded-md border border-emerald-200 bg-white/80 p-3">
            <div class="flex items-center justify-between gap-2">
              <p class="text-xs font-semibold uppercase tracking-wide text-emerald-800">
                {prediction.label}
              </p>
              {#if prediction.weight !== null}
                <span class="text-xs text-zinc-500">
                  weight {(prediction.weight * 100).toFixed(1)}%
                </span>
              {/if}
            </div>
            <div class="mt-2 flex flex-wrap gap-2">
              {#each prediction.numbers as n}
                <span
                  class:bg-zinc-700={(nextPredictionAgreement.get(n)?.length ?? 0) === 1}
                  class:bg-cyan-700={(nextPredictionAgreement.get(n)?.length ?? 0) === 2}
                  class:bg-emerald-700={(nextPredictionAgreement.get(n)?.length ?? 0) >= 3}
                  class="relative inline-flex h-9 min-w-9 items-center justify-center rounded-full px-2 text-sm font-semibold text-white ring-2 ring-white"
                  title={`Selected by ${nextPredictionAgreement.get(n)?.join(", ") ?? prediction.label}`}
                  >{n}{#if (nextPredictionAgreement.get(n)?.length ?? 0) > 1}<span class="absolute -right-1.5 -top-1.5 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-white px-1 text-[10px] font-bold text-zinc-800 shadow">{nextPredictionAgreement.get(n)?.length}</span>{/if}</span
                >
              {/each}
            </div>
          </div>
        {/each}
      </div>

      {#if sharedNextPredictionNumbers.length > 0}
        <div class="mt-3 rounded-md border border-emerald-200 bg-white/80 p-3">
          <p class="text-xs font-semibold uppercase tracking-wide text-emerald-800">
            Shared numbers across methods
          </p>
          <div class="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {#each sharedNextPredictionNumbers as [number, methods]}
              <div class="flex items-center gap-2 rounded-md bg-emerald-50 px-2 py-1.5">
                <span class:bg-cyan-700={methods.length === 2} class:bg-emerald-700={methods.length >= 3} class="inline-flex h-8 min-w-8 items-center justify-center rounded-full px-2 text-sm font-bold text-white">{number}</span>
                <span class="text-xs leading-4 text-zinc-700">{methods.join(" · ")}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}

    <div class="mt-4 rounded-md border border-emerald-200 bg-white/80 p-3">
      <p class="text-xs font-semibold uppercase tracking-wide text-emerald-700">
        Fixed Weights Workflow
      </p>
      <p class="mt-1 text-sm text-zinc-700">
        1) Pin a completed run once. 2) Import new draw data when available. 3)
        Return here to get the next prediction from the same unchanged weights.
      </p>

      <div class="mt-3 rounded-md border border-emerald-100 bg-emerald-50/70 p-3 text-sm text-zinc-700">
        <p class="font-semibold text-emerald-900">How to select a specific run</p>
        <ol class="mt-1 list-decimal space-y-1 pl-5">
          <li>Open the completed-run list below and choose the run number you want.</li>
          <li>Click <span class="font-semibold text-zinc-900">Pin Run</span>.</li>
          <li>Confirm that the prediction heading says it is based on that run and that the fixed-weights message shows the same pinned run number.</li>
        </ol>
        <p class="mt-2 text-xs text-zinc-600">
          The pinned run remains selected after page refreshes and draw-data imports. Its saved model and ensemble weights generate every next-draw ticket until you select another run. Choose <span class="font-semibold text-zinc-800">Use Auto</span> to clear the pin and return to automatic run selection.
        </p>
      </div>

      <div class="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
        <form
          method="POST"
          action="?/setPredictionRun"
          use:enhance
          class="flex flex-1 gap-2"
        >
          <select
            name="predictionRunId"
            bind:value={predictionRunInput}
            class="h-10 min-w-[220px] flex-1 rounded-md border border-emerald-200 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-emerald-400 focus:outline-none"
          >
            <option value="">Select completed run...</option>
            {#each predictionRunCandidates as run}
              <option value={String(run.id)}>
                #{run.id} | holdout {run.holdoutScore ?? "n/a"} | win {run.windowSize}
                | layers [{run.hiddenLayers.join(", ")}]
              </option>
            {/each}
          </select>
          <Button type="submit" variant="outline">Pin Run</Button>
        </form>

        <form method="POST" action="?/clearPredictionRun" use:enhance>
          <Button type="submit" variant="outline">Use Auto</Button>
        </form>
      </div>

      {#if form?.predictionMessage}
        <p
          class:!text-emerald-700={form?.success}
          class:!text-rose-700={!form?.success}
          class="mt-2 text-sm text-zinc-700"
        >
          {form.predictionMessage}
        </p>
      {:else if predictionRunMessage}
        <p class="mt-2 text-sm text-amber-700">{predictionRunMessage}</p>
      {/if}
    </div>
  </Card>

  <Card>
    <div class="mb-4 flex items-center justify-between">
      <h2 class="text-xl font-semibold text-zinc-900">Create Training Run</h2>
      <Badge variant="subtle">Custom Config</Badge>
    </div>

    {#if latestRunFormConfig}
      <div class="mb-4 rounded-md border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
        <p class="font-semibold">
          Showing the exact values used by run #{latestRunFormConfig.runId}.
        </p>
        <p class="mt-1 text-xs text-emerald-800">
          Run status: {latestRunFormConfig.status}. These recorded values remain visible after the run ends and can be edited to create the next run.
        </p>
      </div>
    {/if}

    <div
      class="mb-4 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3 text-sm text-zinc-700"
    >
      <p class="font-semibold text-zinc-900">What Each Option Does</p>
      <p class="mt-1 text-xs">
        Preset / Hidden layers: controls model size. Bigger can fit more
        patterns but may overfit.
      </p>
      <p class="mt-1 text-xs">
        Holdout weeks: how much recent data is kept for testing. Higher gives
        stricter evaluation but less training data.
      </p>
      <p class="mt-1 text-xs">
        Window size: how many past draws are used as input. Higher means more
        context and more parameters.
      </p>
      <p class="mt-1 text-xs">
        Epochs: number of passes over training data. More can improve fit up to
        a point, then overfit.
      </p>
      <p class="mt-1 text-xs">
        Batch size: samples per weight update. Larger is smoother/faster,
        smaller can generalize better.
      </p>
      <p class="mt-1 text-xs">
        Learning rate: step size for optimization. Too high can destabilize; too
        low can train very slowly.
      </p>
      <p class="mt-1 text-xs">
        Dropout rate: regularization strength. Higher reduces overfitting but
        too high can underfit.
      </p>
      <p class="mt-1 text-xs">
        Positive-class weight: how strongly the loss emphasizes numbers that
        were actually drawn. Lower values are less aggressive.
      </p>
      <p class="mt-1 text-xs">
        Early stopping / seed: controls when training stops after validation
        stalls and makes repeated runs reproducible.
      </p>
    </div>

    <form method="POST" action="?/createRun" use:enhance={preserveCreateRunValues}>
      <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Preset
          <select
            name="preset"
            bind:value={selectedPreset}
            onchange={(event) => {
              if (event.currentTarget.value !== "custom") {
                customHiddenLayersText = "";
              }
            }}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          >
            <option value="custom">custom (use layers field)</option>
            <option value="small">small (64,32)</option>
            <option value="medium">medium (128,64)</option>
            <option value="large">large (256,128)</option>
            <option value="xlarge">xlarge (512,256)</option>
          </select>
          <p class="text-xs font-normal text-zinc-500">
            Quick model-size choice. Start with medium unless testing capacity
            limits.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Custom hidden layers
          <input
            name="hiddenLayers"
            placeholder="128,64"
            bind:value={customHiddenLayersText}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Overrides preset. Example: 128,64. More neurons increase capacity
            and runtime.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Holdout weeks
          <input
            type="number"
            name="holdoutWeeks"
            min="1"
            bind:value={holdoutWeeksInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Recent weeks reserved for validation. Increase for tougher realism
            checks.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Window size
          <input
            type="number"
            name="windowSize"
            min="1"
            bind:value={windowSizeInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Number of past draws used as input. Higher adds context and
            complexity.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Epochs
          <input
            type="number"
            name="epochs"
            min="1"
            bind:value={epochsInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Training rounds over the dataset. More epochs can improve or
            overfit.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Batch size
          <input
            type="number"
            name="batchSize"
            min="1"
            bind:value={batchSizeInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Samples per update. Lower is noisier; higher is steadier and often
            faster.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Learning rate
          <input
            type="number"
            name="learningRate"
            min="0.000001"
            step="0.0001"
            bind:value={learningRateInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Optimizer step size. Reduce if loss is unstable; increase if
            learning is too slow.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Dropout rate
          <input
            type="number"
            name="dropoutRate"
            min="0"
            max="0.99"
            step="0.01"
            bind:value={dropoutRateInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Regularization level. Raise to fight overfitting; lower if model
            underfits.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Positive-class weight
          <input
            type="number"
            name="positiveClassWeight"
            min="0.1"
            max="20"
            step="0.1"
            bind:value={positiveClassWeightInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Try 3.0 for the next run. The class-balance default is {defaults.positiveClassWeight.toFixed(2)}.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Early-stopping patience
          <input
            type="number"
            name="earlyStoppingPatience"
            min="1"
            max="100"
            step="1"
            bind:value={earlyStoppingPatienceInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Epochs allowed without meaningful validation improvement. Try 8.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Minimum improvement
          <input
            type="number"
            name="earlyStoppingMinDelta"
            min="0.000001"
            max="1"
            step="0.000001"
            bind:value={earlyStoppingMinDeltaInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Smallest validation-loss reduction counted as progress. Try 0.00005.
          </p>
        </label>

        <label class="space-y-1 text-sm font-medium text-zinc-700">
          Training seed
          <input
            type="number"
            name="trainingSeed"
            min="1"
            max="2147483646"
            step="1"
            bind:value={trainingSeedInput}
            class="h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none"
          />
          <p class="text-xs font-normal text-zinc-500">
            Reproduces initialization and dropout randomness. Start with 42.
          </p>
        </label>
      </div>

      <div class="mt-4 flex flex-wrap items-center gap-2">
        <Badge>Layers [{selectedHiddenLayers.join(", ")}]</Badge>
        <Badge variant="subtle">Estimated Params {previewParamCount}</Badge>
      </div>

      <div class="mt-5">
        <Button type="submit">Create Run</Button>
      </div>
    </form>

    {#if form?.message}
      <p
        class:!text-emerald-700={form?.success}
        class:!text-rose-700={!form?.success}
        class="mt-3 rounded-md bg-zinc-50 px-3 py-2 text-sm text-zinc-700"
      >
        {form.message}
      </p>
    {/if}

    {#if createdRun}
      <p class="mt-3 text-sm text-emerald-700">
        Run #{createdRun.id} created with hidden layers [{createdRun.hiddenLayers.join(
          ", ",
        )}]
      </p>
    {/if}
  </Card>

  <Card class="border-sky-200 bg-gradient-to-br from-sky-50 to-white">
    <div
      class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"
    >
      <h2 class="text-xl font-semibold text-zinc-900">Feedback Loop Status</h2>
      <form
        method="POST"
        action="?/setPolicyMode"
        use:enhance
        class="flex items-center gap-2"
      >
        <select
          name="policyMode"
          bind:value={policyModeInput}
          class="h-10 rounded-md border border-sky-200 bg-white px-3 text-sm text-zinc-900 shadow-sm focus:border-sky-400 focus:outline-none"
        >
          <option value="off">off</option>
          <option value="shadow">shadow</option>
          <option value="active">active</option>
        </select>
        <Button type="submit" variant="outline">Update Mode</Button>
      </form>
    </div>
    {#if form?.policyMessage}
      <p
        class:!text-emerald-700={form?.success}
        class:!text-rose-700={!form?.success}
        class="mt-2 text-sm text-zinc-700"
      >
        {form.policyMessage}
      </p>
    {/if}
    <div class="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <div class="rounded-md border border-sky-100 bg-white p-3">
        <p class="text-xs uppercase tracking-wide text-zinc-500">Policy Mode</p>
        <p class="text-lg font-semibold text-zinc-900">{feedbackStatus.mode}</p>
      </div>
      <div class="rounded-md border border-sky-100 bg-white p-3">
        <p class="text-xs uppercase tracking-wide text-zinc-500">
          Pending Decisions
        </p>
        <p class="text-lg font-semibold text-zinc-900">
          {feedbackStatus.pendingDecisions}
        </p>
      </div>
      <div class="rounded-md border border-sky-100 bg-white p-3">
        <p class="text-xs uppercase tracking-wide text-zinc-500">
          Resolved Outcomes
        </p>
        <p class="text-lg font-semibold text-zinc-900">
          {feedbackStatus.resolvedOutcomes}
        </p>
      </div>
      <div class="rounded-md border border-sky-100 bg-white p-3">
        <p class="text-xs uppercase tracking-wide text-zinc-500">
          Recent Avg Reward
        </p>
        <p class="text-lg font-semibold text-zinc-900">
          {feedbackStatus.recentAverageReward}
        </p>
      </div>
      <div class="rounded-md border border-sky-100 bg-white p-3">
        <p class="text-xs uppercase tracking-wide text-zinc-500">Last Reward</p>
        <p class="text-lg font-semibold text-zinc-900">
          {feedbackStatus.lastReward ?? "n/a"}
        </p>
      </div>
      <div class="rounded-md border border-sky-100 bg-white p-3">
        <p class="text-xs uppercase tracking-wide text-zinc-500">
          Exploration Rate
        </p>
        <p class="text-lg font-semibold text-zinc-900">
          {feedbackStatus.explorationRate}
        </p>
      </div>
    </div>
    {#if feedbackStatus.lastResolvedAt}
      <p class="mt-3 text-xs text-zinc-600">
        Last resolved at: {feedbackStatus.lastResolvedAt}
      </p>
    {/if}
  </Card>

  <Card>
    <h2 class="text-xl font-semibold text-zinc-900">Active Progress</h2>

    {#if activeProgress}
      <div class="mt-4 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        <div class="rounded-md border border-zinc-200 bg-white p-3">
          <p class="text-xs uppercase tracking-wide text-zinc-500">Run</p>
          <p class="text-lg font-semibold text-zinc-900">
            #{activeProgress.id}
          </p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-white p-3">
          <p class="text-xs uppercase tracking-wide text-zinc-500">Status</p>
          <p class="text-lg font-semibold text-zinc-900">
            {activeProgress.status}
          </p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-white p-3">
          <p class="text-xs uppercase tracking-wide text-zinc-500">Epoch</p>
          <p class="text-lg font-semibold text-zinc-900">
            {activeProgress.currentEpoch}/{activeProgress.totalEpochs}
          </p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-white p-3">
          <p class="text-xs uppercase tracking-wide text-zinc-500">
            Rounds Remaining
          </p>
          <p class="text-lg font-semibold text-zinc-900">
            {activeProgress.roundsRemaining}
          </p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-white p-3">
          <p class="text-xs uppercase tracking-wide text-zinc-500">
            Total Work
          </p>
          <p class="text-lg font-semibold text-zinc-900">
            {activeProgress.samplesProcessed}
            {#if activeProgress.samplesTotal}
              / {activeProgress.samplesTotal} ({progressPct}%)
            {/if}
          </p>
        </div>
        <div class="rounded-md border border-zinc-200 bg-white p-3">
          <p class="text-xs uppercase tracking-wide text-zinc-500">
            Parameters
          </p>
          <p class="text-lg font-semibold text-zinc-900">
            {activeProgress.paramCount ?? "n/a"}
          </p>
        </div>
      </div>

      <div class="mt-4 flex flex-wrap gap-2 text-sm text-zinc-700">
        <span
          >Holdout score: <strong>{activeProgress.holdoutScore ?? "n/a"}</strong
          ></span
        >
        <span
          >Final losses: <strong
            >{activeProgress.finalTrainLoss ?? "n/a"}</strong
          >
          / <strong>{activeProgress.finalValLoss ?? "n/a"}</strong></span
        >
      </div>

      <form method="POST" action="?/progress" use:enhance class="mt-4">
        <input type="hidden" name="runId" value={activeProgress.id} />
        <Button type="submit" variant="outline">Refresh Progress</Button>
      </form>
    {:else}
      <p class="mt-2 text-sm text-zinc-600">No run selected yet.</p>
    {/if}
  </Card>

  <Card>
    <h2 class="text-xl font-semibold text-zinc-900">Recent Runs</h2>

    {#if latestRuns.length === 0}
      <p class="mt-2 text-sm text-zinc-600">No runs yet.</p>
    {:else}
      <div class="mt-4 overflow-x-auto">
        <table class="w-full min-w-[760px] border-collapse text-sm">
          <thead>
            <tr class="border-b border-zinc-200 text-left text-zinc-600">
              <th class="px-3 py-2 font-semibold">ID</th>
              <th class="px-3 py-2 font-semibold">Status</th>
              <th class="px-3 py-2 font-semibold">Epoch</th>
              <th class="px-3 py-2 font-semibold">Window</th>
              <th class="px-3 py-2 font-semibold">Remaining</th>
              <th class="px-3 py-2 font-semibold">Hidden Layers</th>
              <th class="px-3 py-2 font-semibold">Holdout</th>
              <th class="px-3 py-2 font-semibold">Params</th>
            </tr>
          </thead>
          <tbody>
            {#each latestRuns as run}
              <tr class="border-b border-zinc-100 text-zinc-800">
                <td class="px-3 py-2">{run.id}</td>
                <td class="px-3 py-2">{run.status}</td>
                <td class="px-3 py-2">{run.currentEpoch}/{run.totalEpochs}</td>
                <td class="px-3 py-2">{run.windowSize}</td>
                <td class="px-3 py-2">{run.roundsRemaining}</td>
                <td class="px-3 py-2">[{run.hiddenLayers.join(", ")}]</td>
                <td class="px-3 py-2">{run.holdoutScore ?? "n/a"}</td>
                <td class="px-3 py-2">{run.paramCount ?? "n/a"}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </Card>

  <Card>
    <h2 class="text-xl font-semibold text-zinc-900">Run Comparison</h2>
    <p class="mt-2 text-sm text-zinc-600">
      Compares methods on the same holdout draws. New runs will include the
      learned ensemble method automatically. Ensemble weights update only after
      each holdout draw has been scored.
    </p>

    {#if runComparisonSummaries.length === 0}
      <p class="mt-2 text-sm text-zinc-600">No comparable runs yet.</p>
    {:else}
      <div class="mt-4 overflow-x-auto">
        <table class="w-full min-w-[980px] border-collapse text-sm">
          <thead>
            <tr class="border-b border-zinc-200 text-left text-zinc-600">
              <th class="px-3 py-2 font-semibold">Run</th>
              <th class="px-3 py-2 font-semibold">Status</th>
              <th class="px-3 py-2 font-semibold">Holdout</th>
              <th class="px-3 py-2 font-semibold">Window</th>
              <th class="px-3 py-2 font-semibold">Model</th>
              <th class="px-3 py-2 font-semibold">Frequency</th>
              <th class="px-3 py-2 font-semibold">Random</th>
              <th class="px-3 py-2 font-semibold">Heuristic</th>
              <th class="px-3 py-2 font-semibold">Ensemble</th>
              <th class="px-3 py-2 font-semibold">Model - Heuristic</th>
              <th class="px-3 py-2 font-semibold">Winner</th>
            </tr>
          </thead>
          <tbody>
            {#each runComparisonSummaries as summary}
              <tr class="border-b border-zinc-100 text-zinc-800">
                <td class="px-3 py-2">#{summary.runId}</td>
                <td class="px-3 py-2">{summary.status}</td>
                <td class="px-3 py-2">{summary.holdoutWeeks}w</td>
                <td class="px-3 py-2">{summary.windowSize}</td>
                <td class="px-3 py-2">
                  {summary.modelAverage?.toFixed(2) ?? "n/a"}
                </td>
                <td class="px-3 py-2">
                  {summary.frequencyAverage?.toFixed(2) ?? "n/a"}
                </td>
                <td class="px-3 py-2">
                  {summary.randomAverage?.toFixed(2) ?? "n/a"}
                </td>
                <td class="px-3 py-2">
                  {summary.heuristicAverage?.toFixed(2) ?? "n/a"}
                </td>
                <td class="px-3 py-2">
                  {summary.ensembleAverage?.toFixed(2) ?? "n/a"}
                </td>
                <td class="px-3 py-2">
                  {summary.modelVsHeuristic?.toFixed(2) ?? "n/a"}
                </td>
                <td class="px-3 py-2">{summary.winner ?? "n/a"}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </Card>

  <div>
    <Card>
      <h2 class="text-xl font-semibold text-zinc-900">
        Holdout Results {holdoutResultsRunId ? `(Run #${holdoutResultsRunId})` : ""}
      </h2>
      {#if pinnedPredictionRunId && holdoutResultsRunId === pinnedPredictionRunId}
        <p class="mt-1 text-xs font-semibold text-emerald-700">
          Showing the holdout predictions and scores for pinned run #{pinnedPredictionRunId}.
        </p>
      {/if}
      {#if holdoutComparison}
        <p class="mt-2 text-sm text-zinc-600">
          Average matches: model {holdoutComparison.model?.toFixed(2) ?? "n/a"},
          frequency {holdoutComparison.frequency?.toFixed(2) ?? "n/a"}, random {holdoutComparison.random?.toFixed(
            2,
          ) ?? "n/a"}, heuristic {holdoutComparison.heuristic?.toFixed(2) ??
            "n/a"}, ensemble {holdoutComparison.ensemble?.toFixed(2) ?? "n/a"}.
        </p>
      {/if}

      {#if holdoutRunDetail && holdoutRunDetail.methodSummaries.length > 0}
        <div class="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {#each holdoutRunDetail.methodSummaries as summary}
            <div class="rounded-md border border-zinc-200 bg-white p-3">
              <p class="text-xs uppercase tracking-wide text-zinc-500">
                {summary.method}
              </p>
              <p class="mt-1 text-lg font-semibold text-zinc-900">
                {summary.averageMatches.toFixed(2)}
              </p>
              <p class="text-xs text-zinc-600">
                hit {(summary.topKHitRate * 100).toFixed(0)}%, 2+
                {(summary.twoPlusMatchRate * 100).toFixed(0)}%, 3+
                {(summary.threePlusMatchRate * 100).toFixed(0)}%
              </p>
              <p class="mt-1 text-xs text-zinc-500">
                95% CI {summary.confidenceLow95.toFixed(2)}–{summary.confidenceHigh95.toFixed(2)}
              </p>
            </div>
          {/each}
        </div>
      {/if}

      {#if holdoutRunDetail && holdoutRunDetail.testResults.length > 0}
        <div class="mt-4 flex flex-wrap items-center gap-4 text-xs text-zinc-600">
          <span class="font-semibold text-zinc-700">Number key:</span>
          <span class="inline-flex items-center gap-1.5">
            <span class="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-emerald-600 px-1.5 font-semibold text-white">✓</span>
            Matches actual
          </span>
          <span class="inline-flex items-center gap-1.5">
            <span class="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-zinc-200 px-1.5 font-semibold text-zinc-700">–</span>
            No match
          </span>
        </div>
        <div class="mt-4 overflow-x-auto">
          <table class="w-full min-w-[980px] border-collapse text-sm">
            <thead>
              <tr class="border-b border-zinc-200 text-left text-zinc-600">
                <th class="px-3 py-2 font-semibold">Draw #</th>
                <th class="px-3 py-2 font-semibold">Draw Date</th>
                <th class="px-3 py-2 font-semibold">Game</th>
                <th class="px-3 py-2 font-semibold">Seq</th>
                <th class="px-3 py-2 font-semibold">Round</th>
                <th class="px-3 py-2 font-semibold">Draw Row ID</th>
                <th class="px-3 py-2 font-semibold">Predicted</th>
                <th class="px-3 py-2 font-semibold">Actual</th>
                <th class="px-3 py-2 font-semibold">Matches</th>
                <th class="px-3 py-2 font-semibold">Frequency</th>
                <th class="px-3 py-2 font-semibold">Random</th>
                <th class="px-3 py-2 font-semibold">Heuristic</th>
                <th class="px-3 py-2 font-semibold">Ensemble</th>
                <th class="px-3 py-2 font-semibold">Top-K Hit</th>
              </tr>
            </thead>
            <tbody>
              {#each holdoutRunDetail.testResults as result}
                <tr class="border-b border-zinc-100 text-zinc-800">
                  <td class="px-3 py-2"
                    >{result.drawNumber ? `#${result.drawNumber}` : "n/a"}</td
                  >
                  <td class="px-3 py-2">{result.drawDate ?? "n/a"}</td>
                  <td class="px-3 py-2">{result.gameName ?? "n/a"}</td>
                  <td class="px-3 py-2">{result.drawSequence ?? "n/a"}</td>
                  <td class="px-3 py-2">{result.drawRound ?? "n/a"}</td>
                  <td class="px-3 py-2">{result.drawId ?? "n/a"}</td>
                  <td class="px-3 py-2">
                    <div class="flex min-w-48 flex-wrap gap-1">
                      {#each result.predictedNumbers as number}
                        <span class:bg-emerald-600={result.actualNumbers.includes(number)} class:text-white={result.actualNumbers.includes(number)} class:bg-zinc-200={!result.actualNumbers.includes(number)} class:text-zinc-700={!result.actualNumbers.includes(number)} class="inline-flex h-7 min-w-7 items-center justify-center rounded-full px-1.5 text-xs font-semibold">{number}</span>
                      {/each}
                    </div>
                  </td>
                  <td class="px-3 py-2">
                    <div class="flex min-w-48 flex-wrap gap-1">
                      {#each result.actualNumbers as number}
                        <span class="inline-flex h-7 min-w-7 items-center justify-center rounded-full bg-sky-700 px-1.5 text-xs font-semibold text-white">{number}</span>
                      {/each}
                    </div>
                  </td>
                  <td class="px-3 py-2">
                    <span class:bg-emerald-100={result.matchCount > 0} class:text-emerald-800={result.matchCount > 0} class:bg-zinc-100={result.matchCount === 0} class:text-zinc-600={result.matchCount === 0} class="inline-flex min-w-8 justify-center rounded-full px-2 py-1 font-bold">{result.matchCount}</span>
                  </td>
                  <td class="px-3 py-2">
                    {#if result.frequencyPredictedNumbers}
                      <div class="flex min-w-48 flex-wrap gap-1">
                        {#each result.frequencyPredictedNumbers as number}
                          <span class:bg-emerald-600={result.actualNumbers.includes(number)} class:text-white={result.actualNumbers.includes(number)} class:bg-zinc-200={!result.actualNumbers.includes(number)} class:text-zinc-700={!result.actualNumbers.includes(number)} class="inline-flex h-7 min-w-7 items-center justify-center rounded-full px-1.5 text-xs font-semibold">{number}</span>
                        {/each}
                        <span class="ml-1 self-center font-bold text-zinc-600">{result.frequencyMatchCount ?? "n/a"}</span>
                      </div>
                    {:else}
                      n/a
                    {/if}
                  </td>
                  <td class="px-3 py-2">
                    {#if result.randomPredictedNumbers}
                      <div class="flex min-w-48 flex-wrap gap-1">
                        {#each result.randomPredictedNumbers as number}
                          <span class:bg-emerald-600={result.actualNumbers.includes(number)} class:text-white={result.actualNumbers.includes(number)} class:bg-zinc-200={!result.actualNumbers.includes(number)} class:text-zinc-700={!result.actualNumbers.includes(number)} class="inline-flex h-7 min-w-7 items-center justify-center rounded-full px-1.5 text-xs font-semibold">{number}</span>
                        {/each}
                        <span class="ml-1 self-center font-bold text-zinc-600">{result.randomMatchCount ?? "n/a"}</span>
                      </div>
                    {:else}
                      n/a
                    {/if}
                  </td>
                  <td class="px-3 py-2">
                    {#if result.heuristicPredictedNumbers}
                      <div class="flex min-w-48 flex-wrap gap-1">
                        {#each result.heuristicPredictedNumbers as number}
                          <span class:bg-emerald-600={result.actualNumbers.includes(number)} class:text-white={result.actualNumbers.includes(number)} class:bg-zinc-200={!result.actualNumbers.includes(number)} class:text-zinc-700={!result.actualNumbers.includes(number)} class="inline-flex h-7 min-w-7 items-center justify-center rounded-full px-1.5 text-xs font-semibold">{number}</span>
                        {/each}
                        <span class="ml-1 self-center font-bold text-zinc-600">{result.heuristicMatchCount ?? "n/a"}</span>
                      </div>
                    {:else}
                      n/a
                    {/if}
                  </td>
                  <td class="px-3 py-2">
                    {#if result.ensemblePredictedNumbers}
                      <div class="flex min-w-48 flex-wrap gap-1">
                        {#each result.ensemblePredictedNumbers as number}
                          <span class:bg-emerald-600={result.actualNumbers.includes(number)} class:text-white={result.actualNumbers.includes(number)} class:bg-zinc-200={!result.actualNumbers.includes(number)} class:text-zinc-700={!result.actualNumbers.includes(number)} class="inline-flex h-7 min-w-7 items-center justify-center rounded-full px-1.5 text-xs font-semibold">{number}</span>
                        {/each}
                        <span class="ml-1 self-center font-bold text-zinc-600">{result.ensembleMatchCount ?? "n/a"}</span>
                      </div>
                    {:else}
                      n/a
                    {/if}
                  </td>
                  <td class="px-3 py-2">{result.topKHit ? "yes" : "no"}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <p class="mt-2 text-sm text-zinc-600">
          No holdout predictions are stored for this run yet.
        </p>
      {/if}
    </Card>
  </div>
</div>
