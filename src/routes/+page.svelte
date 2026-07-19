<script lang="ts">
  import { enhance } from "$app/forms";
  import type { SubmitFunction } from "@sveltejs/kit";
  import { toast } from "svelte-sonner";

  import { Button } from "$lib/components/ui/button";
  import { Card } from "$lib/components/ui/card";

  function getMessages(data: unknown, fallback: string): string[] {
    if (data && typeof data === "object" && "messages" in data && Array.isArray(data.messages)) {
      const messages = data.messages.filter((msg): msg is string => typeof msg === "string");
      if (messages.length > 0) {
        return messages;
      }
    }

    if (data && typeof data === "object" && "message" in data && typeof data.message === "string") {
      return [data.message];
    }

    return [fallback];
  }

  const submitHandler: SubmitFunction = () => {
    const loadingId = toast.loading("Downloading lottery data...");

    return async ({ result, update }) => {
      await update({ reset: false });

      if (result.type === "success") {
        const [first, ...rest] = getMessages(result.data, "Lottery data downloaded successfully.");
        toast.success(first, { id: loadingId });

        rest.forEach((message, index) => {
          setTimeout(() => {
            toast.success(message);
          }, 1000 * (index + 1));
        });
        return;
      }

      if (result.type === "failure") {
        toast.error(getMessages(result.data, "Download failed.")[0], { id: loadingId });
        return;
      }

      if (result.type === "error") {
        toast.error(result.error?.message ?? "An unexpected error occurred.", { id: loadingId });
        return;
      }

      if (result.type === "redirect") {
        toast.success("Lottery data downloaded successfully.", { id: loadingId });
      }
    };
  };
</script>

<main class="relative mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 sm:px-6 lg:px-8">
  <div class="pointer-events-none absolute inset-x-0 top-0 -z-10 h-56 bg-gradient-to-r from-sky-100 via-cyan-100 to-emerald-100 blur-3xl"></div>

  <header class="space-y-3">
    <h1 class="text-4xl font-bold tracking-tight text-zinc-900">Lottery Control Center</h1>
    <p class="max-w-3xl text-zinc-600">
      Download the latest lottery dataset, inspect the SQLite database, and run TensorFlow training workflows from one place.
    </p>
  </header>

  <div class="grid gap-4 md:grid-cols-3">
    <Card class="border-sky-200 bg-gradient-to-br from-sky-50 to-white">
      <h2 class="text-lg font-semibold text-zinc-900">Data Ingestion</h2>
      <p class="mt-2 text-sm text-zinc-600">Fetch the newest draw history, update CSV, and reload the local database.</p>
      <form method="POST" use:enhance={submitHandler} class="mt-4">
        <Button type="submit">Download and Reload Database</Button>
      </form>
    </Card>

    <Card>
      <h2 class="text-lg font-semibold text-zinc-900">Database View</h2>
      <p class="mt-2 text-sm text-zinc-600">Browse imported entries and verify the ingestion pipeline output.</p>
      <div class="mt-4">
        <a href="/database" class="inline-flex items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50">
          Open Database Route
        </a>
      </div>
    </Card>

    <Card class="border-emerald-200 bg-gradient-to-br from-emerald-50 to-white">
      <h2 class="text-lg font-semibold text-zinc-900">Neural Training</h2>
      <p class="mt-2 text-sm text-zinc-600">Start training runs, monitor progress, and compare holdout results.</p>
      <div class="mt-4 flex flex-wrap gap-2">
        <a href="/training" class="inline-flex items-center rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-zinc-50 hover:bg-zinc-800">
          Open Training Dashboard
        </a>
        <a href="/training/runs" class="inline-flex items-center rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50">
          Runs API
        </a>
      </div>
    </Card>
  </div>

  <Card>
    <h2 class="text-lg font-semibold text-zinc-900">Relevant Routes</h2>
    <div class="mt-3 grid gap-2 text-sm text-zinc-700 sm:grid-cols-2">
      <a href="/" class="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 hover:bg-zinc-100">/</a>
      <a href="/database" class="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 hover:bg-zinc-100">/database</a>
      <a href="/training" class="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 hover:bg-zinc-100">/training</a>
      <a href="/training/runs" class="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 hover:bg-zinc-100">/training/runs</a>
    </div>
  </Card>
</main>
