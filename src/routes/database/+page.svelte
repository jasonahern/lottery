<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
</script>

<main class="mx-auto max-w-6xl px-6 py-12">
	<div class="mb-6 flex items-center justify-between gap-4">
		<div>
			<h1 class="text-3xl font-bold tracking-tight">Database Viewer</h1>
			<p class="mt-2 text-slate-600">
				Showing <span class="font-semibold">{data.entries.length}</span> rows from
				<code class="rounded bg-slate-100 px-1 py-0.5 text-sm">lottery_draws + lottery_draw_balls</code>.
			</p>
		</div>
		<a href="/" class="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-900">
			Back to Download
		</a>
	</div>

	<div class="overflow-x-auto rounded-lg border border-slate-200">
		<table class="min-w-full divide-y divide-slate-200 text-sm">
			<thead class="bg-slate-50">
				<tr>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Draw ID</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Draw Number</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Round</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Date</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Game</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Main Balls</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Bonus</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Jackpot</th>
					<th class="px-4 py-3 text-left font-semibold text-slate-700">Imported At</th>
				</tr>
			</thead>
			<tbody class="divide-y divide-slate-100 bg-white">
				{#if data.entries.length === 0}
					<tr>
						<td colspan="9" class="px-4 py-6 text-center text-slate-500">No data in database.</td>
					</tr>
				{:else}
					{#each data.entries as entry}
						<tr>
							<td class="whitespace-nowrap px-4 py-3 font-mono text-slate-700">{entry.id}</td>
							<td class="whitespace-nowrap px-4 py-3 font-mono text-slate-700">#{entry.drawNumber}</td>
							<td class="whitespace-nowrap px-4 py-3 font-semibold text-slate-700">Round {entry.drawRound}</td>
							<td class="whitespace-nowrap px-4 py-3 text-slate-700">
								{new Date(entry.drawDate).toLocaleDateString('en-GB')}
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-slate-700">{entry.gameName ?? 'n/a'}</td>
							<td class="px-4 py-3 font-mono text-slate-700">[{entry.mainBalls.join(', ')}]</td>
							<td class="whitespace-nowrap px-4 py-3 font-mono text-slate-700">{entry.bonusBall ?? 'n/a'}</td>
							<td class="whitespace-nowrap px-4 py-3 text-slate-700">{entry.jackpotAmount ?? 'n/a'}</td>
							<td class="whitespace-nowrap px-4 py-3 text-slate-700">
								{new Date(entry.createdAt).toLocaleString('en-GB')}
							</td>
						</tr>
					{/each}
				{/if}
			</tbody>
		</table>
	</div>
</main>
