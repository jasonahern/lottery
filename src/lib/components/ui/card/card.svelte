<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    children,
    class: className = "",
    ...rest
  }: {
    children?: Snippet;
    class?: string;
    [key: string]: unknown;
  } = $props();
</script>

<section class={`rounded-xl border border-zinc-200 bg-white/90 p-5 shadow-sm backdrop-blur ${className}`} {...rest}>
  {@render children?.()}
</section>
