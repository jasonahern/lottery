<script lang="ts">
  import type { Snippet } from "svelte";

  type BadgeVariant = "default" | "subtle";

  let {
    children,
    class: className = "",
    variant = "default",
  }: {
    children?: Snippet;
    class?: string;
    variant?: BadgeVariant;
  } = $props();

  const variants: Record<BadgeVariant, string> = {
    default: "bg-zinc-900 text-zinc-50",
    subtle: "bg-zinc-100 text-zinc-700",
  };
</script>

<span class={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${variants[variant]} ${className}`}>
  {@render children?.()}
</span>
