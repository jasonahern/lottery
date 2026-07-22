<script lang="ts">
  import type { Snippet } from "svelte";

  type ButtonVariant = "default" | "secondary" | "outline" | "ghost";
  type ButtonSize = "default" | "sm" | "lg";

  let {
    children,
    class: className = "",
    variant = "default",
    size = "default",
    type = "button",
    ...rest
  }: {
    children?: Snippet;
    class?: string;
    variant?: ButtonVariant;
    size?: ButtonSize;
    type?: "button" | "submit" | "reset";
    [key: string]: unknown;
  } = $props();

  const base =
    "inline-flex cursor-pointer items-center justify-center rounded-md text-sm font-medium transition-[transform,colors,box-shadow] duration-100 active:scale-[0.98] active:translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ring-offset-white";

  const variants: Record<ButtonVariant, string> = {
    default: "bg-zinc-900 text-zinc-50 hover:bg-zinc-800 active:bg-zinc-700",
    secondary: "bg-zinc-100 text-zinc-900 hover:bg-zinc-200 active:bg-zinc-300",
    outline:
      "border border-zinc-300 bg-white text-zinc-900 hover:bg-zinc-50 active:bg-zinc-100 active:border-zinc-400",
    ghost: "text-zinc-800 hover:bg-zinc-100 active:bg-zinc-200",
  };

  const sizes: Record<ButtonSize, string> = {
    default: "h-10 px-4 py-2",
    sm: "h-9 rounded-md px-3",
    lg: "h-11 rounded-md px-8",
  };
</script>

<button
  {type}
  class={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
  {...rest}
>
  {@render children?.()}
</button>
