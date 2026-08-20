<script lang="ts">
  import { featureFlags, setMemoryGuard } from "$lib/stores/app.svelte";

  let saving = $state(false);
  let errorMessage = $state<string | null>(null);

  const flags = $derived(featureFlags());
  const enabled = $derived(flags["prefillMemoryGuard"] === true);

  async function toggle() {
    if (saving) return;
    saving = true;
    errorMessage = null;
    try {
      await setMemoryGuard(!enabled);
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }
</script>

<div class="font-mono text-foreground">
  <details open class="group [&_summary::-webkit-details-marker]:hidden">
    <summary
      class="cursor-pointer list-none text-exo-yellow text-xs font-mono tracking-widest uppercase flex items-center gap-2 hover:opacity-80 transition-opacity"
    >
      <span
        class="inline-block transition-transform group-open:rotate-90 text-exo-light-gray"
        >▶</span
      >
      Prefill memory guard
    </summary>
    <div class="mt-2 text-white/80 text-sm leading-relaxed space-y-2">
      <p>
        When enabled, EXO estimates each prefill's peak memory before it runs
        and rejects — with a clean error instead of a crash — any prefill that
        would exceed the reclaim-based memory ceiling
        (<code class="text-exo-yellow">phys_footprint + free + inactive +
          active&nbsp;×&nbsp;reclaim</code>). A per-chunk abort acts as a
        last-resort safety net mid-prefill.
      </p>
      <p class="text-white/60 text-xs">
        Ship default is <strong>off</strong> (pre-task-#11 behaviour). Enable
        on clusters running large models near the memory limit, e.g. a 2×256 GB
        pair with <code class="text-exo-yellow">iogpu.wired_limit_mb=256000</code>.
        Tiers (safe/balanced/aggressive) are set via
        <code class="text-exo-yellow">EXO_MEMORY_GUARD_TIER</code>; the escape
        hatch is <code class="text-exo-yellow">EXO_DISABLE_PREFILL_GUARD=1</code>.
      </p>
    </div>
  </details>

  {#if errorMessage}
    <div
      class="mt-3 px-4 py-3 bg-red-500/10 border border-red-500/40 text-red-300 text-sm"
    >
      {errorMessage}
    </div>
  {/if}

  <div
    class="mt-4 flex items-center justify-between bg-exo-dark-gray/60 border border-exo-medium-gray/40 px-4 py-3"
  >
    <div class="flex flex-col gap-0.5">
      <span class="text-sm text-white/90">
        Prefill memory guard
        {#if enabled}
          <span class="text-green-400 text-xs ml-2">● ON</span>
        {:else}
          <span class="text-white/40 text-xs ml-2">○ OFF</span>
        {/if}
      </span>
      <span class="text-[11px] text-white/45">
        {saving
          ? "Applying…"
          : enabled
            ? "Prefills that would OOM are rejected cleanly."
            : "No preflight rejection (pre-task-#11 behaviour)."}
      </span>
    </div>
    <button
      class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase transition-colors border disabled:opacity-50 disabled:cursor-not-allowed {enabled
        ? 'bg-red-500/15 border-red-500/40 text-red-300 hover:bg-red-500/25'
        : 'bg-exo-yellow/15 border-exo-yellow/50 text-exo-yellow hover:bg-exo-yellow/25 hover:border-exo-yellow/80'}"
      onclick={toggle}
      disabled={saving}
    >
      {saving ? "…" : enabled ? "Disable" : "Enable"}
    </button>
  </div>
</div>