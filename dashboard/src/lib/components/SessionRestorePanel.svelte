<script lang="ts">
  import {
    sessions,
    launchLastSessionModel,
    launchSessionModel,
    saveSession,
    deleteSession,
  } from "$lib/stores/app.svelte";
  import type { SavedSession, SessionEntry } from "$lib/stores/app.svelte";
  import { addToast } from "$lib/stores/toast.svelte";

  interface Props {
    instanceCount: number;
  }
  let { instanceCount }: Props = $props();

  let expanded = $state(false);
  let expandedSaved: Record<string, boolean> = $state({});
  let checked: Record<string, boolean> = $state({});
  let launching: Record<string, boolean> = $state({});
  let savingSession = $state(false);
  let newSessionName = $state("");
  let showNameInput = $state(false);

  const data = $derived(sessions());
  const lastEntries = $derived(data.lastSession?.entries ?? []);
  const savedSessions = $derived(data.sessions ?? []);

  function entryKey(sessionId: string, modelId: string): string {
    return `${sessionId}:${modelId}`;
  }

  function isLaunching(sessionId: string, modelId: string): boolean {
    return launching[entryKey(sessionId, modelId)] === true;
  }

  function isChecked(sessionId: string, modelId: string): boolean {
    const key = entryKey(sessionId, modelId);
    return checked[key] !== false;
  }

  function toggleCheck(sessionId: string, modelId: string): void {
    const key = entryKey(sessionId, modelId);
    checked[key] = !isChecked(sessionId, modelId);
  }

  async function launchOne(
    sessionId: string,
    entry: SessionEntry,
  ): Promise<void> {
    const key = entryKey(sessionId, entry.model_id);
    launching[key] = true;
    try {
      const result =
        sessionId === "last"
          ? await launchLastSessionModel(entry.model_id)
          : await launchSessionModel(sessionId, entry.model_id);
      if (result.action === "already_running") {
        addToast({ type: "info", message: `${entry.model_id} is already running` });
      } else {
        addToast({ type: "success", message: `Launching ${entry.model_id}…` });
      }
    } catch (err) {
      addToast({
        type: "error",
        message: `Failed to launch ${entry.model_id}: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      launching[key] = false;
    }
  }

  async function launchChecked(
    sessionId: string,
    entries: SessionEntry[],
  ): Promise<void> {
    // Launch one by one, sequentially — user watches each model come up.
    for (const entry of entries) {
      if (!isChecked(sessionId, entry.model_id)) continue;
      await launchOne(sessionId, entry);
    }
  }

  async function handleSaveSession(): Promise<void> {
    const name = newSessionName.trim();
    if (!name) return;
    savingSession = true;
    try {
      await saveSession(name);
      addToast({ type: "success", message: `Session "${name}" saved` });
      newSessionName = "";
      showNameInput = false;
    } catch (err) {
      addToast({
        type: "error",
        message: `Failed to save session: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      savingSession = false;
    }
  }

  async function handleDeleteSession(session: SavedSession): Promise<void> {
    try {
      await deleteSession(session.id);
      addToast({ type: "info", message: `Session "${session.name}" deleted` });
    } catch (err) {
      addToast({
        type: "error",
        message: `Failed to delete: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  }

  function fmtIdle(seconds: number | null): string {
    return seconds != null ? `${seconds}s idle` : "no idle-unload";
  }

  function fmtDate(ts: number): string {
    return new Date(ts * 1000).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  }
</script>

<div class="border-b border-exo-medium-gray/30">
  <!-- Header row -->
  <button
    class="w-full p-4 pb-3 text-left cursor-pointer hover:bg-exo-medium-gray/10 transition-colors"
    onclick={() => (expanded = !expanded)}
    aria-expanded={expanded}
  >
    <div class="flex items-center justify-between">
      <div
        class="text-xs text-exo-yellow tracking-[0.2em] uppercase flex items-center gap-2"
      >
        SESSIONS
      </div>
      <span class="text-xs text-white/50">
        {lastEntries.length} last · {savedSessions.length} saved
        <span class="ml-1">{expanded ? "▲" : "▼"}</span>
      </span>
    </div>
  </button>

  {#if expanded}
    <div class="px-4 pb-4 space-y-4">
      <!-- LAST SESSION -->
      <div>
        <div class="flex items-center justify-between mb-2">
          <div class="text-[10px] font-mono text-white/40 uppercase tracking-widest">
            Last session
          </div>
          {#if lastEntries.length > 0}
            <button
              class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 border border-exo-yellow/40 text-exo-yellow hover:bg-exo-yellow/20 transition-all cursor-pointer"
              onclick={() => launchChecked("last", lastEntries)}
              disabled={lastEntries.some((e) => isLaunching("last", e.model_id))}
            >
              Launch {lastEntries.filter((e) => isChecked("last", e.model_id)).length}
            </button>
          {/if}
        </div>
        {#if lastEntries.length === 0}
          <div class="text-xs text-white/40 font-mono">No instances yet</div>
        {:else}
          <div class="space-y-1.5">
            {#each lastEntries as entry (entry.model_id)}
              {@const key = entryKey("last", entry.model_id)}
              <div
                class="flex items-center gap-2 p-2 rounded bg-exo-medium-gray/20 border border-exo-medium-gray/30"
              >
                <input
                  type="checkbox"
                  class="cursor-pointer accent-yellow-500"
                  checked={isChecked("last", entry.model_id)}
                  onchange={() => toggleCheck("last", entry.model_id)}
                  aria-label="Select {entry.model_id} for launch"
                />
                <div class="flex-1 min-w-0">
                  <div
                    class="text-xs text-white/80 font-mono truncate"
                    title={entry.model_id}
                  >
                    {entry.model_id}
                  </div>
                  <div class="text-[10px] text-white/40 font-mono">
                    {entry.sharding} · {entry.instance_meta === "MlxRing"
                      ? "MLX Ring"
                      : "MLX RDMA"} · {entry.min_nodes} node{entry.min_nodes === 1
                      ? ""
                      : "s"} · {fmtIdle(entry.idle_unload_seconds)}
                  </div>
                </div>
                <button
                  class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 border border-white/20 text-white/70 hover:border-exo-yellow/50 hover:text-exo-yellow transition-all cursor-pointer disabled:opacity-40"
                  onclick={() => launchOne("last", entry)}
                  disabled={isLaunching("last", entry.model_id)}
                  title="Launch this model with its saved settings"
                >
                  {isLaunching("last", entry.model_id) ? "…" : "Launch"}
                </button>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- SAVE CURRENT -->
      <div>
        {#if showNameInput}
          <div class="flex items-center gap-2">
            <input
              class="flex-1 bg-exo-dark-gray border border-exo-medium-gray/40 rounded px-2 py-1 text-xs font-mono text-white/80 focus:border-exo-yellow/50 outline-none"
              placeholder="Session name…"
              bind:value={newSessionName}
              onkeydown={(e) => {
                if (e.key === "Enter") handleSaveSession();
                if (e.key === "Escape") (showNameInput = false);
              }}
            />
            <button
              class="text-[10px] font-mono uppercase px-2 py-1 border border-exo-yellow/40 text-exo-yellow hover:bg-exo-yellow/20 cursor-pointer disabled:opacity-40"
              onclick={handleSaveSession}
              disabled={savingSession || !newSessionName.trim()}
            >
              {savingSession ? "…" : "Save"}
            </button>
          </div>
        {:else}
          <button
            class="text-[10px] font-mono uppercase tracking-wider text-white/50 hover:text-exo-yellow transition-colors cursor-pointer"
            onclick={() => (showNameInput = true)}
            disabled={instanceCount === 0}
            title={instanceCount === 0
              ? "No instances running to save"
              : "Save current instances as a named session"}
          >
            + Save current as session
          </button>
        {/if}
      </div>

      <!-- SAVED SESSIONS -->
      {#if savedSessions.length > 0}
        <div>
          <div class="text-[10px] font-mono text-white/40 uppercase tracking-widest mb-2">
            Saved sessions
          </div>
          <div class="space-y-2">
            {#each savedSessions as session (session.id)}
              {@const isOpen = expandedSaved[session.id] === true}
              <div
                class="rounded bg-exo-medium-gray/15 border border-exo-medium-gray/30"
              >
                <div class="flex items-center gap-2 p-2">
                  <button
                    class="flex-1 text-left cursor-pointer min-w-0"
                    onclick={() =>
                      (expandedSaved[session.id] = !isOpen)}
                  >
                    <div class="text-xs text-white/80 font-mono truncate">
                      {session.name}
                      <span class="text-white/30">({session.entries.length})</span>
                    </div>
                    <div class="text-[10px] text-white/30 font-mono">
                      {fmtDate(session.created_at)}
                    </div>
                  </button>
                  {#if isOpen}
                    <button
                      class="text-[10px] font-mono uppercase px-2 py-1 border border-white/20 text-white/50 hover:border-red-500/50 hover:text-red-400 transition-all cursor-pointer"
                      onclick={() => handleDeleteSession(session)}
                      title="Delete this saved session"
                    >
                      Del
                    </button>
                  {/if}
                  <button
                    class="text-[10px] font-mono uppercase px-2 py-1 border border-white/20 text-white/50 hover:text-white/80 transition-all cursor-pointer"
                    onclick={() => (expandedSaved[session.id] = !isOpen)}
                  >
                    {isOpen ? "▲" : "▼"}
                  </button>
                </div>
                {#if isOpen}
                  <div class="px-2 pb-2 space-y-1.5">
                    <div class="flex justify-end">
                      <button
                        class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 border border-exo-yellow/40 text-exo-yellow hover:bg-exo-yellow/20 transition-all cursor-pointer"
                        onclick={() => launchChecked(session.id, session.entries)}
                        disabled={session.entries.some((e) =>
                          isLaunching(session.id, e.model_id),
                        )}
                      >
                        Launch
                      </button>
                    </div>
                    {#each session.entries as entry (entry.model_id)}
                      <div
                        class="flex items-center gap-2 p-2 rounded bg-exo-dark-gray/60 border border-exo-medium-gray/20"
                      >
                        <input
                          type="checkbox"
                          class="cursor-pointer accent-yellow-500"
                          checked={isChecked(session.id, entry.model_id)}
                          onchange={() => toggleCheck(session.id, entry.model_id)}
                          aria-label="Select {entry.model_id} for launch"
                        />
                        <div class="flex-1 min-w-0">
                          <div
                            class="text-xs text-white/80 font-mono truncate"
                            title={entry.model_id}
                          >
                            {entry.model_id}
                          </div>
                          <div class="text-[10px] text-white/40 font-mono">
                            {entry.sharding} · {entry.instance_meta === "MlxRing"
                              ? "MLX Ring"
                              : "MLX RDMA"} · {entry.min_nodes} node{entry.min_nodes === 1
                              ? ""
                              : "s"} · {fmtIdle(entry.idle_unload_seconds)}
                          </div>
                        </div>
                        <button
                          class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 border border-white/20 text-white/70 hover:border-exo-yellow/50 hover:text-exo-yellow transition-all cursor-pointer disabled:opacity-40"
                          onclick={() => launchOne(session.id, entry)}
                          disabled={isLaunching(session.id, entry.model_id)}
                        >
                          {isLaunching(session.id, entry.model_id) ? "…" : "Launch"}
                        </button>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>
