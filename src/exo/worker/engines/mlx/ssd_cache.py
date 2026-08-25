# ruff: noqa
"""Cold (SSD) tier for the KV prefix cache (oMLX doc 01, Phases 2–3).

The hot tier (``KVPrefixCache``) is RAM-only and lost on every restart. For
agentic/coding workloads (Claude Code, Pi, Codex) the same long system prompt
is re-sent across requests; recomputing that prefill every restart — or every
time LRU evicts it — is the dominant latency and the explicit reason oMLX
exists.

This module is the SSD spill/restore + restart-recovery layer that turns the
already-shipped tiered-cache **settings** (``turboquant.is_tiered_cache_enabled``
/ ``hot_cache_only`` / ``ssd_cache_dir`` / ``ssd_cache_max_size_bytes``) into a
working cold tier. It is a pure adjunct to the existing entry-based
``KVPrefixCache`` (not a paged-block rewrite): when an entry is evicted from
RAM it is serialized to SSD; when a future prefix lookup misses RAM but hits
SSD, it is restored instead of recomputed — even after a process restart.

Serialization uses mlx-lm's ``save_prompt_cache`` / ``load_prompt_cache``
(safetensors, with per-layer ``state`` / ``meta_state`` / class names), so the
round-trip is byte-exact for the standard cache classes (``KVCache``,
``QuantizedKVCache``, ``RotatingKVCache``, ``ArraysCache``, ``CacheList``).
Cache classes that mlx-lm cannot reconstruct (anything not in
``mlx_lm.models.cache`` globals and without ``from_state`` — e.g.
``DeepseekV4Cache``) are **SSD-ineligible**: an entry containing one is never
spilled, degrading gracefully to today's RAM-only behaviour. This mirrors
oMLX's ``type_handlers`` block-slice-eligibility concept — not every cache
class is SSD-eligible.

A **cache signature** (model id + per-layer cache class names + quant
bits/group_size) is stored in the safetensors metadata and checked on restore,
so a stale SSD block left by a different model / quant config is refused
rather than restoring incompatible state (the design doc's
"cache-signature-for / _cache_compat_signature" guard).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
from mlx.utils import tree_unflatten

from exo.worker.engines.mlx import turboquant
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS

if TYPE_CHECKING:
    from exo.shared.models.model_cards import ModelId
    from exo.worker.engines.mlx.types import KVCacheType

logger = logging.getLogger(__name__)

# Cache classes mlx-lm's ``load_prompt_cache`` can reconstruct (it looks them
# up in ``mlx_lm.models.cache`` globals via ``globals()[class_name]``). Only
# entries whose every layer is one of these are SSD-eligible; an exotic layer
# (e.g. ``DeepseekV4Cache``) makes the whole entry ineligible and it degrades
# to today's RAM-only behaviour. Computed once at import.
import mlx_lm.models.cache as _mlx_cache_module

_SSD_ELIGIBLE_CACHE_CLASSES: frozenset[str] = frozenset(
    name
    for name in (
        "KVCache",
        "QuantizedKVCache",
        "RotatingKVCache",
        "ArraysCache",
        "CacheList",
    )
    if name in _mlx_cache_module.__dict__
)

# Metadata keys written into the safetensors file (mlx-lm stores the caller's
# metadata dict alongside its own cache-info; these are the keys we add).
_META_SIGNATURE = "exo_cache_signature"
_META_TOKENS = "exo_token_count"
_META_PROMPT_HASH = "exo_prompt_hash"
_META_MODEL_ID = "exo_model_id"
_META_SAVED_AT = "exo_saved_at_epoch"

# Suffix for the per-entry safetensors file. oMLX stores blocks under
# hash-prefix subdirs; EXO stores one file per evicted entry under a
# 2-level hash-prefix dir to keep any single directory's entry count bounded
# (mirrors oMLX's hash-prefix layout and keeps ``rglob`` scans cheap).
_ENTRY_SUFFIX = ".safetensors"
# Sidecar holding the entry's prompt token ids (int32 ``.npy``), written
# alongside the safetensors file so the in-RAM index can do longest-common-
# prefix matching for prefix-SSD restore without touching the (large) KV
# file. The token array is small relative to the KV (50k tokens ≈ 200 KB vs
# a multi-GB KV file) and scales with entry size, so the SSD size cap bounds
# the total index memory.
_TOKENS_SUFFIX = ".tokens.npy"

# Cache classes whose state ``trim(n)`` cannot safely roll back. A prefix
# restore of such an entry would need a snapshot (which SSD restore doesn't
# have), so partial prefix restore is only offered for entries with none of
# these. Exact match (no trim) is still offered for every SSD-eligible entry.
# Mirrors ``cache.is_non_trimmable_cache_entry`` but checked by class name to
# keep this module decoupled from ``cache.py`` (no circular import).
_NON_TRIMMABLE_BY_NAME: frozenset[str] = frozenset(
    {"ArraysCache", "RotatingKVCache", "DeepseekV4Cache"}
)


def _entry_is_ssd_eligible(cache: "KVCacheType") -> bool:
    """Whether every layer of ``cache`` can be serialized + reconstructed.

    ``save_prompt_cache`` serializes whatever ``state``/``meta_state`` a layer
    exposes, but ``load_prompt_cache`` reconstructs via
    ``globals()[class_name].from_state`` — so a layer whose class is not in
    ``mlx_lm.models.cache`` (or lacks ``from_state``) cannot be restored. We
    refuse to spill such an entry so it degrades to RAM-only instead of
    writing a file we could never correctly load back.
    """
    return all(
        type(layer).__name__ in _SSD_ELIGIBLE_CACHE_CLASSES
        and hasattr(type(layer), "from_state")
        for layer in cache
    )


def _has_non_trimmable_layers(cache: "KVCacheType") -> bool:
    """Whether ``cache`` has any layer a prefix restore can't trim back.

    A partial prefix restore trims the restored cache down to the common
    prefix length; layers whose state ``trim(n)`` can't roll back
    (``ArraysCache``/``RotatingKVCache``/``DeepseekV4Cache``/non-trimmable
    ``CacheList``) need a snapshot for that, which SSD restore doesn't carry.
    So a partial prefix restore is only offered when this returns False.
    Exact-match restore (no trim) is offered for every SSD-eligible entry.
    """
    for layer in cache:
        name = type(layer).__name__
        if name in _NON_TRIMMABLE_BY_NAME:
            return True
        if name == "CacheList":
            try:
                if not bool(layer.is_trimmable()):  # type: ignore[reportUnknownMemberType]
                    return True
            except Exception:  # pragma: no cover - defensive
                return True
    return False


def _common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two int token arrays.

    Vectorised: ``argmax`` of the first mismatch gives its index (0 when the
    very first token differs); if no token differs in the overlap, the whole
    overlap is the prefix. O(min(len(a), len(b))) per entry, but the compare
    is a single vectorised numpy op so it's fast for 50k-token prompts.
    """
    n = min(int(len(a)), int(len(b)))
    if n == 0:
        return 0
    # Element-wise inequality; cast because ndarray comparison stubs are Any.
    diff = cast("np.ndarray", a[:n] != b[:n])
    if not bool(diff.any()):
        return n
    return int(np.argmax(diff))


def _cache_signature(cache: "KVCacheType", model_id: "ModelId | None") -> str:
    """A string fingerprint guarding against model/quant-config swaps on restore.

    Covers: model id, per-layer cache class names, and the effective KV
    quantization (TurboQuant bits/group-size, falling back to the legacy
    ``KV_CACHE_BITS`` global). Two entries with different signatures must not
    restore into each other — e.g. a 4-bit TurboQuant block must not be loaded
    under an fp16 model, and a GLM-5.2 hybrid cache must not be loaded under a
    Qwen model. The signature is stored in the safetensors metadata and
    re-checked on restore.
    """
    layer_classes = [type(layer).__name__ for layer in cache]
    # Per-layer quant params (for QuantizedKVCache). Non-quant layers contribute
    # nothing, so the signature is stable for plain KVCache regardless of the
    # global bits setting.
    layer_quant: list[str] = []
    for layer in cache:
        bits = getattr(layer, "bits", None)
        group_size = getattr(layer, "group_size", None)
        if bits is not None or group_size is not None:
            layer_quant.append(f"{bits}/{group_size}")
    tq_bits = turboquant.effective_kv_bits()
    effective_bits = tq_bits if tq_bits is not None else KV_CACHE_BITS
    payload = {
        "model_id": str(model_id) if model_id is not None else "",
        "layers": layer_classes,
        "layer_quant": layer_quant,
        "kv_bits": effective_bits,
        "kv_group_size": CACHE_GROUP_SIZE,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _prompt_hash(prompt_tokens: mx.array) -> str:
    """SHA-256 of the prompt token ids (hex, truncated for use as a key)."""
    try:
        data = np.array(prompt_tokens).tobytes()
    except Exception:  # pragma: no cover - defensive: never block eviction
        return ""
    return hashlib.sha256(data).hexdigest()[:32]


def _hash_prefix_dirs(digest: str) -> Path:
    """2-level hash-prefix layout (mirrors oMLX) to bound directory entry count."""
    return Path(digest[0]) / digest[1:3]


def _tokens_sidecar(entry_path: Path) -> Path:
    """Path of the ``.tokens.npy`` sidecar for a given entry safetensors file."""
    return entry_path.with_suffix(_TOKENS_SUFFIX)


@dataclass(frozen=True)
class SSDEntryIndex:
    """In-RAM index record for one SSD-cached entry.

    The full prompt **token ids** are held in RAM (as a compact int32 array) so
    the longest-common-prefix lookup for prefix-SSD restore can run without
    touching the (large) KV file — the token array is ~200 KB for a 50k-token
    prompt vs a multi-GB KV file, and it scales with entry size so the SSD size
    cap bounds total index memory. Also stored: the hash (cheap exact-membership
    key), the signature (model/quant-swap guard), the file paths, and the LRU
    access epoch. ``prompt_tokens`` is ``None`` only when an entry was recovered
    from disk but its sidecar was missing/unreadable (degrades that entry to
    exact-match-only).
    """

    prompt_hash: str
    token_count: int
    signature: str
    model_id: str
    file_path: Path
    file_size_bytes: int
    last_access_epoch: int
    prompt_tokens: np.ndarray | None = None


@dataclass
class SSDKVCacheStore:
    """Cold-tier store: spill, restore, restart-recovery scan, LRU size cap.

    All operations are no-ops when the tiered cache is disabled
    (``turboquant.is_tiered_cache_enabled()`` is False) or ``hot_cache_only``
    is set, so the existing RAM-only behaviour is untouched unless an operator
    explicitly enables the SSD tier. Per-node (not shared across the cluster);
    cross-node sharing is the disaggregated-prefill path's job.
    """

    ssd_dir: Path
    max_size_bytes: int
    # Monotonic access counter so LRU eviction on the SSD tier is O(n) in the
    # index (the index is small: one record per spilled entry, not per block).
    _access_counter: int = 0
    _index: dict[str, SSDEntryIndex] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.is_active():
            self._ensure_dir()
            self._recover_from_disk()

    def is_active(self) -> bool:
        """Whether the SSD tier is enabled (master switch on, not hot-only)."""
        return turboquant.is_tiered_cache_enabled() and not turboquant.hot_cache_only()

    # ─── spill (RAM eviction → SSD) ──────────────────────────────────────────
    def spill(
        self,
        prompt_tokens: mx.array,
        cache: "KVCacheType",
        *,
        model_id: "ModelId | None",
    ) -> bool:
        """Serialize an evicted RAM entry to SSD. Returns False if ineligible.

        Called from ``KVPrefixCache`` when an entry is about to be dropped from
        RAM. If the entry is SSD-eligible and the tier is active, it is written
        to a hash-prefix-subdir safetensors file and indexed. Best-effort: a
        write failure logs and returns False (the entry is simply lost, as it
        would be today without an SSD tier).
        """
        if not self.is_active():
            return False
        if not _entry_is_ssd_eligible(cache):
            return False
        digest = _prompt_hash(prompt_tokens)
        if not digest:
            return False
        signature = _cache_signature(cache, model_id)
        rel = _hash_prefix_dirs(digest) / f"{digest}{_ENTRY_SUFFIX}"
        path = self.ssd_dir / rel
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Restrictive perms: KV state can leak prompt content (design doc,
            # security note). best-effort on non-POSIX.
            with _suppress_os():
                os.chmod(self.ssd_dir, 0o700)
            save_prompt_cache(
                str(path),
                list(cache),  # type: ignore[arg-type]
                metadata={
                    _META_SIGNATURE: signature,
                    _META_TOKENS: str(int(len(prompt_tokens))),
                    _META_PROMPT_HASH: digest,
                    _META_MODEL_ID: str(model_id) if model_id is not None else "",
                    _META_SAVED_AT: str(int(time.time())),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive: never block eviction
            logger.warning("SSD KV spill failed for %s: %s", digest[:12], exc)
            return False
        size = path.stat().st_size
        # Write the prompt-tokens sidecar so the in-RAM index can do longest-
        # common-prefix matching for prefix-SSD restore. Best-effort: a write
        # failure leaves the entry exact-match-only (``prompt_tokens=None``).
        tokens_path = _tokens_sidecar(path)
        tokens_arr: np.ndarray | None = None
        try:
            tokens_arr = np.asarray(np.array(prompt_tokens), dtype=np.int32)
            np.save(tokens_path, tokens_arr)
            with _suppress_os():
                os.chmod(tokens_path, 0o600)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning(
                "SSD KV tokens sidecar write failed for %s: %s", digest[:12], exc
            )
            tokens_arr = None
        size += tokens_path.stat().st_size if tokens_arr is not None else 0
        self._access_counter += 1
        self._index[digest] = SSDEntryIndex(
            prompt_hash=digest,
            token_count=int(len(prompt_tokens)),
            signature=signature,
            model_id=str(model_id) if model_id is not None else "",
            file_path=path,
            file_size_bytes=size,
            last_access_epoch=self._access_counter,
            prompt_tokens=tokens_arr,
        )
        self._enforce_size_cap()
        logger.info(
            "SSD KV spill: %d tokens → %s (%d bytes; SSD tier %d/%d bytes, %d entries)",
            int(len(prompt_tokens)),
            path,
            size,
            self._total_size(),
            self.max_size_bytes,
            len(self._index),
        )
        return True

    # ─── restore (SSD → RAM on prefix hit) ───────────────────────────────────
    def restore(
        self, prompt_tokens: mx.array, *, model_id: "ModelId | None"
    ) -> "tuple[KVCacheType | None, int]":
        """Restore an SSD entry matching ``prompt_tokens`` exactly.

        Returns ``(cache, token_count)`` or ``(None, 0)`` when no exact-match
        SSD entry exists, the tier is inactive, or the signature mismatches
        (stale block from a different model/quant config). Exact-match only:
        a *prefix* restore is a Phase-2+ refinement (oMLX restores the longest
        common prefix across ranks); EXO's prefix match is still served from
        RAM, and SSD is the restart/eviction-recovery path that avoids the
        full re-prefill.
        """
        if not self.is_active():
            return None, 0
        digest = _prompt_hash(prompt_tokens)
        if not digest:
            return None, 0
        record = self._index.get(digest)
        if record is None:
            return None, 0
        try:
            loaded, meta = load_prompt_cache(
                str(record.file_path), return_metadata=True
            )
        except Exception as exc:  # pragma: no cover - corrupt file
            logger.warning(
                "SSD KV restore failed for %s: %s — removing stale file",
                digest[:12],
                exc,
            )
            self._remove(digest)
            return None, 0
        # ``load_prompt_cache`` returns ``list[Any]`` (mlx-lm is untyped); the
        # signature guard below re-validates the structure via the per-layer
        # class names, so the cast is sound.
        cache = cast("KVCacheType", cast(object, loaded))
        # Signature guard: refuse a block saved under a different model/quant
        # config rather than restoring incompatible state.
        stored_sig = _meta_get(meta, _META_SIGNATURE)
        if stored_sig and stored_sig != _cache_signature(cache, model_id):
            logger.info(
                "SSD KV signature mismatch for %s — refusing restore (model/quant "
                "swap); removing stale block",
                digest[:12],
            )
            self._remove(digest)
            return None, 0
        self._access_counter += 1
        record = replace(record, last_access_epoch=self._access_counter)
        self._index[digest] = record
        # Re-new the file's mtime so an LRU-atime-based recovery scan keeps it.
        with _suppress_os():
            os.utime(record.file_path, None)
        logger.info(
            "SSD KV restore: %d tokens ← %s (SSD tier %d/%d bytes, %d entries)",
            record.token_count,
            record.file_path,
            self._total_size(),
            self.max_size_bytes,
            len(self._index),
        )
        return cache, record.token_count

    def has(self, prompt_tokens: mx.array) -> bool:
        """Cheap (no-disk) exact-match membership check, for the lookup path."""
        if not self.is_active():
            return False
        digest = _prompt_hash(prompt_tokens)
        return digest in self._index

    # ─── prefix restore (SSD → RAM on partial prefix hit) ─────────────────────
    def restore_prefix(
        self, prompt_tokens: mx.array, *, model_id: "ModelId | None"
    ) -> "tuple[KVCacheType | None, int]":
        """Restore the longest common prefix held on SSD (oMLX doc 01 finish).

        The dominant agentic workload is re-sending the same long system
        prompt plus a few new turns. After a restart/LRU eviction the RAM tier
        is empty, so without this the whole prompt is re-prefilled. This finds
        the SSD entry whose prompt shares the longest common prefix with the
        query, loads + signature-validates its cache, and returns it along with
        the matched prefix length so the caller trims it to the prefix and only
        prefills the suffix.

        Returns ``(cache, prefix_len)`` with the cache at the entry's **full**
        offset (the caller trims) and ``prefix_len`` the common prefix length;
        ``(None, 0)`` when the tier is inactive, no entry has a common prefix,
        or the only partial match is on a non-trimmable entry (which can't be
        trimmed back to the prefix without a snapshot). Exact matches
        (``prefix_len == entry length``) are offered for every SSD-eligible
        entry regardless of trimmability (no trim needed).
        """
        if not self.is_active() or not self._index:
            return None, 0
        try:
            query = np.asarray(np.array(prompt_tokens), dtype=np.int32)
        except Exception:  # pragma: no cover - defensive
            return None, 0
        if int(query.size) == 0:
            return None, 0

        # Find the SSD entry with the longest common prefix (RAM-only compare
        # over the indexed token arrays — no disk I/O).
        best_digest: str | None = None
        best_len = 0
        for digest, record in self._index.items():
            entry_tokens = record.prompt_tokens
            if entry_tokens is None or int(entry_tokens.size) == 0:
                # Sidecar missing (legacy/recovery gap) → only exact match via
                # the hash is possible for this entry; skip prefix scan.
                continue
            common = _common_prefix_len(query, entry_tokens)
            if common > best_len:
                best_len = common
                best_digest = digest

        # Fall back to exact hash match when no indexed tokens produced a hit
        # (e.g. all sidecars missing) — preserves the Phase-2 exact-restore
        # behaviour for entries recovered without their sidecar.
        if best_digest is None:
            exact_digest = _prompt_hash(prompt_tokens)
            if exact_digest and exact_digest in self._index:
                best_digest = exact_digest
                best_len = self._index[exact_digest].token_count

        if best_digest is None or best_len <= 0:
            return None, 0
        record = self._index[best_digest]

        try:
            loaded, meta = load_prompt_cache(
                str(record.file_path), return_metadata=True
            )
        except Exception as exc:  # pragma: no cover - corrupt file
            logger.warning(
                "SSD KV prefix restore failed for %s: %s — removing stale file",
                best_digest[:12],
                exc,
            )
            self._remove(best_digest)
            return None, 0
        cache = cast("KVCacheType", cast(object, loaded))
        stored_sig = _meta_get(meta, _META_SIGNATURE)
        if stored_sig and stored_sig != _cache_signature(cache, model_id):
            logger.info(
                "SSD KV signature mismatch for %s — refusing prefix restore "
                "(model/quant swap); removing stale block",
                best_digest[:12],
            )
            self._remove(best_digest)
            return None, 0

        # A partial prefix restore trims the cache back to the prefix; entries
        # with non-trimmable layers can't be trimmed without a snapshot, so a
        # partial match on one is refused (exact match is fine — no trim).
        if best_len < record.token_count and _has_non_trimmable_layers(cache):
            logger.info(
                "SSD KV prefix restore: best match %d/%d tokens but entry has "
                "non-trimmable layers — refusing partial restore (no snapshot).",
                best_len,
                record.token_count,
            )
            return None, 0

        self._access_counter += 1
        record = replace(record, last_access_epoch=self._access_counter)
        self._index[best_digest] = record
        with _suppress_os():
            os.utime(record.file_path, None)
        logger.info(
            "SSD KV prefix restore: %d/%d tokens ← %s (SSD tier %d/%d bytes, "
            "%d entries)",
            best_len,
            record.token_count,
            record.file_path,
            self._total_size(),
            self.max_size_bytes,
            len(self._index),
        )
        return cache, best_len

    # ─── restart recovery ────────────────────────────────────────────────────
    def _recover_from_disk(self) -> None:
        """Scan the SSD dir on startup and rebuild the in-RAM index (Phase 3).

        Mirrors oMLX's startup scan + ``recovery.py``: walk the hash-prefix
        subdirs, read each entry's metadata, and index it. Files missing the
        EXO metadata keys (written by an older/different build) are removed —
        they cannot be signature-validated. The LRU access order is seeded from
        file mtime so recently-used entries survive the size cap.
        """
        if not self.ssd_dir.exists():
            return
        found = 0
        # Sort by mtime ascending so the access counter (and thus LRU order)
        # tracks recency.
        entries: list[tuple[float, Path]] = []
        for path in self.ssd_dir.rglob(f"*{_ENTRY_SUFFIX}"):
            if not path.is_file():
                continue
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                continue
        for _mtime, path in sorted(entries):
            try:
                _arrays, meta = _load_metadata_only(str(path))
            except Exception as exc:  # pragma: no cover - corrupt file
                logger.warning("SSD KV recovery: skipping unreadable %s: %s", path, exc)
                _unlink_quiet(path)
                continue
            digest = _meta_get(meta, _META_PROMPT_HASH)
            if not digest or _meta_get(meta, _META_SIGNATURE) is None:
                logger.info("SSD KV recovery: removing unkeyed/legacy file %s", path)
                _unlink_quiet(path)
                continue
            try:
                size = path.stat().st_size
            except OSError:
                _unlink_quiet(path)
                continue
            # Load the prompt-tokens sidecar so prefix-SSD restore works for
            # recovered entries too. Missing/unreadable sidecar → exact-match
            # only (``prompt_tokens=None``).
            tokens_path = _tokens_sidecar(path)
            tokens_arr: np.ndarray | None = None
            if tokens_path.is_file():
                try:
                    tokens_arr = np.asarray(
                        np.load(tokens_path, allow_pickle=False),
                        dtype=np.int32,
                    )
                except OSError as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "SSD KV recovery: unreadable tokens sidecar %s: %s",
                        tokens_path,
                        exc,
                    )
                    _unlink_quiet(tokens_path)
            if tokens_arr is not None:
                size += tokens_path.stat().st_size
            self._access_counter += 1
            self._index[digest] = SSDEntryIndex(
                prompt_hash=digest,
                token_count=int(_meta_get(meta, _META_TOKENS) or 0),
                signature=_meta_get(meta, _META_SIGNATURE) or "",
                model_id=_meta_get(meta, _META_MODEL_ID) or "",
                file_path=path,
                file_size_bytes=size,
                last_access_epoch=self._access_counter,
                prompt_tokens=tokens_arr,
            )
            found += 1
        if found:
            logger.info(
                "SSD KV recovery: indexed %d entr(y/ies) from %s", found, self.ssd_dir
            )
        self._enforce_size_cap()

    # ─── LRU size cap ─────────────────────────────────────────────────────────
    def _enforce_size_cap(self) -> None:
        """Evict least-recently-used SSD entries until under the size cap.

        oMLX's ``PagedSSDCacheIndex.evict_until_size`` analogue. An entry is
        removed by deleting its file and dropping its index record. No-op when
        the cap is 0 (disabled) or the tier is already under cap.
        """
        if self.max_size_bytes <= 0:
            return
        while self._total_size() > self.max_size_bytes and self._index:
            # Least-recently-used = smallest access epoch.
            digest, record = min(
                self._index.items(), key=lambda kv: kv[1].last_access_epoch
            )
            logger.info(
                "SSD KV LRU evict: %d tokens (%d bytes) — tier over cap %d/%d",
                record.token_count,
                record.file_size_bytes,
                self._total_size(),
                self.max_size_bytes,
            )
            self._remove(digest)

    def _total_size(self) -> int:
        return sum(r.file_size_bytes for r in self._index.values())

    def _remove(self, digest: str) -> None:
        record = self._index.pop(digest, None)
        if record is None:
            return
        _unlink_quiet(record.file_path)
        _unlink_quiet(_tokens_sidecar(record.file_path))

    def _ensure_dir(self) -> None:
        try:
            self.ssd_dir.mkdir(parents=True, exist_ok=True)
            with _suppress_os():
                os.chmod(self.ssd_dir, 0o700)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("SSD KV dir create failed %s: %s", self.ssd_dir, exc)

    # ─── observability (feeds the dashboard gauge) ────────────────────────────
    def status(self) -> dict[str, int]:
        """Live SSD-tier block/file counts + size for the dashboard gauge."""
        return {
            "ssd_entries": len(self._index),
            "ssd_size_bytes": self._total_size(),
            "ssd_max_size_bytes": self.max_size_bytes,
        }

    def clear(self) -> int:
        """Delete every SSD-cached entry; return the count removed."""
        removed = len(self._index)
        for digest in list(self._index):
            self._remove(digest)
        # Also sweep any orphaned files not in the index (e.g. from a crash
        # mid-spill before indexing).
        if self.ssd_dir.exists():
            for path in self.ssd_dir.rglob(f"*{_ENTRY_SUFFIX}"):
                if path.is_file():
                    _unlink_quiet(path)
            for path in self.ssd_dir.rglob(f"*{_TOKENS_SUFFIX}"):
                if path.is_file():
                    _unlink_quiet(path)
        return removed


# ─── helpers ──────────────────────────────────────────────────────────────────
def _meta_get(meta: object, key: str) -> "str | None":
    """Read a string metadata value from mlx-lm's metadata dict (typed)."""
    if not isinstance(meta, dict):
        return None
    typed_meta: dict[str, object] = cast("dict[str, object]", meta)
    value = typed_meta.get(key)
    if value is None:
        return None
    return str(value)


def _load_metadata_only(path: str) -> tuple[object, dict[str, str]]:
    """Load a safetensors file's arrays + **user** metadata (recovery scan).

    ``mx.load(path, return_metadata=True)`` returns the raw flattened metadata
    (keys like ``1.exo_prompt_hash`` because mlx-lm's ``save_prompt_cache``
    flattens the nested ``[info, user_metadata, classes]`` via ``tree_flatten``).
    We ``tree_unflatten`` it back and return the user-metadata subtree (element
    1 of the 3-element list) so recovery reads ``exo_prompt_hash`` /
    ``exo_cache_signature`` directly -- mirroring what
    ``load_prompt_cache(return_metadata=True)`` hands the restore path. The
    arrays are discarded (recovery only indexes metadata).
    """
    arrays, raw_meta = mx.load(path, return_metadata=True)
    raw_items: list[tuple[str, object]] = list(raw_meta.items())  # type: ignore[reportAny,reportUnknownMemberType]
    unflattened: object = tree_unflatten(raw_items)  # type: ignore[reportAny]
    # mlx-lm flattens [cache_info, user_metadata, classes]; ``tree_unflatten``
    # returns that 3-element list, with the user metadata at index 1.
    user_meta: dict[str, str] = {}
    if isinstance(unflattened, list) and len(unflattened) > 1:  # type: ignore[reportUnknownArgumentType]
        candidate: object = unflattened[1]  # type: ignore[reportUnknownMemberType]
        if isinstance(candidate, dict):
            user_meta = {str(k): str(v) for k, v in candidate.items()}  # type: ignore[reportAny,reportUnknownMemberType]
    return arrays, user_meta


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("SSD KV unlink failed %s: %s", path, exc)


class _suppress_os:
    """``contextlib.suppress(OSError)`` inline, to keep the import list tight."""

    def __enter__(self) -> "_suppress_os":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return exc_type is OSError
