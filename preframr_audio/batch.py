"""Parallel batch rendering + render-equivalence over many parsed dfs, for
offline augmentation prebakes (e.g. the inaudible-perturbation and
voice-permutation/melody-transfer families baked on a many-core host). Mirrors
``fingerprint.fingerprint_batch``: a ``spawn`` process pool, since pyresidfp is
not thread-safe and each worker owns its SID instance. ``n_workers=1`` runs
sequentially in the calling process (debugging / when pool overhead dominates).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Iterable, List, Tuple

import numpy as np

from preframr_audio.fidelity import (
    FRAME_RMS_TOLERANCE,
    AudioFidelityResult,
    _render_df_samples,
    dfs_render_equivalent,
)

_BATCH_KWARGS: dict = {}


def _batch_init(kwargs: dict) -> None:
    global _BATCH_KWARGS
    _BATCH_KWARGS = kwargs


def _render_worker(item):
    idx, df = item
    return idx, _render_df_samples(df, **_BATCH_KWARGS)


def _verify_worker(item):
    idx, (df_a, df_b) = item
    return idx, dfs_render_equivalent(df_a, df_b, **_BATCH_KWARGS)


def _run(items, worker, init_kwargs, n_workers, chunksize):
    """Map ``worker`` over ``(idx, payload)`` items, sequentially when
    ``n_workers==1`` else across a spawn pool, and reorder results by idx."""
    if n_workers == 1:
        _batch_init(init_kwargs)
        results = [worker(item) for item in items]
    else:
        if n_workers == -1:
            n_workers = mp.cpu_count()
        with mp.get_context("spawn").Pool(
            n_workers, initializer=_batch_init, initargs=(init_kwargs,)
        ) as pool:
            results = pool.map(worker, items, chunksize=chunksize)
    out: list = [None] * len(items)
    for idx, val in results:
        out[idx] = val
    return out


def render_batch(
    dfs: Iterable,
    *,
    n_workers: int = -1,
    chunksize: int = 1,
    cents: int = 50,
    chip_model: str = "MOS8580",
) -> List[Tuple[np.ndarray, int]]:
    """Render many parsed dfs to ``[(samples_int16, sample_rate)]`` in parallel.
    Order matches the input. Use for the plausibility-gate families (melody
    transfer / voice permutation) where the caller extracts features from each
    render. ``n_workers=-1`` uses ``os.cpu_count()``.
    """
    seqs = list(dfs)
    if not seqs:
        return []
    items = list(enumerate(seqs))
    return _run(
        items,
        _render_worker,
        {"cents": cents, "chip_model": chip_model},
        n_workers,
        chunksize,
    )


def verify_equivalent_batch(
    pairs: Iterable[Tuple[object, object]],
    *,
    n_workers: int = -1,
    chunksize: int = 1,
    cents: int = 50,
    chip_model: str = "MOS8580",
    tolerance: float = FRAME_RMS_TOLERANCE,
    max_frame_drift: int = 1,
) -> List[AudioFidelityResult]:
    """Render-equivalence gate over many ``(df_orig, df_perturbed)`` pairs in
    parallel -- the inaudible-perturbation augmentation accept/reject step.
    Each worker renders both dfs and compares in-process, returning only the
    small ``AudioFidelityResult`` (no sample arrays cross the process boundary).
    Order matches the input.
    """
    seqs = list(pairs)
    if not seqs:
        return []
    items = list(enumerate(seqs))
    return _run(
        items,
        _verify_worker,
        {
            "cents": cents,
            "chip_model": chip_model,
            "tolerance": tolerance,
            "max_frame_drift": max_frame_drift,
        },
        n_workers,
        chunksize,
    )
