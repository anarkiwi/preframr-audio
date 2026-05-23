"""Audio fingerprint framework: render register-write sequences via pyresidfp and extract a fixed-length feature vector. Reusable primitives for content-token clustering, register-write equivalence search, generation-quality audits, and any future workload that needs ``sequence_of_writes -> acoustic_fingerprint``. Parallelism via ``multiprocessing.Pool`` (pyresidfp is not thread-safe; processes own their SID instances)."""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from preframr_audio._sid_constants import (
    FRAME_REG,
    MODE_VOL_REG,
    VOICE_CTRL_REG,
)
from preframr_audio.audio_driver import render_to_samples
from preframr_audio.features import FEATURE_FNS

DEFAULT_IRQ_PAL = 19656
DEFAULT_DURATION_FRAMES = 5
DEFAULT_CENTS = 50
DEFAULT_CHIP_MODEL = "MOS8580"
_PAL_CLOCK_HZ = 985248

WriteTriple = Tuple[int, int, int]
WritesArg = Union[Sequence[WriteTriple], pd.DataFrame]
FeatureFn = Callable[[np.ndarray, int], np.ndarray]


_DEFAULT_VOICE_SETUP = {
    0: {
        "freq_lo": 0x42,
        "freq_hi": 0x10,
        "pw_lo": 0x00,
        "pw_hi": 0x08,
        "ctrl": 0x11,
        "ad": 0x88,
        "sr": 0xCC,
    },
    1: {
        "freq_lo": 0x53,
        "freq_hi": 0x14,
        "pw_lo": 0x00,
        "pw_hi": 0x08,
        "ctrl": 0x11,
        "ad": 0x88,
        "sr": 0xCC,
    },
    2: {
        "freq_lo": 0x90,
        "freq_hi": 0x18,
        "pw_lo": 0x00,
        "pw_hi": 0x08,
        "ctrl": 0x11,
        "ad": 0x88,
        "sr": 0xCC,
    },
}


def _voice_reg(voice: int, offset: int) -> int:
    return voice * 7 + offset


def _default_scaffold_writes() -> list[WriteTriple]:
    """Three voices in a playing triad + master volume on. clock=0 for all (the renderer applies them at frame init)."""
    writes: list[WriteTriple] = [(0, MODE_VOL_REG, 15)]
    for voice, regs in _DEFAULT_VOICE_SETUP.items():
        writes.append((0, _voice_reg(voice, 0), regs["freq_lo"]))
        writes.append((0, _voice_reg(voice, 1), regs["freq_hi"]))
        writes.append((0, _voice_reg(voice, 2), regs["pw_lo"]))
        writes.append((0, _voice_reg(voice, 3), regs["pw_hi"]))
        writes.append((0, _voice_reg(voice, 5), regs["ad"]))
        writes.append((0, _voice_reg(voice, 6), regs["sr"]))
        writes.append((0, VOICE_CTRL_REG[voice], regs["ctrl"]))
    return writes


def canonical_scaffold(
    test_writes: Sequence[WriteTriple],
    irq: int = DEFAULT_IRQ_PAL,
    pre_frames: int = 1,
    post_frames: int = 1,
    scaffold_writes: Optional[Sequence[WriteTriple]] = None,
) -> pd.DataFrame:
    """Wrap ``test_writes`` in a canonical audible context: scaffold writes + ``pre_frames`` of silence + test writes + ``post_frames`` of silence. Returns a DataFrame in the schema ``render_to_samples`` expects (cols: reg, val, delay)."""
    if scaffold_writes is None:
        scaffold_writes = _default_scaffold_writes()
    frame_delay_s = irq / _PAL_CLOCK_HZ
    rows: list[dict] = []
    for _clock, reg, val in scaffold_writes:
        rows.append({"reg": int(reg), "val": int(val), "delay": 0.0})
    rows.append({"reg": FRAME_REG, "val": 0, "delay": frame_delay_s})
    for _ in range(pre_frames):
        rows.append({"reg": FRAME_REG, "val": 0, "delay": frame_delay_s})
    for _clock, reg, val in test_writes:
        rows.append({"reg": int(reg), "val": int(val), "delay": 0.0})
    rows.append({"reg": FRAME_REG, "val": 0, "delay": frame_delay_s})
    for _ in range(post_frames):
        rows.append({"reg": FRAME_REG, "val": 0, "delay": frame_delay_s})
    df = pd.DataFrame(rows)
    df.attrs["irq"] = int(irq)
    return df


def _normalise_writes(writes: WritesArg) -> list[WriteTriple]:
    if isinstance(writes, pd.DataFrame):
        out: list[WriteTriple] = []
        for row in writes.itertuples(index=False):
            clock = int(getattr(row, "clock", 0) or 0)
            out.append((clock, int(row.reg), int(row.val)))
        return out
    return [(int(c), int(r), int(v)) for c, r, v in writes]


def _resolve_feature_fn(feature: Union[str, FeatureFn]) -> FeatureFn:
    if callable(feature):
        return feature
    if feature in FEATURE_FNS:
        return FEATURE_FNS[feature]
    raise ValueError(
        f"unknown feature {feature!r}; pass a callable or one of "
        f"{sorted(FEATURE_FNS.keys())}"
    )


def fingerprint_writes(
    test_writes: WritesArg,
    irq: int = DEFAULT_IRQ_PAL,
    *,
    feature: Union[str, FeatureFn] = "mel",
    cents: int = DEFAULT_CENTS,
    chip_model: str = DEFAULT_CHIP_MODEL,
    pre_frames: int = 1,
    post_frames: int = DEFAULT_DURATION_FRAMES,
    scaffold_writes: Optional[Sequence[WriteTriple]] = None,
    reg_widths: Optional[dict] = None,
    **feature_kwargs,
) -> np.ndarray:
    """Single render: wrap ``test_writes`` in canonical scaffold, render via pyresidfp, extract feature vector. ``feature`` is either a string key from ``FEATURE_FNS`` (mel, spectral, raw_pcm) or a custom callable ``(samples, sr) -> np.ndarray``."""
    feature_fn = _resolve_feature_fn(feature)
    triples = _normalise_writes(test_writes)
    df = canonical_scaffold(
        triples,
        irq=irq,
        pre_frames=pre_frames,
        post_frames=post_frames,
        scaffold_writes=scaffold_writes,
    )
    samples, sample_rate = render_to_samples(
        df,
        reg_widths=reg_widths or {},
        irq=irq,
        cents=cents,
        chip_model=chip_model,
    )
    return feature_fn(samples, sample_rate, **feature_kwargs)


_BATCH_KWARGS: dict = {}


def _batch_worker(item: Tuple[int, WritesArg]) -> Tuple[int, np.ndarray]:
    idx, writes = item
    feat = fingerprint_writes(writes, **_BATCH_KWARGS)
    return idx, feat


def _batch_init(kwargs: dict) -> None:
    global _BATCH_KWARGS
    _BATCH_KWARGS = kwargs


def fingerprint_batch(
    sequences: Iterable[WritesArg],
    *,
    n_workers: int = -1,
    chunksize: int = 1,
    feature: Union[str, FeatureFn] = "mel",
    **kwargs,
) -> np.ndarray:
    """Parallel rendering + fingerprinting across many write sequences. Returns ``(N, feature_dim)`` stacked. ``n_workers=-1`` uses ``os.cpu_count()``; ``n_workers=1`` runs sequentially in the calling process (useful for debugging or when the feature fn is cheap and the multiprocessing overhead dominates)."""
    seqs = list(sequences)
    if not seqs:
        return np.zeros((0, 0), dtype=np.float64)
    full_kwargs = dict(kwargs)
    full_kwargs["feature"] = feature
    if n_workers == 1:
        feats = []
        for w in seqs:
            feats.append(fingerprint_writes(w, **full_kwargs))
        return np.stack(feats)
    if n_workers == -1:
        n_workers = mp.cpu_count()
    items = list(enumerate(seqs))
    with mp.get_context("spawn").Pool(
        n_workers, initializer=_batch_init, initargs=(full_kwargs,)
    ) as pool:
        results = pool.map(_batch_worker, items, chunksize=chunksize)
    feats = [None] * len(seqs)
    for idx, feat in results:
        feats[idx] = feat
    return np.stack(feats)
