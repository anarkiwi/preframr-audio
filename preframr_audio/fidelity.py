"""Shared audio-equivalence helper for macro-pass validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PAL_FRAME_HZ = 50.12

FRAME_RMS_TOLERANCE = 0.05


@dataclasses.dataclass(frozen=True)
class AudioFidelityResult:
    """Outcome of comparing two rendered streams."""

    passed: bool
    shape: str
    diagnostic: str
    lag_samples: int = 0
    worst_frame_rel_rms: float = 0.0
    n_frames_compared: int = 0


_PASS = AudioFidelityResult(passed=True, shape="PASS", diagnostic="")


def _frame_window(sample_rate: int) -> int:
    """Samples per PAL frame at the given output rate."""
    return max(1, int(round(sample_rate / PAL_FRAME_HZ)))


def _estimate_lag(a: np.ndarray, b: np.ndarray, search_radius: int = 2048) -> int:
    """Cross-correlate a windowed slice from the middle of both streams
    to find the integer sample lag at which they best align. Lag 0 ->
    sample-aligned; non-zero -> a timing shift signature."""
    n = min(len(a), len(b))
    if n < 2 * search_radius + 16:
        return 0
    window = min(8192, n // 4)
    mid = n // 2
    chunk_a = a[mid : mid + window].astype(np.float64)
    chunk_a -= chunk_a.mean()
    base = b[mid - search_radius : mid + window + search_radius].astype(np.float64)
    base -= base.mean()
    if chunk_a.std() < 1e-9 or base.std() < 1e-9:
        return 0
    corr = np.correlate(base, chunk_a, mode="valid")
    return int(corr.argmax()) - search_radius


def _per_frame_rms(
    samples_a: np.ndarray, samples_b: np.ndarray, frame_window: int
) -> np.ndarray:
    """Per-frame RMS divergence between aligned streams. Frame size =
    ``frame_window`` samples (one PAL frame)."""
    n = min(len(samples_a), len(samples_b))
    a = samples_a[:n].astype(np.float64)
    b = samples_b[:n].astype(np.float64)
    n_frames = n // frame_window
    if n_frames == 0:
        return np.array([])
    a = a[: n_frames * frame_window].reshape(n_frames, frame_window)
    b = b[: n_frames * frame_window].reshape(n_frames, frame_window)
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def compare_renders(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    sample_rate: int,
    tolerance: float = FRAME_RMS_TOLERANCE,
) -> AudioFidelityResult:
    """Compare two int16 sample streams; return an
    ``AudioFidelityResult``. Pure-numpy, no I/O.
    """
    if samples_a.shape != samples_b.shape:
        return AudioFidelityResult(
            passed=False,
            shape="DURATION_MISMATCH",
            diagnostic=(
                f"DURATION MISMATCH: {samples_a.shape} vs {samples_b.shape} "
                f"(catastrophic cadence break: stream lengths differ)"
            ),
        )

    fw = _frame_window(sample_rate)
    lag = _estimate_lag(samples_a, samples_b)
    if abs(lag) >= fw:
        return AudioFidelityResult(
            passed=False,
            shape="FRAME_CADENCE_BREAK",
            diagnostic=(
                f"FRAME CADENCE BREAK: cross-corr lag = {lag} samples "
                f"(~{lag / fw:+.1f} frames at {sample_rate} Hz). Likely a "
                f"producer/consumer race in audio_driver.AudioRenderBuffer."
            ),
            lag_samples=lag,
        )

    rms_per_frame = _per_frame_rms(samples_a, samples_b, fw)
    if rms_per_frame.size == 0:
        return AudioFidelityResult(
            passed=True,
            shape="NO_INFO",
            diagnostic=(
                "NO INFO: input streams shorter than one PAL frame "
                f"({fw} samples at {sample_rate} Hz); cannot compute "
                "per-frame RMS. Treating as pass; consider longer inputs "
                "if this is a real comparison."
            ),
            n_frames_compared=0,
        )

    peak = max(1.0, float(max(np.abs(samples_a).max(), np.abs(samples_b).max())))
    rel = rms_per_frame / peak
    worst = int(rel.argmax())
    worst_rel = float(rel[worst])

    n = len(rel)
    if worst_rel <= tolerance:
        return dataclasses.replace(
            _PASS,
            worst_frame_rel_rms=worst_rel,
            n_frames_compared=n,
            lag_samples=lag,
        )

    head_n = max(1, n // 20)
    tail_n = max(1, n // 20)
    head = float(rel[:head_n].mean())
    tail = float(rel[-tail_n:].mean())
    overall = float(rms_per_frame.mean())

    if head > 5 * max(tail, 1e-9):
        shape = "INITIAL_STATE_DIVERGENCE"
        msg = (
            f"INITIAL-STATE DIVERGENCE: head-RMS {head:.1%} >> tail-RMS "
            f"{tail:.1%}. Likely resid SID state leaking between "
            f"render_to_wav calls (worker doesn't fully reset)."
        )
    elif tail > 3 * max(head, 1e-9):
        shape = "DRIFTING_DIVERGENCE"
        msg = (
            f"DRIFTING DIVERGENCE: head-RMS {head:.1%}, tail-RMS "
            f"{tail:.1%}. Divergence grows over the stream -- per-frame "
            f"timing accumulates."
        )
    else:
        shape = "CONSTANT_DIVERGENCE"
        msg = (
            f"CONSTANT DIVERGENCE: head-RMS {head:.1%} ~ tail-RMS "
            f"{tail:.1%} ~ {overall:.1f} RMS. Stream-wide difference "
            f"with no obvious cadence shape -- typical 'decoder picked "
            f"the wrong slot' signature."
        )
    diagnostic = (
        f"{msg} Worst frame: {worst} of {n} (rel-RMS {worst_rel:.1%}, "
        f"tolerance {tolerance:.0%}). Lag at sub-frame: {lag} samples."
    )
    return AudioFidelityResult(
        passed=False,
        shape=shape,
        diagnostic=diagnostic,
        lag_samples=lag,
        worst_frame_rel_rms=worst_rel,
        n_frames_compared=n,
    )


def render_df_to_wav(df, irq: int, args, wav_path: Path) -> Tuple[int, "object"]:
    """Render ``df`` to a WAV file via the production audio path.
    Returns ``(n_samples_written, df_audio)`` so callers can assert on
    the prepared df shape if needed.
    """
    from preframr_audio.audio_driver import render_to_wav
    from preframr_tokens.reglogparser import prepare_df_for_audio
    from preframr_audio.sidwav import sidq

    df_audio, reg_widths = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    n = render_to_wav(
        df_audio, str(wav_path), reg_widths=reg_widths, irq=irq, cents=args.cents
    )
    return n, df_audio


def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    """Returns (sample_rate, samples_int16). Imported lazily to keep
    the unit-test path off scipy."""
    from scipy.io import wavfile

    sr, samples = wavfile.read(str(path))
    return int(sr), samples


_IRQ_MISSING_SENTINEL = -1


def _irq_from_df(df) -> int:
    """Pull the IRQ rate from the first FRAME row's diff column. Raises if no FRAME rows present (audio pipeline needs the PAL/NTSC cadence to be explicit, not defaulted)."""
    from preframr_tokens.reglog_helpers import (
        read_initial_irq,
    )  # pylint: disable=import-outside-toplevel

    irq = read_initial_irq(df, default=_IRQ_MISSING_SENTINEL)
    if irq == _IRQ_MISSING_SENTINEL:
        raise ValueError(
            "df has no FRAME rows; cannot determine IRQ. The audio "
            "pipeline needs at least one frame marker to know the PAL/NTSC "
            "cadence."
        )
    return irq


def assert_dfs_render_equivalent(
    df_a,
    df_b,
    args,
    tmp_path: Path,
    label_a: str = "a",
    label_b: str = "b",
    tolerance: float = FRAME_RMS_TOLERANCE,
) -> Optional[AudioFidelityResult]:
    """Render both dfs to disk WAVs, read back, compare. Raise
    AssertionError with the diagnostic on fail; return the result on
    pass (so the caller can inspect ``worst_frame_rel_rms`` if it
    cares).
    """
    irq_a = _irq_from_df(df_a)
    irq_b = _irq_from_df(df_b)
    wav_a = tmp_path / f"{label_a}.wav"
    wav_b = tmp_path / f"{label_b}.wav"
    n_a, _ = render_df_to_wav(df_a, irq_a, args, wav_a)
    n_b, _ = render_df_to_wav(df_b, irq_b, args, wav_b)
    assert n_a == n_b, (
        f"sample counts diverge: {label_a}={n_a} vs {label_b}={n_b}. The "
        "render path stages disagree on stream length before per-frame "
        "comparison; check FRAME_REG counts or IRQ divergence "
        f"({label_a} irq={irq_a}, {label_b} irq={irq_b})."
    )
    sr_a, samples_a = read_wav(wav_a)
    sr_b, samples_b = read_wav(wav_b)
    assert sr_a == sr_b, f"sample-rate divergence: {label_a}={sr_a} vs {label_b}={sr_b}"
    result = compare_renders(samples_a, samples_b, sr_a, tolerance=tolerance)
    if not result.passed:
        raise AssertionError(
            f"audio fidelity check {label_a} vs {label_b} failed: "
            f"{result.diagnostic}"
        )
    return result
