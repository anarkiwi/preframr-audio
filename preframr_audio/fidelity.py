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
    max_frame_drift: int = 1,
) -> AudioFidelityResult:
    """Compare two int16 sample streams; return an
    ``AudioFidelityResult``. Pure-numpy, no I/O. ``max_frame_drift`` is
    the cross-correlation lag, in PAL frames, tolerated before a
    FRAME_CADENCE_BREAK is declared (default 1 == sample-aligned).
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
    if abs(lag) >= max(1, max_frame_drift) * fw:
        return AudioFidelityResult(
            passed=False,
            shape="FRAME_CADENCE_BREAK",
            diagnostic=(
                f"FRAME CADENCE BREAK: cross-corr lag = {lag} samples "
                f"(~{lag / fw:+.1f} frames at {sample_rate} Hz, tolerance "
                f"{max_frame_drift} frames). Likely a producer/consumer race "
                f"in audio_driver.AudioRenderBuffer."
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


def per_frame_rel_rms(
    samples_a,
    samples_b,
    sample_rate: int,
) -> np.ndarray:
    """Per-PAL-frame relative-RMS difference between two int16 streams, as a
    1-D array (one value per frame, as a fraction of peak amplitude). Unlike
    ``compare_renders`` — which returns only the worst frame — this exposes the
    whole timeline so callers can pinpoint exactly which frames diverge (e.g.
    ``np.where(per_frame_rel_rms(...) > tolerance)`` and multiply by the frame
    period for timestamps). Streams are truncated to the shorter length; DC and
    other common-mode content cancels in the per-frame difference.
    """
    a = np.asarray(samples_a)
    b = np.asarray(samples_b)
    n = min(len(a), len(b))
    rms = _per_frame_rms(a[:n], b[:n], _frame_window(sample_rate))
    if rms.size == 0:
        return rms
    peak = max(1.0, float(max(np.abs(a[:n]).max(), np.abs(b[:n]).max())))
    return rms / peak


def compare_renders_per_voice(
    samples_a,
    samples_b,
    sample_rate: int,
    tolerance: float = FRAME_RMS_TOLERANCE,
    max_frame_drift: int = 1,
):
    """Run ``compare_renders`` per voice over two ``{voice: samples}`` maps
    (e.g. from ``audio_driver.render_per_voice``). Soloed-voice comparison
    surfaces divergence the full mix masks (a quiet percussion or bass voice
    is not drowned by a loud lead). Streams are truncated to the shorter
    length per voice. Returns ``{voice: AudioFidelityResult}``.
    """
    results = {}
    for voice in sorted(set(samples_a) & set(samples_b)):
        a = np.asarray(samples_a[voice])
        b = np.asarray(samples_b[voice])
        n = min(len(a), len(b))
        results[voice] = compare_renders(
            a[:n],
            b[:n],
            sample_rate,
            tolerance=tolerance,
            max_frame_drift=max_frame_drift,
        )
    return results


def render_df_to_wav(df, irq: int, args, wav_path: Path) -> Tuple[int, "object"]:
    """Render ``df`` to a WAV file via the production audio path.
    Returns ``(n_samples_written, df_audio)`` so callers can assert on
    the prepared df shape if needed.
    """
    from preframr_tokens.reglogparser import prepare_df_for_audio

    from preframr_audio.audio_driver import render_to_wav
    from preframr_audio.sidwav import sidq

    df_audio, reg_widths = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    n = render_to_wav(
        df_audio, str(wav_path), reg_widths=reg_widths, irq=irq, cents=args.cents
    )
    return n, df_audio


def _render_df_samples(df, *, cents: int = 50, chip_model: str = "MOS8580"):
    """Render a parsed df straight to ``(samples_int16, sample_rate)`` in memory
    -- ``prepare_df_for_audio`` then ``render_to_samples``, no WAV/scipy round
    trip. Shared by ``dfs_render_equivalent`` and the batch helpers."""
    from preframr_tokens.reglogparser import prepare_df_for_audio

    from preframr_audio.audio_driver import render_to_samples
    from preframr_audio.sidwav import sidq

    irq = _irq_from_df(df)
    df_audio, reg_widths = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    return render_to_samples(
        df_audio, reg_widths=reg_widths, irq=irq, cents=cents, chip_model=chip_model
    )


def dfs_render_equivalent(
    df_a,
    df_b,
    *,
    cents: int = 50,
    chip_model: str = "MOS8580",
    tolerance: float = FRAME_RMS_TOLERANCE,
    max_frame_drift: int = 1,
) -> AudioFidelityResult:
    """In-memory render-equivalence gate for the inaudible-perturbation
    augmentation family: render both parsed dfs to samples and compare with no
    WAV/scipy round trip (the ``assert_dfs_render_equivalent`` disk path is for
    test assertions). Returns the ``AudioFidelityResult`` -- ``.passed`` is the
    accept/reject signal; non-passing results carry the divergence ``shape`` +
    ``diagnostic``.
    """
    samples_a, sr_a = _render_df_samples(df_a, cents=cents, chip_model=chip_model)
    samples_b, sr_b = _render_df_samples(df_b, cents=cents, chip_model=chip_model)
    if sr_a != sr_b:
        return AudioFidelityResult(
            passed=False,
            shape="SAMPLE_RATE_MISMATCH",
            diagnostic=f"sample-rate divergence: a={sr_a} vs b={sr_b}",
        )
    return compare_renders(
        samples_a,
        samples_b,
        sr_a,
        tolerance=tolerance,
        max_frame_drift=max_frame_drift,
    )


# --------------------------------------------------------------------------- #
# Perceptual render-equivalence gate.
#
# ``compare_renders`` above is a strict per-frame-RMS oracle: it answers "is B a
# bit-for-bit-ish re-render of A?". Some downstream renderings are deliberately
# lossy -- the register stream is re-expressed with small, intentional pitch
# quantization (cents-binned frequency, preset/transpose rounding). A ~50-cent
# pitch shift decorrelates the waveform over time, so it fails strict RMS even
# though it sounds identical. The perceptual gate below is the lossy sibling:
# "does B sound like A?". It is windowed (worst-window, like the worst-frame
# design of ``compare_renders``) so a localized wrong note is not diluted.
#
# Metric design note: the perceptual distance is built on a *coarse linear
# band-power* spectral envelope (``features.band_power_features``) plus the
# classical ``features.spectral_features``. ``features.mel_features`` was
# evaluated and rejected for this gate: mel spacing allocates narrow bins to low
# frequencies, so a sub-semitone detune of a bass-heavy SID spectrum moves its
# harmonics across bins and reads as large as a real timbre change -- it fails
# the headline "≤50-cent shift is perceptually equivalent" property. Summing a
# sharp spectrum into wide linear bands makes a within-band detune invisible
# while keeping timbre/filter/voicing changes audible (calibrated >2x class
# separation; see ``calibrate``).
# --------------------------------------------------------------------------- #

# Calibrated by ``calibrate()`` (Deliverable 2): geometric mean of the INERT
# distance ceiling (~0.152) and the BAD distance floor (~0.437) over the
# synthetic calibration set -- a ~2.9x class-separation margin. Re-run
# ``python -m preframr_audio.fidelity`` (or ``fidelity.calibrate()``) to reproduce.
PERCEPTUAL_DISTANCE_THRESHOLD = 0.257

# Weight of the (scaled) spectral-descriptor family relative to the band-power
# envelope in the per-window distance. The band-power envelope carries the
# class separation; the spectral term is a complementary brightness/rolloff
# check. Kept low enough that the calibrated margin stays comfortably above 2x.
_PERCEPTUAL_SPECTRAL_WEIGHT = 4.0


@dataclasses.dataclass(frozen=True)
class PerceptualFidelityResult:
    """Outcome of a perceptual render-equivalence check (the lossy-rendering
    sibling of ``AudioFidelityResult``)."""

    passed: bool
    distance: float
    threshold: float
    worst_window_s: float
    n_windows: int
    shape: str
    diagnostic: str


def _scaled_spectral(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """``features.spectral_features`` with the three Hz-valued descriptors
    (centroid, bandwidth, rolloff) divided by Nyquist so all five dims sit in a
    comparable ~[0, 1] range (zcr and rms already do)."""
    from preframr_audio.features import spectral_features

    sp = spectral_features(samples, sample_rate).copy()
    nyq = max(1.0, sample_rate / 2.0)
    sp[:3] /= nyq
    return sp


def perceptual_distance(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    sample_rate: int,
    *,
    win_s: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """Windowed perceptual distance between two int16 streams. Pure-numpy, no
    I/O. Streams are truncated to the shorter length; each ~``win_s`` window
    yields one distance and the worst (max) is returned alongside the full
    per-window array. Per window the distance combines a coarse linear
    band-power log-envelope (``features.band_power_features``, the pitch-tolerant
    workhorse) with the scaled classical ``features.spectral_features``.
    Deterministic; reuses ``features.py`` (no STFT/mel reimplementation)."""
    from preframr_audio.features import band_power_features

    a = np.asarray(samples_a)
    b = np.asarray(samples_b)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    win = max(1, int(round(win_s * sample_rate)))
    n_windows = max(1, n // win)
    per = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        sl = slice(i * win, (i + 1) * win)
        aw = a[sl]
        bw = b[sl]
        band = band_power_features(aw, sample_rate) - band_power_features(
            bw, sample_rate
        )
        d_band_sq = float(np.mean(band**2))
        spec = _scaled_spectral(aw, sample_rate) - _scaled_spectral(bw, sample_rate)
        d_spec_sq = float(np.mean(spec**2))
        per[i] = np.sqrt(d_band_sq + _PERCEPTUAL_SPECTRAL_WEIGHT * d_spec_sq)
    return float(per.max()), per


def _perceptual_result(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    sample_rate: int,
    *,
    threshold: float,
    win_s: float,
) -> PerceptualFidelityResult:
    """Build a ``PerceptualFidelityResult`` from two streams, handling the
    duration-mismatch / too-short-input cases the same way ``compare_renders``
    does (truncate-to-shorter for the distance itself; flag a gross length
    divergence; NO_INFO pass when shorter than one window)."""
    a = np.asarray(samples_a)
    b = np.asarray(samples_b)
    win = max(1, int(round(win_s * sample_rate)))
    n = min(len(a), len(b))
    if n < win:
        return PerceptualFidelityResult(
            passed=True,
            distance=0.0,
            threshold=threshold,
            worst_window_s=0.0,
            n_windows=0,
            shape="NO_INFO",
            diagnostic=(
                "NO INFO: input streams shorter than one perceptual window "
                f"({win} samples / {win_s}s at {sample_rate} Hz); cannot compute "
                "a windowed perceptual distance. Treating as pass; consider "
                "longer inputs if this is a real comparison."
            ),
        )
    if abs(len(a) - len(b)) >= win:
        return PerceptualFidelityResult(
            passed=False,
            distance=0.0,
            threshold=threshold,
            worst_window_s=0.0,
            n_windows=0,
            shape="DURATION_MISMATCH",
            diagnostic=(
                f"DURATION MISMATCH: {len(a)} vs {len(b)} samples (differ by "
                f">= one {win_s}s window); streams are not the same rendering."
            ),
        )
    worst, per = perceptual_distance(a, b, sample_rate, win_s=win_s)
    worst_idx = int(np.argmax(per))
    worst_window_s = worst_idx * win / float(sample_rate)
    passed = bool(worst <= threshold)
    if passed:
        return PerceptualFidelityResult(
            passed=True,
            distance=worst,
            threshold=threshold,
            worst_window_s=worst_window_s,
            n_windows=len(per),
            shape="PASS",
            diagnostic="",
        )
    return PerceptualFidelityResult(
        passed=False,
        distance=worst,
        threshold=threshold,
        worst_window_s=worst_window_s,
        n_windows=len(per),
        shape="PERCEPTUAL_DIVERGENCE",
        diagnostic=(
            f"PERCEPTUAL DIVERGENCE: worst-window distance {worst:.3f} > "
            f"threshold {threshold:.3f} at t={worst_window_s:.2f}s "
            f"(window {worst_idx + 1} of {len(per)}). Render B does not sound "
            f"perceptually equivalent to render A in that window."
        ),
    )


def dfs_perceptually_equivalent(
    df_a,
    df_b,
    *,
    threshold: float = PERCEPTUAL_DISTANCE_THRESHOLD,
    cents: int = 50,
    chip_model: str = "MOS8580",
    win_s: float = 1.0,
) -> PerceptualFidelityResult:
    """In-memory perceptual render-equivalence gate: render both parsed dfs to
    samples (same ``_render_df_samples`` path ``dfs_render_equivalent`` uses),
    then compute the windowed ``perceptual_distance``. ``.passed`` is True when
    the worst-window distance is within ``threshold`` -- i.e. B sounds like A
    even if it is not a strict per-frame re-render. The perceptual sibling of
    ``dfs_render_equivalent``."""
    samples_a, sr_a = _render_df_samples(df_a, cents=cents, chip_model=chip_model)
    samples_b, sr_b = _render_df_samples(df_b, cents=cents, chip_model=chip_model)
    if sr_a != sr_b:
        return PerceptualFidelityResult(
            passed=False,
            distance=0.0,
            threshold=threshold,
            worst_window_s=0.0,
            n_windows=0,
            shape="SAMPLE_RATE_MISMATCH",
            diagnostic=f"sample-rate divergence: a={sr_a} vs b={sr_b}",
        )
    return _perceptual_result(
        samples_a, samples_b, sr_a, threshold=threshold, win_s=win_s
    )


def assert_dfs_perceptually_equivalent(
    df_a,
    df_b,
    *,
    threshold: float = PERCEPTUAL_DISTANCE_THRESHOLD,
    cents: int = 50,
    chip_model: str = "MOS8580",
    win_s: float = 1.0,
    label_a: str = "a",
    label_b: str = "b",
) -> PerceptualFidelityResult:
    """Render both dfs and assert perceptual equivalence. Raise
    ``AssertionError`` with the diagnostic on fail; return the result on pass
    (so the caller can inspect ``distance`` / ``worst_window_s``). Mirrors
    ``assert_dfs_render_equivalent``."""
    result = dfs_perceptually_equivalent(
        df_a,
        df_b,
        threshold=threshold,
        cents=cents,
        chip_model=chip_model,
        win_s=win_s,
    )
    if not result.passed:
        raise AssertionError(
            f"perceptual fidelity check {label_a} vs {label_b} failed: "
            f"{result.diagnostic}"
        )
    return result


def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    """Returns (sample_rate, samples_int16). Imported lazily to keep
    the unit-test path off scipy."""
    from scipy.io import wavfile

    sr, samples = wavfile.read(str(path))
    return int(sr), samples


_IRQ_MISSING_SENTINEL = -1


# mirrors preframr_tokens.stfconstants.FRAME_REG; replicated to avoid an
# audio<->tokens import cycle
_FRAME_REG = -128


def _read_initial_irq(df, default: int) -> int:
    """First FRAME-row positive ``diff`` = the PAL/NTSC frame period in cycles.
    Local copy of preframr_tokens.reglogparser.read_initial_irq -- trivial, kept
    here so the render path does not import the parser package (preframr_tokens
    imports preframr_audio.fidelity, so importing it back would be a cycle)."""
    positive = df.loc[df["reg"] == _FRAME_REG, "diff"]
    positive = positive[positive > 0]
    return int(positive.iloc[0]) if not positive.empty else int(default)


def _irq_from_df(df) -> int:
    """Pull the IRQ rate from the first FRAME row's diff column. Raises if no FRAME rows present (audio pipeline needs the PAL/NTSC cadence to be explicit, not defaulted)."""
    irq = _read_initial_irq(df, default=_IRQ_MISSING_SENTINEL)
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
    max_frame_drift: int = 1,
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
    result = compare_renders(
        samples_a, samples_b, sr_a, tolerance=tolerance, max_frame_drift=max_frame_drift
    )
    if not result.passed:
        raise AssertionError(
            f"audio fidelity check {label_a} vs {label_b} failed: "
            f"{result.diagnostic}"
        )
    return result


# --------------------------------------------------------------------------- #
# Calibration (Deliverable 2): the perceptual threshold is only meaningful if
# calibrated. The set below is built synthetically, in-repo, from
# ``fingerprint.canonical_scaffold`` plus deterministic df edits -- two classes:
#   INERT  (must read BELOW threshold -- perceptually equivalent)
#   BAD    (must read ABOVE threshold -- audibly different)
# ``calibrate()`` renders every pair, prints the INERT/BAD distance ranges, the
# chosen threshold (geometric mean of the INERT ceiling and BAD floor), and the
# class-separation margin, so the constant ``PERCEPTUAL_DISTANCE_THRESHOLD`` is
# reproducible. ``tests/test_perceptual_fidelity.py`` asserts the same set.
# --------------------------------------------------------------------------- #

# A spectrally-distinct sustained triad: voice 0 sawtooth (low, ~262 Hz),
# voice 1 pulse (mid), voice 2 triangle (high), all gated on, routed through a
# wide-open low-pass filter so a cutoff change is audible. Each voice occupies a
# distinct register/timbre so dropping or re-enveloping one moves the spectrum.
_CAL_NOTES = (120, 132, 144)
_CAL_WAVES = (0x21, 0x41, 0x11)  # saw, pulse, triangle (+ gate bit set below)
_CAL_PRE_FRAMES = 2
_CAL_POST_FRAMES = 150


def _cal_voice_reg(voice: int, offset: int) -> int:
    from preframr_audio._sid_constants import VOICE_REG_SIZE

    return voice * VOICE_REG_SIZE + offset


def _cal_baseline_writes(
    cutoff: int = 0xF800,
    waves: Tuple[int, ...] = _CAL_WAVES,
    notes: Tuple[int, ...] = _CAL_NOTES,
    sustains: Tuple[int, ...] = (0xF0, 0xF0, 0xF0),
) -> list:
    """(clock, reg, val) scaffold writes for the calibration triad. Frequency is
    written as a note index into reg 0/7/14 (expanded to 2 freq bytes by the
    renderer's ``FreqMapper``), so a note-index +1 is exactly one ``cents`` step
    (50 cents at the default quantization)."""
    from preframr_audio._sid_constants import MODE_VOL_REG, VOICE_CTRL_REG

    writes: list = []
    for v in range(3):
        base = _cal_voice_reg(v, 0)
        writes.append((0, base + 0, notes[v]))  # freq (note index)
        writes.append((0, base + 2, 0x00))  # pulse width lo (2-byte reg)
        writes.append((0, base + 3, 0x08))  # pulse width hi (50%)
        writes.append((0, base + 5, 0x00))  # attack/decay (fast)
        writes.append((0, base + 6, sustains[v]))  # sustain/release
        writes.append((0, VOICE_CTRL_REG[v], waves[v]))  # waveform + gate
    writes.append((0, 21, cutoff))  # filter cutoff (2-byte reg)
    writes.append((0, 23, 0xF7))  # resonance + route all voices to filter
    writes.append((0, MODE_VOL_REG, 0x1F))  # low-pass mode + volume
    return writes


def _cal_df(writes: list):
    from preframr_audio.fingerprint import canonical_scaffold

    return canonical_scaffold(
        [],
        pre_frames=_CAL_PRE_FRAMES,
        post_frames=_CAL_POST_FRAMES,
        scaffold_writes=writes,
    )


def _calibration_pairs() -> list:
    """Return ``[(label, kind, df_a, df_b)]`` where ``kind`` is ``"INERT"`` or
    ``"BAD"`` and ``df_a`` is always the unperturbed baseline. Pure df edits; no
    rendering here so callers can render once and reuse."""
    from preframr_audio._sid_constants import VOICE_CTRL_REG

    base = _cal_baseline_writes()

    def edit(reg_pred, new_val):
        return [(c, r, new_val if reg_pred(r) else v) for (c, r, v) in base]

    def freq_reg(voice):
        return _cal_voice_reg(voice, 0)

    base_df = _cal_df(base)
    pairs: list = []

    # INERT -------------------------------------------------------------------
    # reorder writes across voices (independent regs -> identical SID state)
    reorder = base[6:12] + base[0:6] + base[12:]
    pairs.append(("reorder_writes", "INERT", base_df, _cal_df(reorder)))
    # duplicate a same-value write (idempotent)
    pairs.append(("duplicate_write", "INERT", base_df, _cal_df(base + [base[-1]])))
    # nudge each voice's frequency by one note-index step (~50 cents)
    for v in range(3):
        nudged = [(c, r, (v_ + 1) if r == freq_reg(v) else v_) for (c, r, v_) in base]
        pairs.append((f"freq_nudge_50c_v{v}", "INERT", base_df, _cal_df(nudged)))

    # BAD ---------------------------------------------------------------------
    # octave shifts (note-index +/-24 == +/-1200 cents)
    up = [(c, r, (v_ + 24) if r == freq_reg(0) else v_) for (c, r, v_) in base]
    pairs.append(("octave_up_v0", "BAD", base_df, _cal_df(up)))
    down = [(c, r, (v_ - 24) if r == freq_reg(2) else v_) for (c, r, v_) in base]
    pairs.append(("octave_down_v2", "BAD", base_df, _cal_df(down)))
    # waveform changes (flip the control-register waveform bits, keep gate)
    pairs.append(
        (
            "waveform_saw_to_tri_v0",
            "BAD",
            base_df,
            _cal_df(edit(lambda r: r == VOICE_CTRL_REG[0], 0x11)),
        )
    )
    pairs.append(
        (
            "waveform_tri_to_pulse_v2",
            "BAD",
            base_df,
            _cal_df(edit(lambda r: r == VOICE_CTRL_REG[2], 0x41)),
        )
    )
    # substantial filter-cutoff change
    pairs.append(
        ("filter_cutoff", "BAD", base_df, _cal_df(_cal_baseline_writes(cutoff=0x5000)))
    )
    # envelope (sustain -> 0: each note attacks then decays to silence)
    pairs.append(
        (
            "envelope_sustain_v1",
            "BAD",
            base_df,
            _cal_df(_cal_baseline_writes(sustains=(0xF0, 0x00, 0xF0))),
        )
    )
    pairs.append(
        (
            "envelope_sustain_v2",
            "BAD",
            base_df,
            _cal_df(_cal_baseline_writes(sustains=(0xF0, 0xF0, 0x00))),
        )
    )
    # drop a voice (clear its control register -> no waveform, silent)
    for v in range(3):
        pairs.append(
            (
                f"drop_voice_v{v}",
                "BAD",
                base_df,
                _cal_df(edit(lambda r, _v=v: r == VOICE_CTRL_REG[_v], 0x00)),
            )
        )
    return pairs


def _render_calibration_df(df, *, cents: int = 50, chip_model: str = "MOS8580"):
    """Render a calibration (``canonical_scaffold``) df straight to samples via
    ``render_to_samples`` -- the scaffold schema (reg/val/delay) is what
    ``render_to_samples`` consumes directly, so this does not need the
    ``prepare_df_for_audio`` raw-df path of ``_render_df_samples``."""
    from preframr_audio.audio_driver import render_to_samples

    return render_to_samples(
        df,
        reg_widths={},
        irq=int(df.attrs["irq"]),
        cents=cents,
        chip_model=chip_model,
    )


def calibrate(*, win_s: float = 1.0, verbose: bool = True) -> dict:
    """Render the synthetic calibration set, compute each pair's worst-window
    ``perceptual_distance``, and report the INERT/BAD distance ranges, the chosen
    threshold (geometric mean of the INERT ceiling and BAD floor) and the
    class-separation margin. Returns a dict; prints a report when ``verbose``.
    This is the reproducer for ``PERCEPTUAL_DISTANCE_THRESHOLD``."""
    pairs = _calibration_pairs()
    cache: dict = {}

    def render(df):
        key = id(df)
        if key not in cache:
            cache[key] = _render_calibration_df(df)
        return cache[key]

    inert: dict = {}
    bad: dict = {}
    for label, kind, df_a, df_b in pairs:
        samples_a, sr = render(df_a)
        samples_b, _ = render(df_b)
        worst, _ = perceptual_distance(samples_a, samples_b, sr, win_s=win_s)
        (inert if kind == "INERT" else bad)[label] = worst

    inert_ceiling = max(inert.values())
    bad_floor = min(bad.values())
    threshold = float(np.sqrt(inert_ceiling * bad_floor))
    margin = bad_floor / inert_ceiling if inert_ceiling > 0 else float("inf")
    result = {
        "inert": inert,
        "bad": bad,
        "inert_ceiling": inert_ceiling,
        "bad_floor": bad_floor,
        "threshold": threshold,
        "margin": margin,
    }
    if verbose:
        print("INERT (must be BELOW threshold):")
        for label, d in sorted(inert.items(), key=lambda kv: -kv[1]):
            print(f"  {d:7.4f}  {label}")
        print("BAD (must be ABOVE threshold):")
        for label, d in sorted(bad.items(), key=lambda kv: kv[1]):
            print(f"  {d:7.4f}  {label}")
        print(
            f"INERT ceiling = {inert_ceiling:.4f}   BAD floor = {bad_floor:.4f}\n"
            f"chosen threshold (geomean) = {threshold:.4f}   "
            f"separation margin = {margin:.2f}x\n"
            f"module PERCEPTUAL_DISTANCE_THRESHOLD = "
            f"{PERCEPTUAL_DISTANCE_THRESHOLD}"
        )
    return result


if __name__ == "__main__":  # pragma: no cover
    calibrate()
