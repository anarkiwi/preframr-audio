"""Unit tests for ``preframr_audio.fidelity.compare_renders``."""

from __future__ import annotations

import numpy as np
import pytest

from preframr_audio.fidelity import (
    FRAME_RMS_TOLERANCE,
    PAL_FRAME_HZ,
    AudioFidelityResult,
    compare_renders,
)

SAMPLE_RATE = 44100
DEFAULT_LEN = SAMPLE_RATE * 4
RNG_SEED = 12345


def _signal(n: int, seed: int = RNG_SEED, amp: int = 8000) -> np.ndarray:
    """Generate a deterministic non-periodic int16 signal of length ``n``."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(scale=amp / 3.0, size=n)
    return raw.clip(-32768, 32767).astype(np.int16)


def _frame_samples() -> int:
    return int(round(SAMPLE_RATE / PAL_FRAME_HZ))


def test_pass_identical_arrays():
    """Two byte-identical streams pass with shape=PASS, empty diagnostic."""
    a = _signal(DEFAULT_LEN)
    result = compare_renders(a, a.copy(), SAMPLE_RATE)
    assert result.passed is True
    assert result.shape == "PASS"
    assert result.diagnostic == ""
    assert result.worst_frame_rel_rms == pytest.approx(0.0, abs=1e-9)


def test_pass_within_tolerance_noise():
    """Tiny gaussian noise (~1% of peak) under tolerance still passes."""
    rng = np.random.default_rng(RNG_SEED)
    a = _signal(DEFAULT_LEN, amp=10000)
    noise = rng.normal(scale=80, size=DEFAULT_LEN).astype(np.int16)
    b = (a.astype(np.int32) + noise).clip(-32768, 32767).astype(np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is True, f"unexpected fail: {result.diagnostic}"
    assert result.shape == "PASS"
    assert 0.0 < result.worst_frame_rel_rms < FRAME_RMS_TOLERANCE


def test_duration_mismatch():
    """Different stream lengths -> DURATION_MISMATCH."""
    a = _signal(DEFAULT_LEN)
    b = _signal(DEFAULT_LEN - 1000)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is False
    assert result.shape == "DURATION_MISMATCH"
    assert "DURATION MISMATCH" in result.diagnostic


def test_frame_cadence_break():
    """Shift a stream by 2 whole frames -> FRAME_CADENCE_BREAK; lag
    should be detected near +2 frames."""
    fw = _frame_samples()
    shift = 2 * fw
    a = _signal(DEFAULT_LEN)
    b = np.concatenate([np.zeros(shift, dtype=np.int16), a[:-shift]])
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is False
    assert result.shape == "FRAME_CADENCE_BREAK"
    assert abs(result.lag_samples - shift) <= fw


def test_max_frame_drift_relaxes_cadence_gate():
    """A 2-frame shift is a cadence break at the default tolerance but not
    when max_frame_drift admits more frames of lag (it then falls through to
    the per-frame RMS path instead of the cadence gate)."""
    fw = _frame_samples()
    shift = 2 * fw
    a = _signal(DEFAULT_LEN)
    b = np.concatenate([np.zeros(shift, dtype=np.int16), a[:-shift]])
    assert compare_renders(a, b, SAMPLE_RATE).shape == "FRAME_CADENCE_BREAK"
    relaxed = compare_renders(a, b, SAMPLE_RATE, max_frame_drift=3)
    assert relaxed.shape != "FRAME_CADENCE_BREAK"
    assert abs(relaxed.lag_samples - shift) <= fw


def test_initial_state_divergence():
    """Noise burst in the head N samples decays into a clean tail ->
    INITIAL_STATE_DIVERGENCE (head-RMS >> tail-RMS)."""
    rng = np.random.default_rng(RNG_SEED)
    a = _signal(DEFAULT_LEN, amp=10000)
    b = a.copy()
    head_n = DEFAULT_LEN // 20
    noise = rng.normal(scale=5000, size=head_n).astype(np.int16)
    b[:head_n] = (
        (b[:head_n].astype(np.int32) + noise).clip(-32768, 32767).astype(np.int16)
    )
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is False, f"expected fail; got {result}"
    assert result.shape == "INITIAL_STATE_DIVERGENCE"
    assert "INITIAL-STATE DIVERGENCE" in result.diagnostic


def test_drifting_divergence():
    """Linearly-growing noise envelope -> DRIFTING_DIVERGENCE
    (tail-RMS >> head-RMS)."""
    rng = np.random.default_rng(RNG_SEED)
    a = _signal(DEFAULT_LEN, amp=10000)
    envelope = np.linspace(0.0, 3000.0, DEFAULT_LEN)
    noise = (rng.normal(size=DEFAULT_LEN) * envelope).astype(np.int16)
    b = (a.astype(np.int32) + noise).clip(-32768, 32767).astype(np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is False
    assert result.shape == "DRIFTING_DIVERGENCE"
    assert "DRIFTING DIVERGENCE" in result.diagnostic


def test_constant_divergence():
    """Uniform noise across the whole stream above tolerance ->
    CONSTANT_DIVERGENCE (head-RMS ~ tail-RMS)."""
    rng = np.random.default_rng(RNG_SEED)
    a = _signal(DEFAULT_LEN, amp=10000)
    noise = rng.normal(scale=1000, size=DEFAULT_LEN).astype(np.int16)
    b = (a.astype(np.int32) + noise).clip(-32768, 32767).astype(np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is False
    assert result.shape == "CONSTANT_DIVERGENCE"
    assert "CONSTANT DIVERGENCE" in result.diagnostic


def test_tolerance_boundary_pass():
    """An RMS divergence slightly under the configured tolerance passes;
    slightly over fails. Catches off-by-one tolerance bugs."""
    a = _signal(DEFAULT_LEN, amp=10000)
    target_rms = int(0.04 * 10000)
    delta = np.full(DEFAULT_LEN, target_rms, dtype=np.int32)
    b = (a.astype(np.int32) + delta).clip(-32768, 32767).astype(np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is True, f"unexpected fail at 4% RMS: {result.diagnostic}"
    assert result.shape == "PASS"


def test_tolerance_boundary_fail():
    """RMS divergence above tolerance fails."""
    a = _signal(DEFAULT_LEN, amp=10000)
    target_rms = int(0.10 * 10000)
    delta = np.full(DEFAULT_LEN, target_rms, dtype=np.int32)
    b = (a.astype(np.int32) + delta).clip(-32768, 32767).astype(np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is False


def test_custom_tolerance_relaxes_judgement():
    """Caller can pass tolerance=0.15 to admit a 10% divergence."""
    a = _signal(DEFAULT_LEN, amp=10000)
    target_rms = int(0.10 * 10000)
    delta = np.full(DEFAULT_LEN, target_rms, dtype=np.int32)
    b = (a.astype(np.int32) + delta).clip(-32768, 32767).astype(np.int16)
    result = compare_renders(a, b, SAMPLE_RATE, tolerance=0.15)
    assert result.passed is True
    assert result.shape == "PASS"


def test_no_info_short_streams():
    """Streams shorter than one PAL frame return NO_INFO, treat-as-pass."""
    fw = _frame_samples()
    a = _signal(fw // 2)
    b = _signal(fw // 2)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.shape == "NO_INFO"
    assert result.passed is True
    assert "NO INFO" in result.diagnostic


def test_zero_length_inputs():
    """Two zero-length arrays compare as PASS (same shape) but
    NO_INFO since there's nothing to compute on."""
    a = np.zeros(0, dtype=np.int16)
    b = np.zeros(0, dtype=np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.shape == "NO_INFO"
    assert result.passed is True


def test_silent_streams():
    """Both streams all-zero: no signal, no peak, no divergence. Pass."""
    a = np.zeros(DEFAULT_LEN, dtype=np.int16)
    b = np.zeros(DEFAULT_LEN, dtype=np.int16)
    result = compare_renders(a, b, SAMPLE_RATE)
    assert result.passed is True
    assert result.shape == "PASS"


def test_result_is_frozen_dataclass():
    """AudioFidelityResult should be immutable; modifying a field raises."""
    result = compare_renders(_signal(DEFAULT_LEN), _signal(DEFAULT_LEN), SAMPLE_RATE)
    assert isinstance(result, AudioFidelityResult)
    with pytest.raises(dataclasses_FrozenInstanceError()):
        result.shape = "MUTATED"  # type: ignore[misc]


def dataclasses_FrozenInstanceError():
    """Late-bound exception type so the test file imports without
    pulling FrozenInstanceError from the stdlib at module top."""
    import dataclasses

    return dataclasses.FrozenInstanceError
