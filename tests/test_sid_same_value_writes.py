"""Pin the audio impact of same-value SID register writes."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel


RENDER_SECONDS = 0.04

SAME_VALUE_MAX_INT16_DELTA = 16

STATE_CHANGE_MIN_INT16_DELTA = 1000


def _make_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for r in range(25):
        sid.write_register(r, 0)
    _ = sid.clock(timedelta(seconds=0.1))
    return sid


def _render_seconds(sid, seconds):
    chunk = sid.clock(timedelta(seconds=seconds))
    return np.asarray(chunk if chunk else [], dtype=np.int16)


def _setup_voice_pulse(sid):
    """Voice 0 sustained-pulse program. AD=$09 (fast attack), SR=$F0
    (sustain max, release 0), FREQ=$0880, PW=$0800 (50% duty),
    CTRL=$41 (pulse + gate). Holds for the full render."""
    sid.write_register(5, 0x09)
    sid.write_register(6, 0xF0)
    sid.write_register(0, 0x80)
    sid.write_register(1, 0x08)
    sid.write_register(2, 0x00)
    sid.write_register(3, 0x08)
    sid.write_register(24, 0x0F)
    sid.write_register(4, 0x41)


def _wave_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0, 0.0, 0.0
    aa = a[:n].astype(np.int32)
    bb = b[:n].astype(np.int32)
    d = np.abs(aa - bb)
    return int(d.max()), float(d.mean()), float((d > 0).mean())


@pytest.mark.parametrize(
    "reg, val, label",
    [
        (0, 0x80, "FREQ_LO"),
        (1, 0x08, "FREQ_HI"),
        (2, 0x00, "PW_LO"),
        (3, 0x08, "PW_HI"),
        (4, 0x41, "CTRL (pulse + gate)"),
        (5, 0x09, "AD"),
        (6, 0xF0, "SR"),
        (24, 0x0F, "MODE_VOL"),
    ],
)
def test_same_value_writes_are_audibly_bounded(  # pylint: disable=unused-argument
    reg, val, label, capsys
):
    """Re-writing the same value to a SID register is NOT semantic
    re-triggering, but each pyresidfp ``write_register`` call advances
    internal cycle accounting -- a second identical write moves the
    continuous-time state by a few cycles vs the single-write path.
    """
    half = RENDER_SECONDS / 2

    sid_a = _make_sid()
    _setup_voice_pulse(sid_a)
    sid_a.write_register(reg, val)
    wav_a = _render_seconds(sid_a, RENDER_SECONDS)

    sid_b = _make_sid()
    _setup_voice_pulse(sid_b)
    sid_b.write_register(reg, val)
    wav_b1 = _render_seconds(sid_b, half)
    sid_b.write_register(reg, val)
    wav_b2 = _render_seconds(sid_b, half)
    wav_b = np.concatenate([wav_b1, wav_b2])

    mx, mn, fr = _wave_diff(wav_a, wav_b)
    print(
        f"  {label:>22} re-write: max|Δ|={mx:>4d} mean|Δ|={mn:>6.3f} "
        f"frac_diff={fr:>5.1%}"
    )
    assert mx <= SAME_VALUE_MAX_INT16_DELTA, (
        f"same-value {label} re-write produced max|Δ|={mx} > "
        f"{SAME_VALUE_MAX_INT16_DELTA}. That's larger than the "
        f"cycle-accounting perturbation expected; check if this register "
        f"semantically re-triggers on same-value writes, or if a "
        f"pyresidfp update changed write timing."
    )


def test_distinct_value_ctrl_test_bit_does_change_audio():
    """Sanity check: a DIFFERENT CTRL value (toggling TEST=1) IS
    audio-affecting -- TEST forces the oscillator phase to 0 while
    asserted. This is what ``_squeeze_changes`` correctly KEEPS (val
    differs from previous write).
    """
    half = RENDER_SECONDS / 2

    sid_a = _make_sid()
    _setup_voice_pulse(sid_a)
    wav_a = _render_seconds(sid_a, RENDER_SECONDS)

    sid_b = _make_sid()
    _setup_voice_pulse(sid_b)
    wav_b1 = _render_seconds(sid_b, half)
    sid_b.write_register(4, 0x49)
    sid_b.write_register(4, 0x41)
    wav_b2 = _render_seconds(sid_b, half)
    wav_b = np.concatenate([wav_b1, wav_b2])

    mx, mn, fr = _wave_diff(wav_a, wav_b)
    print(
        f"  CTRL TEST toggle mid-render: max|Δ|={mx} mean|Δ|={mn:.3f} "
        f"frac_diff={fr:.1%}"
    )
    assert mx >= STATE_CHANGE_MIN_INT16_DELTA, (
        f"expected CTRL TEST-toggle to produce max|Δ| ≥ "
        f"{STATE_CHANGE_MIN_INT16_DELTA} but got {mx}; either the diff "
        f"metric is broken or pyresidfp's CTRL TEST emulation regressed"
    )


def test_cumulative_effect_of_repeated_same_value_writes():
    """Real-world relevance: glow_worm's source has ~1500 same-value
    MODE_VOL=15 writes per second that ``_squeeze_changes`` drops.
    Does the cumulative effect of N same-value writes scale linearly,
    or does it saturate?
    """
    n_ms = 40
    sid_a = _make_sid()
    _setup_voice_pulse(sid_a)
    sid_a.write_register(24, 0x0F)
    wav_a = _render_seconds(sid_a, n_ms / 1000)

    sid_b = _make_sid()
    _setup_voice_pulse(sid_b)
    sid_b.write_register(24, 0x0F)
    chunks = []
    for _ in range(n_ms):
        sid_b.write_register(24, 0x0F)
        chunks.append(_render_seconds(sid_b, 0.001))
    wav_b = np.concatenate(chunks)

    mx, mn, fr = _wave_diff(wav_a, wav_b)
    print(
        f"  MODE_VOL=$0F × {n_ms} identical re-writes: max|Δ|={mx} "
        f"mean|Δ|={mn:.3f} frac_diff={fr:.1%}"
    )
    PERCEPTUAL_THRESHOLD = 327
    assert mx < PERCEPTUAL_THRESHOLD, (
        f"N={n_ms} same-value re-writes produced max|Δ|={mx} ≥ "
        f"{PERCEPTUAL_THRESHOLD} (1% of full scale). That's above the "
        f"perceptual floor and ``_squeeze_changes`` is dropping audibly "
        f"significant rows on dense-write fixtures."
    )
