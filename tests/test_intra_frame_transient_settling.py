"""License the tokenizer's intra-frame freq/PW transient settling against pyresidfp.

preframr-tokens ``canonical_writes`` keeps only the SETTLED end-of-frame value of a freq/PW register and
drops any earlier same-register write in that frame (an intra-frame transient). These tests are the
chip-level evidence that the drop is audibly safe: under the renderer's real per-write timing (each write
held ~``_MIN_DIFF`` cycles), a same-register freq/PW value that is written then overwritten within one
frame is PERCEPTUALLY identical to writing the settled value alone.

A freq transient is not bit-identical -- those ~32 cycles of a different oscillator increment permanently
phase-shift the accumulator, so a strict per-sample comparison sees a large delta forever after. That is
inaudible phase, not timbre: the perceptual gate (``fidelity.perceptual_distance``, the pitch/phase-tolerant
band-power oracle) reads ~0. The companion that the drop is NOT free for audible state lives in
``test_register_canonicalization`` (intra-frame CTRL/ADSR order and PW-on-test-frames are preserved).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from preframr_audio.fidelity import PERCEPTUAL_DISTANCE_THRESHOLD, perceptual_distance

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel

PAL_FRAME_SECONDS = 1.0 / 50.123
PAL_CLOCK_HZ = 985248
INTER_WRITE_SECONDS = (
    32 / PAL_CLOCK_HZ
)  # nominal _MIN_DIFF gap; each write takes effect

_GATE_ON_FRAMES = 4
_HOLD_FRAMES = 100
_TOTAL_FRAMES = _GATE_ON_FRAMES + 1 + _HOLD_FRAMES

# A freq transient phase-decorrelates the oscillator: per-sample it is plainly visible (cf.
# AUDIBLE_MIN_INT16_DELTA in test_register_canonicalization), yet perceptually it is silent.
PHASE_DECORRELATION_MIN_DELTA = 500
PERCEPTUAL_NOISE_FLOOR = 0.05


def _make_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for reg in range(25):
        sid.write_register(reg, 0)
    sid.clock(timedelta(seconds=0.05))
    return sid


def _frame(sid, writes):
    """Clock each write through a nominal inter-write gap so multiple writes in one frame take effect in
    sequence; the final write holds the remainder of the PAL frame (mirrors the renderer's per-write
    timing used in test_register_canonicalization)."""
    if not writes:
        chunk = sid.clock(timedelta(seconds=PAL_FRAME_SECONDS))
        return np.asarray(chunk if chunk else [], dtype=np.int16)
    chunks = []
    held = 0.0
    n = len(writes)
    for i, (reg, val) in enumerate(writes):
        sid.write_register(reg, val)
        dur = INTER_WRITE_SECONDS if i < n - 1 else max(PAL_FRAME_SECONDS - held, 0.0)
        held += dur
        chunk = sid.clock(timedelta(seconds=dur))
        chunks.append(np.asarray(chunk if chunk else [], dtype=np.int16))
    return np.concatenate(chunks)


def _render(first_frame_writes):
    """A sustained pulse note, then one frame carrying ``first_frame_writes`` (the transient-or-settled
    frame), then a long hold so the perceptual windows are full."""
    sid = _make_sid()
    for reg, val in [(5, 0x00), (6, 0xF0), (24, 0x0F)]:
        sid.write_register(reg, val)
    for reg, val in [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08)]:
        sid.write_register(reg, val)
    out = [_frame(sid, [(4, 0x41)]) for _ in range(_GATE_ON_FRAMES)]
    out.append(_frame(sid, first_frame_writes))
    out.extend(_frame(sid, []) for _ in range(_HOLD_FRAMES))
    return np.concatenate(out)


def _sample_rate(samples):
    return int(round(len(samples) / (_TOTAL_FRAMES * PAL_FRAME_SECONDS)))


def _max_abs_delta(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    return int(np.abs(a[:n].astype(np.int32) - b[:n].astype(np.int32)).max())


@pytest.mark.parametrize(
    "reg, transient, settled, label",
    [
        (0, 0x40, 0x80, "freq_lo"),
        (1, 0x10, 0x20, "freq_hi"),
        (0, 0x00, 0xFF, "freq_lo_max"),
        (1, 0x00, 0xFF, "freq_hi_max"),
        (3, 0x08, 0x0F, "pw_hi"),
    ],
)
def test_intra_frame_transient_is_perceptually_equivalent_to_settled(
    reg, transient, settled, label
):
    """A same-register freq/PW value written then overwritten within one frame renders perceptually
    equivalent to writing the settled value alone -- so canonical_writes may drop the transient.
    """
    with_transient = _render([(reg, transient), (reg, settled), (4, 0x41)])
    settled_only = _render([(reg, settled), (4, 0x41)])
    sample_rate = _sample_rate(with_transient)
    distance, _per_window = perceptual_distance(
        with_transient, settled_only, sample_rate, win_s=1.0
    )
    assert distance <= PERCEPTUAL_DISTANCE_THRESHOLD, (
        f"{label} intra-frame transient should be perceptually equivalent to its settled value, "
        f"but perceptual distance {distance:.4f} exceeds the gate threshold "
        f"{PERCEPTUAL_DISTANCE_THRESHOLD}; the canonicalization that drops it would be audible."
    )


def test_freq_transient_is_inaudible_phase_not_timbre():
    """The equivalence is perceptual, not bit-exact: a freq transient leaves a large per-sample delta
    (phase decorrelation that persists), yet the perceptual distance stays near zero -- which is exactly
    why the strict per-sample oracle is the wrong tool for licensing this drop and the perceptual gate is
    the right one."""
    with_transient = _render([(1, 0x10), (1, 0x20), (4, 0x41)])
    settled_only = _render([(1, 0x20), (4, 0x41)])
    sample_rate = _sample_rate(with_transient)
    per_sample = _max_abs_delta(with_transient, settled_only)
    distance, _per_window = perceptual_distance(
        with_transient, settled_only, sample_rate, win_s=1.0
    )
    assert per_sample > PHASE_DECORRELATION_MIN_DELTA, (
        f"expected the freq transient to phase-decorrelate the oscillator (per-sample delta "
        f"> {PHASE_DECORRELATION_MIN_DELTA}); got {per_sample}. If this dropped to the same-value "
        f"floor the transient would be a true no-op and this test would be vacuous."
    )
    assert distance < PERCEPTUAL_NOISE_FLOOR, (
        f"freq transient perceptual distance {distance:.4f} should sit at the noise floor "
        f"(< {PERCEPTUAL_NOISE_FLOOR}); a higher value would mean the phase shift is actually audible."
    )
