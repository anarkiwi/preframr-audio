"""Emulator-proven register-write EQUIVALENCES (and non-equivalences) -- the basis
for canonicalising the token stream: if two writes/orders render the same audio, the
tokenizer may collapse them to one form, cutting variation the model must learn.

Proven against pyresidfp. The parser already canonicalises intra-frame write ORDER
(``_norm_pr_order``) and same-value writes (``_squeeze_changes``); these tests pin the
*audio* basis for that plus the control-aware equivalences that are not yet exploited.
"""

from __future__ import annotations

import itertools
from datetime import timedelta

import numpy as np
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel

PAL_FRAME_SECONDS = 1.0 / 50.123
EQUIVALENT_MAX_INT16_DELTA = 16  # the cycle-accounting floor (cf. same-value writes)
AUDIBLE_MIN_INT16_DELTA = 500


def _make_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for r in range(25):
        sid.write_register(r, 0)
    sid.clock(timedelta(seconds=0.05))
    return sid


def _frame(sid, writes):
    for reg, val in writes:
        sid.write_register(reg, val)
    chunk = sid.clock(timedelta(seconds=PAL_FRAME_SECONDS))
    return np.asarray(chunk if chunk else [], dtype=np.int16)


def _max_diff(a, b):
    n = min(len(a), len(b))
    return (
        int(np.abs(a[:n].astype(np.int32) - b[:n].astype(np.int32)).max()) if n else 0
    )


def _settle(sid):
    for reg, val in [(5, 0x00), (6, 0xF0), (24, 0x0F)]:
        sid.write_register(reg, val)


# ---- EQUIVALENCES (safe to canonicalise) ----


def test_intra_frame_write_order_is_equivalent():
    """The order of register writes WITHIN one frame does not change the audio --
    only the frame's final per-register state matters. (Justifies _norm_pr_order's
    canonical ordering.)"""

    def run(order):
        sid = _make_sid()
        _settle(sid)
        out = [_frame(sid, list(order))]
        for _ in range(4):
            out.append(_frame(sid, [(4, 0x41)]))
        return np.concatenate(out)

    base = [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08), (4, 0x41)]
    ref = run(base)
    worst = max(_max_diff(ref, run(p)) for p in itertools.permutations(base))
    assert (
        worst <= EQUIVALENT_MAX_INT16_DELTA
    ), f"intra-frame write order should be inaudible, worst max|Δ|={worst}"


def test_intra_frame_redundant_gate_toggles_collapse():
    """Gate on->off->on within ONE frame is equivalent to a single gate-on -- only
    the frame's final gate state matters (intermediate toggles are inaudible)."""

    def run(ctrl_writes):
        sid = _make_sid()
        _settle(sid)
        out = [_frame(sid, [(0, 0x80), (1, 0x10)] + ctrl_writes)]
        for _ in range(4):
            out.append(_frame(sid, [(4, 0x41)]))
        return np.concatenate(out)

    toggled = run([(4, 0x41), (4, 0x40), (4, 0x41)])
    once = run([(4, 0x41)])
    assert _max_diff(toggled, once) <= EQUIVALENT_MAX_INT16_DELTA


def test_freq_and_pw_on_a_test_frame_are_dont_care():
    """On a frame where the voice has the TEST bit set, freq and PW writes do not
    reach the output (oscillator held in reset). So a test-frame's freq/PW value can
    be canonicalised (e.g. to the surrounding value), removing a spurious distinct
    value -- this is the proven, SAFE version of absorbing the HR-window freq."""

    def run(freq, pw):
        sid = _make_sid()
        _settle(sid)
        for reg, val in [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08)]:
            sid.write_register(reg, val)
        _frame(sid, [(4, 0x41)])
        for _ in range(3):
            _frame(sid, [(4, 0x41)])
        return np.concatenate(
            [
                _frame(
                    sid,
                    [
                        (0, freq & 0xFF),
                        (1, (freq >> 8) & 0xFF),
                        (2, pw & 0xFF),
                        (3, (pw >> 8) & 0xFF),
                        (4, 0x49),  # pulse + gate + TEST
                    ],
                )
                for _ in range(2)
            ]
        )

    assert (
        _max_diff(run(0x0880, 0x800), run(0xFFFF, 0x000)) <= EQUIVALENT_MAX_INT16_DELTA
    )


def test_sync_with_a_non_oscillating_source_is_a_noop():
    """The hard-sync bit is a no-op when the source voice's oscillator is not running
    (source freq 0), so it can be canonicalised off in that case."""

    def run(sync):
        sid = _make_sid()
        _settle(sid)
        for reg, val in [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08)]:
            sid.write_register(reg, val)
        sid.write_register(14, 0x00)  # source voice 2 freq = 0 -> oscillator idle
        sid.write_register(15, 0x00)
        sid.write_register(18, 0x00)
        ctrl = 0x43 if sync else 0x41  # pulse(+sync)+gate
        return np.concatenate([_frame(sid, [(4, ctrl)]) for _ in range(4)])

    assert _max_diff(run(True), run(False)) <= EQUIVALENT_MAX_INT16_DELTA


# ---- NON-equivalences (guards: do NOT canonicalise these) ----


def test_waveform_bits_during_test_are_NOT_dont_care():
    """Guard: unlike freq/PW, the WAVEFORM bits during a test frame DO change the
    output (the held DC level at accumulator 0 depends on the waveform), so the
    waveform must not be canonicalised away on test frames."""

    def run(wf):
        sid = _make_sid()
        _settle(sid)
        for reg, val in [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08)]:
            sid.write_register(reg, val)
        _frame(sid, [(4, 0x41)])
        for _ in range(3):
            _frame(sid, [(4, 0x41)])
        return np.concatenate([_frame(sid, [(4, wf | 0x08)]) for _ in range(2)])

    assert _max_diff(run(0x00), run(0x40)) > AUDIBLE_MIN_INT16_DELTA  # none vs pulse


def test_ring_with_a_silent_source_silences_not_noop():
    """Guard: ring-mod with a non-oscillating source does NOT pass the carrier through
    -- it silences the voice (carrier x 0). So the ring bit is not a free no-op."""

    def run(ring):
        sid = _make_sid()
        _settle(sid)
        sid.write_register(0, 0x80)
        sid.write_register(1, 0x10)
        sid.write_register(14, 0x00)
        sid.write_register(15, 0x00)
        sid.write_register(18, 0x00)
        ctrl = 0x15 if ring else 0x11  # tri(+ring)+gate
        return np.concatenate([_frame(sid, [(4, ctrl)]) for _ in range(4)])

    assert _max_diff(run(True), run(False)) > AUDIBLE_MIN_INT16_DELTA
