"""Emulator-proven register-write behaviours under the renderer's REAL per-write
timing: it clocks ~``_MIN_DIFF`` cycles after each write, so intra-frame order and
repeated writes take effect -- they are NOT collapsed by a single end-of-frame clock.

These pin what the tokenizer must PRESERVE: intra-frame write order is audible, so the
stream must be EMITTED in canonical ascending order (freq, PW, CTRL, AD, SR; filter
last), never reordered after the fact, and multiple CTRL writes must be kept in time
order. A value that is audible is not discardable. Only genuinely redundant writes
(same value) are free to collapse (``_squeeze_changes``).

Proven against pyresidfp.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel

PAL_FRAME_SECONDS = 1.0 / 50.123
PAL_CLOCK_HZ = 985248
INTER_WRITE_SECONDS = (
    32 / PAL_CLOCK_HZ
)  # nominal _MIN_DIFF gap; each write takes effect
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
    """Clock each write through with a nominal inter-write gap so multiple writes
    in one frame take effect in sequence (not collapsed by a single end-of-frame
    clock); the final write holds the remainder of the frame."""
    chunks = []
    held = 0.0
    n = len(writes)
    for i, (reg, val) in enumerate(writes):
        sid.write_register(reg, val)
        dur = INTER_WRITE_SECONDS if i < n - 1 else max(PAL_FRAME_SECONDS - held, 0.0)
        held += dur
        c = sid.clock(timedelta(seconds=dur))
        chunks.append(np.asarray(c if c else [], dtype=np.int16))
    if not writes:
        c = sid.clock(timedelta(seconds=PAL_FRAME_SECONDS))
        chunks.append(np.asarray(c if c else [], dtype=np.int16))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def _max_diff(a, b):
    n = min(len(a), len(b))
    return (
        int(np.abs(a[:n].astype(np.int32) - b[:n].astype(np.int32)).max()) if n else 0
    )


def _settle(sid):
    for reg, val in [(5, 0x00), (6, 0xF0), (24, 0x0F)]:
        sid.write_register(reg, val)


# ---- MUST PRESERVE under the renderer's real per-write timing (NOT collapsible) ----
# The renderer clocks ~_MIN_DIFF cycles after each write, so intra-frame order and
# repeated writes take effect. These pin what the stream must emit / preserve.


def test_intra_frame_write_order_matters_so_emit_canonical():
    """Intra-frame write ORDER is audible under real timing: a CTRL/gate written
    before its freq attacks at the wrong pitch. So the pipeline must EMIT canonical
    ascending order (freq, PW, then CTRL), not reorder/collapse arbitrary orders."""

    def run(order):
        sid = _make_sid()
        _settle(sid)
        out = [_frame(sid, list(order))]
        for _ in range(4):
            out.append(_frame(sid, [(4, 0x41)]))
        return np.concatenate(out)

    canonical = [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08), (4, 0x41)]  # freq,PW,CTRL
    gate_first = [
        (4, 0x41),
        (0, 0x80),
        (1, 0x10),
        (2, 0x00),
        (3, 0x08),
    ]  # gate before freq
    assert _max_diff(run(canonical), run(gate_first)) > AUDIBLE_MIN_INT16_DELTA


def test_intra_frame_gate_toggles_take_effect():
    """Multiple CTRL writes in one frame (gate on->off->on, or the TEST/un-TEST
    pattern) each take effect under real timing -- they do NOT collapse to the final
    gate state, so they must be kept in time order, never merged."""

    def run(ctrl_writes):
        sid = _make_sid()
        _settle(sid)
        out = [_frame(sid, [(0, 0x80), (1, 0x10)] + ctrl_writes)]
        for _ in range(4):
            out.append(_frame(sid, [(4, 0x41)]))
        return np.concatenate(out)

    toggled = run([(4, 0x41), (4, 0x40), (4, 0x41)])
    once = run([(4, 0x41)])
    assert _max_diff(toggled, once) > EQUIVALENT_MAX_INT16_DELTA


def test_test_bit_frame_pw_is_audible_but_freq_is_not():
    """On a frame whose CTRL sets the TEST bit (written last, canonical order), the
    writes before it run for the brief pre-TEST inter-write window. PW takes effect
    immediately -- it is the pulse-compare threshold -- so test-bit-frame PW IS
    audible and must NOT be discarded. Freq only advances the oscillator phase, which
    is negligible over that window, and the TEST reset then zeroes the accumulator, so
    test-bit-frame freq is inaudible and IS discardable. (The old clock-once model
    collapsed the inter-write window and wrongly showed both don't-care; measured
    isolated under real per-write timing: freq Δ<=16, PW Δ>6000.) The companion
    freq-absorption proof is test_freq_write_audibility.test_freq_during_test_bit_is_inaudible.
    """

    def run(freq, pw):
        sid = _make_sid()
        _settle(sid)
        for reg, val in [(0, 0x80), (1, 0x10), (2, 0x00), (3, 0x08)]:
            sid.write_register(reg, val)
        for _ in range(4):
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

    freq_only = _max_diff(run(0x0880, 0x0800), run(0xFFFF, 0x0800))  # PW fixed
    pw_only = _max_diff(run(0x0880, 0x0800), run(0x0880, 0x0000))  # freq fixed
    assert (
        freq_only <= EQUIVALENT_MAX_INT16_DELTA
    ), f"test-bit-frame freq should be inaudible (discardable); got {freq_only}"
    assert (
        pw_only > AUDIBLE_MIN_INT16_DELTA
    ), f"test-bit-frame PW should be audible (not discardable); got {pw_only}"


# ---- EQUIVALENCES (safe to canonicalise) ----


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
