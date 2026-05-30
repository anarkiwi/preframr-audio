"""Combinatorial SID feature behaviours, pinned against pyresidfp -- the reference
for how the SID's voice features interact, so the encoder reasons about real
hardware, not a mental model. Companion to ``test_freq_write_audibility.py``.

Covered: pulse width, ring modulation (cross-voice), oscillator sync (cross-voice),
the filter (routing + attenuation), the ADSR envelope progression, the envelope's
dependence on prior state, the test-bit oscillator reset, and the full **hard
restart** sequence (the SID Wizard mechanism: 2 pre-HR frames of fast-drain HR-ADSR
+ gate-off, then 1 test-bit frame, then the note -- draining the envelope AND
resetting the oscillator phase for a consistent attack).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel

PAL_FRAME_SECONDS = 1.0 / 50.123
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


def _rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if len(a) else 0.0


def _max_diff(a, b):
    n = min(len(a), len(b))
    return (
        int(np.abs(a[:n].astype(np.int32) - b[:n].astype(np.int32)).max()) if n else 0
    )


def test_pulse_width_shapes_the_pulse_output():
    """Pulse width sets the duty cycle, so it shapes the pulse output level/timbre.
    (On the 8580, pw=0 is NOT silent -- a known quirk -- but it is distinct.)"""

    def run(pw):
        sid = _make_sid()
        for reg, val in [(5, 0x00), (6, 0xF0), (24, 0x0F), (0, 0x80), (1, 0x10)]:
            sid.write_register(reg, val)
        out = [
            _frame(sid, [(2, pw & 0xFF), (3, (pw >> 8) & 0xFF), (4, 0x41)])
            for _ in range(4)
        ]
        return _rms(np.concatenate(out))

    quarter, half, full = run(0x400), run(0x800), run(0xFFF)
    assert abs(half - quarter) > 200 and abs(full - half) > 200, (
        f"pulse width should change the output: pw 0x400={quarter:.0f} "
        f"0x800={half:.0f} 0xFFF={full:.0f}"
    )


def test_ring_mod_output_depends_on_the_source_voice():
    """Ring modulation (ctrl bit 2) ring-mods voice 0's triangle with voice 2's
    oscillator MSB, so the output depends on the SOURCE voice's frequency."""

    def run(src_freq):
        sid = _make_sid()
        for reg, val in [(5, 0x00), (6, 0xF0), (24, 0x0F), (0, 0x80), (1, 0x08)]:
            sid.write_register(reg, val)
        sid.write_register(14, src_freq & 0xFF)  # voice 2 freq (the ring source)
        sid.write_register(15, (src_freq >> 8) & 0xFF)
        out = [_frame(sid, [(4, 0x15)]) for _ in range(4)]  # tri+ring+gate
        return np.concatenate(out)

    assert (
        _max_diff(run(0x0400), run(0x4000)) > AUDIBLE_MIN_INT16_DELTA
    ), "ring-mod output must depend on the source voice frequency"


def test_sync_output_depends_on_the_source_voice():
    """Hard sync (ctrl bit 1) resets voice 0's oscillator on voice 2's MSB edge, so
    the output depends on the SOURCE voice's frequency (with its oscillator running)."""

    def run(src_freq):
        sid = _make_sid()
        for reg, val in [(5, 0x00), (6, 0xF0), (24, 0x0F), (0, 0x80), (1, 0x10)]:
            sid.write_register(reg, val)
        sid.write_register(14, src_freq & 0xFF)
        sid.write_register(15, (src_freq >> 8) & 0xFF)
        sid.write_register(18, 0x21)  # voice 2: sawtooth + gate -> its oscillator runs
        out = [_frame(sid, [(4, 0x43)]) for _ in range(4)]  # pulse+sync+gate
        return np.concatenate(out)

    assert (
        _max_diff(run(0x0400), run(0x4000)) > AUDIBLE_MIN_INT16_DELTA
    ), "sync output must depend on the source voice frequency"


def test_filter_attenuates_the_routed_voice():
    """Routing a voice through the (low-pass) filter changes its output vs. the
    unrouted path -- the filter is a real, global, per-voice-routed stage."""

    def run(routed):
        sid = _make_sid()
        for reg, val in [(5, 0x00), (6, 0xF0), (0, 0x80), (1, 0x20)]:
            sid.write_register(reg, val)
        sid.write_register(21, 0x00)  # cutoff lo
        sid.write_register(22, 0x10)  # cutoff hi (low cutoff)
        sid.write_register(23, 0xF0 | (0x01 if routed else 0x00))  # res + route v0
        sid.write_register(24, 0x1F)  # LP + volume
        out = [_frame(sid, [(4, 0x21)]) for _ in range(4)]  # saw + gate
        return _rms(np.concatenate(out))

    routed, unrouted = run(True), run(False)
    assert abs(routed - unrouted) > 200, (
        f"filter routing should change the output: routed={routed:.0f} "
        f"unrouted={unrouted:.0f}"
    )


def test_sustain_level_sets_held_amplitude_and_gate_off_releases():
    """The ADSR sustain nibble sets the held amplitude (a low sustain holds quieter
    than a high one), and gate-off then releases (amplitude falls)."""

    def held(sustain_nibble):
        sid = _make_sid()
        for reg, val in [
            (5, 0x00),
            (24, 0x0F),
            (2, 0x00),
            (3, 0x08),
            (0, 0x80),
            (1, 0x10),
        ]:
            sid.write_register(reg, val)
        sid.write_register(6, (sustain_nibble << 4) | 0x00)  # SR: sustain, release 0
        _frame(sid, [(4, 0x41)])
        for _ in range(3):  # let attack/decay settle to sustain
            _frame(sid, [(4, 0x41)])
        return _rms(_frame(sid, [(4, 0x41)]))

    assert held(0xF) > held(0x4) + 200, "higher sustain should hold louder"

    sid = _make_sid()
    for reg, val in [
        (5, 0x00),
        (6, 0xF0),
        (24, 0x0F),
        (2, 0x00),
        (3, 0x08),
        (0, 0x80),
        (1, 0x10),
    ]:
        sid.write_register(reg, val)
    for _ in range(4):
        _frame(sid, [(4, 0x41)])
    falling = [_rms(_frame(sid, [(4, 0x40)])) for _ in range(5)]
    assert falling[-1] < falling[0], f"release should fall: {falling}"


def test_attack_depends_on_prior_envelope_state():
    """The SID re-attacks from wherever the envelope currently is, so a new note's
    attack depends on the PRIOR note's envelope state -- this is what hard restart
    exists to fix."""

    def run(prior_hold):
        sid = _make_sid()
        for reg, val in [(24, 0x0F), (2, 0x00), (3, 0x08), (0, 0x80), (1, 0x08)]:
            sid.write_register(reg, val)
        sid.write_register(5, 0x44)
        sid.write_register(6, 0xAA)  # slow-ish release -> long-lived envelope
        _frame(sid, [(4, 0x41)])
        for _ in range(prior_hold):
            _frame(sid, [(4, 0x41)])
        _frame(sid, [(4, 0x40)])  # 1-frame gate-off, NO hard restart
        sid.write_register(5, 0x09)
        sid.write_register(6, 0xF0)
        return [_rms(x) for x in [_frame(sid, [(4, 0x41)]) for _ in range(4)]]

    short_prior = run(2)
    long_prior = run(20)
    env_drift = max(abs(a - b) for a, b in zip(short_prior, long_prior, strict=False))
    assert (
        env_drift > 100
    ), f"without HR the attack envelope should differ by prior state, got {env_drift:.0f}"


def test_test_bit_resets_oscillator_phase():
    """The TEST bit resets the oscillator accumulator, so a note attacked right
    after a test-bit frame has a far more prior-independent WAVEFORM than one
    attacked without it (the oscillator-reset half of hard restart)."""

    def attack(prior_hold, use_test):
        sid = _make_sid()
        for reg, val in [
            (5, 0x00),
            (6, 0xF0),
            (24, 0x0F),
            (2, 0x00),
            (3, 0x08),  # PW set, so the pulse is at full level (8580 pw=0 is quiet)
            (0, 0x80),
            (1, 0x08),
        ]:
            sid.write_register(reg, val)
        _frame(sid, [(4, 0x41)])
        for _ in range(prior_hold):
            _frame(sid, [(4, 0x41)])
        _frame(sid, [(4, 0x48 if use_test else 0x40)])  # gate off (+ TEST if reset)
        _frame(sid, [(4, 0x48 if use_test else 0x40)])
        return np.concatenate([_frame(sid, [(4, 0x41)]) for _ in range(3)])

    no_test = _max_diff(attack(2, False), attack(11, False))
    with_test = _max_diff(attack(2, True), attack(11, True))
    assert with_test < no_test * 0.7, (
        f"test bit should make the attack waveform more prior-independent: "
        f"no-test max|Δ|={no_test} vs with-test {with_test}"
    )


def test_hard_restart_real_sequence_reduces_prior_dependence():
    """The full SID Wizard hard restart -- 2 pre-HR frames of fast-drain HR-ADSR +
    gate-off, then a test-bit frame, then the note -- both drains the envelope and
    resets the oscillator, so the new note is markedly more prior-independent than
    a plain 1-frame gate-off re-strike."""

    def newnote(prior_hold, hard_restart):
        sid = _make_sid()
        for reg, val in [(24, 0x0F), (2, 0x00), (3, 0x08), (0, 0x80), (1, 0x08)]:
            sid.write_register(reg, val)
        sid.write_register(5, 0x44)
        sid.write_register(6, 0xAA)
        _frame(sid, [(4, 0x41)])
        for _ in range(prior_hold):
            _frame(sid, [(4, 0x41)])
        if hard_restart:
            sid.write_register(5, 0x00)  # HR-AD: fast attack/decay
            sid.write_register(6, 0x00)  # HR-SR: sustain 0, fastest release
            _frame(sid, [(4, 0x40)])  # PRE_HR_LEAD_FRAMES: gate off, drain
            _frame(sid, [(4, 0x40)])
            _frame(sid, [(4, 0x08)])  # HR_FRAMES: test bit -> reset oscillator
        else:
            _frame(sid, [(4, 0x40)])
        sid.write_register(5, 0x09)  # real note ADSR
        sid.write_register(6, 0xF0)
        return np.concatenate([_frame(sid, [(4, 0x41)]) for _ in range(4)])

    plain = _max_diff(newnote(2, False), newnote(20, False))
    hr = _max_diff(newnote(2, True), newnote(20, True))
    assert hr < plain * 0.7, (
        f"hard restart should reduce prior-dependence of the attack: "
        f"plain re-strike max|Δ|={plain} vs hard-restart {hr}"
    )
