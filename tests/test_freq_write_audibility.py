"""Pin which SID frequency writes actually reach the output -- the reference for
"is it safe to absorb/discard a freq write?" in the skeleton encoder.

Proven here against pyresidfp (reSID-fp), so the encoder never discards an audible
write on a wrong mental model of the envelope/oscillator:

  * **Release is NOT instant** (even SR release-nibble 0): after gate-off the SID
    keeps producing sound for many PAL frames while the envelope decays.
  * **A frequency change is audible in ANY envelope phase, including release.**
    Writing a different freq during gate-off/release changes the output.
  * **The NOISE waveform's freq is the noise pitch/colour -- fully audible.** A
    noise frame's freq is real content, not discardable timbre.
  * **The TEST bit (ctrl bit 3) resets the oscillator**, so a freq write on a
    test-bit frame is (near-)inaudible -- safe to absorb to a NEARBY value. Real
    per-write timing caveat: freq is written BEFORE the test byte (canonical order)
    so it runs for a brief pre-TEST window; a wild multi-octave triangle jump leaks
    there (so absorb to the adjacent note's freq, not an arbitrary constant), and
    **PW on a test-bit frame IS audible** (the pulse threshold takes effect in that
    window) -- not absorbable. Songs use the test bit per note as a HARD RESTART: it
    zeroes the
    oscillator accumulator so each note attacks from a consistent phase (it also
    reduces the attack's dependence on the prior oscillator state -- measured:
    prehold-2-vs-11 attack max|Δ| 9785 without test -> 5462 with). *Foobar*-style
    real example: Wiklund *Facemorph* sets the test bit on ~27% of voice-0 ctrl
    frames (ctrl `0x19` = tri+gate+test, `0x09` = gate+test) -- one HR frame per note.
  * **Combined waveforms are freq-audible.** Setting two waveform bits ANDs the
    SID's oscillator outputs (pulse+saw, tri+saw, tri+pulse, ...): the result is
    still oscillator-driven, so its frequency is fully audible.
  * **The NOISE + waveform "lock" quirk.** Combining noise with another waveform
    feeds 0s back into the noise LFSR, so the output decays toward silence even
    while gate + sustain are held; a test-bit write re-seeds the LFSR. So a
    noise-combo frame's freq still matters and the LFSR state is path-dependent.

Conclusion for the encoder: a freq write on a **test-bit frame** may be absorbed
only to a NEARBY value (see the real-timing caveat above). Noise-frame,
combined-waveform, and release-frame freqs are fully audible and must be encoded,
even though they are not melodic pitch. PW on a test-bit frame is audible -- see
``test_register_canonicalization.test_test_bit_frame_pw_is_audible_but_freq_is_not``.
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
INTER_WRITE_SECONDS = 32 / PAL_CLOCK_HZ  # nominal _MIN_DIFF gap between writes
INAUDIBLE_MAX_INT16_DELTA = 16  # cf. test_sid_same_value_writes
AUDIBLE_MIN_INT16_DELTA = 500


def _make_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for r in range(25):
        sid.write_register(r, 0)
    sid.clock(timedelta(seconds=0.05))
    return sid


def _frame(sid, writes):
    """Clock each write through with a nominal inter-write gap, then hold the
    remainder of the frame -- matching the renderer, which clocks ~_MIN_DIFF cycles
    after each write (NOT a single end-of-frame clock). So writes within a frame take
    effect in sequence; absorption claims proven here therefore hold for the real
    render path, not just an idealised simultaneous-write model."""
    chunks = []
    held = 0.0
    n = len(writes)
    for i, (reg, val) in enumerate(writes):
        sid.write_register(reg, val)
        dur = INTER_WRITE_SECONDS if i < n - 1 else max(PAL_FRAME_SECONDS - held, 0.0)
        held += dur
        chunk = sid.clock(timedelta(seconds=dur))
        chunks.append(np.asarray(chunk if chunk else [], dtype=np.int16))
    if not writes:
        chunk = sid.clock(timedelta(seconds=PAL_FRAME_SECONDS))
        chunks.append(np.asarray(chunk if chunk else [], dtype=np.int16))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def _rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if len(a) else 0.0


def _wave_max_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    return int(np.abs(a[:n].astype(np.int32) - b[:n].astype(np.int32)).max())


def _pulse_setup(sid):
    """Voice-0 sustained pulse, fast attack, max sustain, release nibble 0."""
    sid.write_register(5, 0x09)  # AD: attack 0, decay 9
    sid.write_register(6, 0xF0)  # SR: sustain F, RELEASE 0
    sid.write_register(24, 0x0F)  # volume
    sid.write_register(2, 0x00)  # PW lo
    sid.write_register(3, 0x08)  # PW hi (50%)
    sid.write_register(0, 0x80)  # freq lo
    sid.write_register(1, 0x08)  # freq hi


def test_release_zero_is_not_instant_and_keeps_sounding():
    """Release-nibble 0 does NOT silence the SID immediately -- after gate-off the
    voice keeps producing audible output for many frames as the envelope decays."""
    sid = _make_sid()
    _pulse_setup(sid)
    _frame(sid, [(4, 0x41)])  # gate on
    for _ in range(3):
        _frame(sid, [(4, 0x41)])
    tail = [_rms(_frame(sid, [(4, 0x40)])) for _ in range(8)]  # gate off, 8 frames
    assert tail[0] > 1000, f"release frame 0 already silent ({tail[0]:.0f}); wrong"
    assert tail[5] > 100, f"still audible at frame 5 expected, got {tail[5]:.0f}"
    assert tail[0] > tail[-1], "release envelope should be decaying"


def test_freq_change_during_release_is_audible():
    """Changing the frequency during gate-off/release changes the output: a freq
    write in the release phase is NOT safe to discard."""

    def run(freq_during_release):
        sid = _make_sid()
        _pulse_setup(sid)
        _frame(sid, [(4, 0x41)])
        for _ in range(3):
            _frame(sid, [(4, 0x41)])
        out = []
        for _ in range(6):
            writes = [
                (0, freq_during_release & 0xFF),
                (1, (freq_during_release >> 8) & 0xFF),
                (4, 0x40),  # gate off, keep releasing
            ]
            out.append(_frame(sid, writes))
        return np.concatenate(out)

    same = run(0x0880)  # keep the note's freq
    changed = run(0xFFFF)  # write a wildly different freq during release
    mx = _wave_max_diff(same, changed)
    assert mx > AUDIBLE_MIN_INT16_DELTA, (
        f"freq change during release only moved the wave by {mx}; the emulator "
        f"says release freq IS audible, so this must be large"
    )


def test_noise_freq_is_audible():
    """The NOISE waveform's frequency is the noise pitch/colour and is fully
    audible -- a noise frame's freq is real content, never discardable timbre."""

    def run(freq):
        sid = _make_sid()
        sid.write_register(5, 0x00)
        sid.write_register(6, 0xF0)
        sid.write_register(24, 0x0F)
        out = []
        for _ in range(5):
            out.append(
                _frame(sid, [(0, freq & 0xFF), (1, (freq >> 8) & 0xFF), (4, 0x81)])
            )  # noise + gate
        return np.concatenate(out)

    low = run(0x0400)
    high = run(0xF000)
    mx = _wave_max_diff(low, high)
    assert (
        mx > AUDIBLE_MIN_INT16_DELTA
    ), f"two noise frequencies only differ by {mx}; noise freq must be audible"


def test_freq_during_test_bit_is_inaudible():
    """The TEST bit (ctrl bit 3) holds the oscillator in reset, so a frequency
    written while test is set does NOT reach the output -- the ONE freq write the
    encoder may safely absorb. Proven: two very different freqs during test frames
    produce an (near-)identical wave."""

    def run(freq_during_test):
        sid = _make_sid()
        _pulse_setup(sid)
        _frame(sid, [(4, 0x41)])
        for _ in range(3):
            _frame(sid, [(4, 0x41)])
        out = []
        for _ in range(3):
            writes = [
                (0, freq_during_test & 0xFF),
                (1, (freq_during_test >> 8) & 0xFF),
                (4, 0x49),  # pulse + gate + TEST (bit 3)
            ]
            out.append(_frame(sid, writes))
        return np.concatenate(out)

    note_freq = run(0x0880)
    wild_freq = run(0xFFFF)
    mx = _wave_max_diff(note_freq, wild_freq)
    assert mx <= INAUDIBLE_MAX_INT16_DELTA, (
        f"freq during TEST-bit frames moved the wave by {mx} > "
        f"{INAUDIBLE_MAX_INT16_DELTA}; test bit should make freq inaudible"
    )


def _pre_gate_then_note(pre_gate_freq, hard_restart):
    """Render frames that set a freq BEFORE any gate-on, then a gated note that re-writes its own
    freq. Returns (pre_gate_wave, note_wave). ``hard_restart`` test-bits the note's first frame
    (ctrl 0x49) to reset the oscillator phase, else it attacks from the carried-over phase.
    """
    sid = _make_sid()
    sid.write_register(5, 0x09)
    sid.write_register(6, 0xF0)
    sid.write_register(24, 0x0F)
    sid.write_register(2, 0x00)
    sid.write_register(3, 0x08)
    pre = [
        _frame(sid, [(0, pre_gate_freq & 0xFF), (1, (pre_gate_freq >> 8) & 0xFF)])
        for _ in range(4)
    ]
    note = []
    if hard_restart:
        note.append(_frame(sid, [(0, 0x80), (1, 0x08), (4, 0x49)]))
    for _ in range(6):
        note.append(_frame(sid, [(0, 0x80), (1, 0x08), (4, 0x41)]))
    return np.concatenate(pre), np.concatenate(note)


def test_freq_write_before_first_gate_on_is_inaudible_in_the_pre_gate_window():
    """A frequency write at the START of a song, BEFORE the voice has ever been gated on, is
    inaudible in that pre-gate window: the envelope has never been triggered, so the oscillator
    frequency cannot reach the output. Two pre-gate windows with wildly different freqs render
    (near-)identically."""
    pre_low, _ = _pre_gate_then_note(0x0200, hard_restart=False)
    pre_high, _ = _pre_gate_then_note(0xF000, hard_restart=False)
    mx = _wave_max_diff(pre_low, pre_high)
    assert mx <= INAUDIBLE_MAX_INT16_DELTA, (
        f"a pre-first-gate freq change moved the pre-gate wave by {mx}; it must be "
        f"inaudible (the un-gated voice emits no frequency-dependent output)"
    )


def test_pre_gate_freq_droppable_only_when_first_note_hard_restarts():
    """The pre-gate freq is silent in itself, but it advances the oscillator PHASE, which carries
    into the first note's attack: without an oscillator reset the dropped freq IS heard in that
    attack, WITH a test-bit hard restart it is fully don't-care. So a pre-gate freq write is
    audio-exact to drop only when the first note hard-restarts the oscillator."""
    note_low_n, note_high_n = (
        _pre_gate_then_note(0x0200, hard_restart=False)[1],
        _pre_gate_then_note(0xF000, hard_restart=False)[1],
    )
    note_low_h, note_high_h = (
        _pre_gate_then_note(0x0200, hard_restart=True)[1],
        _pre_gate_then_note(0xF000, hard_restart=True)[1],
    )
    no_reset = _wave_max_diff(note_low_n, note_high_n)
    with_reset = _wave_max_diff(note_low_h, note_high_h)
    assert no_reset > AUDIBLE_MIN_INT16_DELTA, (
        f"without an oscillator reset the pre-gate freq should leak into the first "
        f"note's attack phase; got only {no_reset}"
    )
    assert with_reset <= INAUDIBLE_MAX_INT16_DELTA, (
        f"a test-bit hard restart should reset the phase, making the pre-gate freq "
        f"don't-care; got {with_reset}"
    )


@pytest.mark.parametrize(
    "wf, label",
    [(0x60, "pulse+saw"), (0x30, "tri+saw"), (0x50, "tri+pulse")],
)
def test_combined_waveform_freq_is_audible(wf, label):
    """Two waveform bits AND the oscillator outputs; the result is still
    oscillator-driven, so its frequency is fully audible (cannot be absorbed)."""

    def run(freq):
        sid = _make_sid()
        sid.write_register(5, 0x09)
        sid.write_register(6, 0xF0)
        sid.write_register(24, 0x0F)
        sid.write_register(2, 0x00)
        sid.write_register(3, 0x08)
        out = []
        for _ in range(5):
            out.append(
                _frame(sid, [(0, freq & 0xFF), (1, (freq >> 8) & 0xFF), (4, wf | 1)])
            )
        return np.concatenate(out)

    mx = _wave_max_diff(run(0x0880), run(0x2000))
    assert (
        mx > AUDIBLE_MIN_INT16_DELTA
    ), f"{label} freq only moved the wave by {mx}; combined waveforms are audible"


def test_noise_combined_with_pulse_lfsr_lock_decays():
    """The noise+pulse combo feeds 0s into the noise LFSR, so the output decays
    toward silence even with gate + sustain held -- the SID 'noise lock'. (So a
    noise-combo frame's state is path-dependent; its freq is not free to discard.)"""
    sid = _make_sid()
    sid.write_register(5, 0x09)
    sid.write_register(6, 0xF0)  # sustain MAX, so any decay is the LFSR lock, not ADSR
    sid.write_register(24, 0x0F)
    sid.write_register(0, 0x80)
    sid.write_register(1, 0x08)
    rms = [_rms(_frame(sid, [(4, 0xC1)])) for _ in range(8)]  # noise+pulse + gate held
    assert rms[0] > 2000, f"noise+pulse should start loud, got {rms[0]:.0f}"
    assert rms[-1] < 0.6 * rms[0], (
        f"noise+pulse should lock/decay despite held gate+sustain: "
        f"{rms[0]:.0f} -> {rms[-1]:.0f}"
    )


def test_real_tune_test_bit_hr_freq_absorbable_to_a_nearby_value():
    """Derived from Wiklund *Facemorph*: its per-note hard restart is a test-bit
    frame (ctrl ``0x19`` = tri+gate+test, then ``0x09``). Under real per-write timing
    the freq is written BEFORE the test byte (canonical order), so it runs for the
    brief pre-TEST inter-write window -- it is NOT a free any-value don't-care. For
    triangle, a *wild* (multi-octave) HR freq leaks through that window and IS heard;
    but absorbing the HR freq to a NEARBY value (the adjacent note's pitch, sub-octave)
    is inaudible. So the encoder may absorb an HR-frame freq to the neighbouring note's
    freq -- not to an arbitrary constant."""

    def run(hr_freq):
        sid = _make_sid()
        # a settled triangle note (Facemorph's voice 0 is tri/pulse lead) at 0x1080
        sid.write_register(5, 0x00)
        sid.write_register(6, 0xB9)  # real Facemorph SR
        sid.write_register(24, 0x0F)
        sid.write_register(0, 0x80)
        sid.write_register(1, 0x10)
        for _ in range(5):
            _frame(sid, [(4, 0x11)])  # triangle + gate
        out = []
        # real Facemorph HR: test-bit frames carrying the freq we want to absorb
        for ctrl in (0x19, 0x09):  # tri+gate+test, then gate+test
            out.append(
                _frame(
                    sid, [(0, hr_freq & 0xFF), (1, (hr_freq >> 8) & 0xFF), (4, ctrl)]
                )
            )
        for _ in range(4):  # next note settles
            _frame(sid, [(0, 0x40), (1, 0x10), (4, 0x11)])
            out.append(_frame(sid, [(4, 0x11)]))
        return np.concatenate(out)

    nearby = _wave_max_diff(run(0x1080), run(0x1180))  # ~2 semitones from the note
    wild = _wave_max_diff(run(0x1080), run(0xFFFF))  # multi-octave jump
    assert (
        nearby <= INAUDIBLE_MAX_INT16_DELTA
    ), f"absorbing HR freq to a nearby value should be inaudible; got {nearby}"
    assert wild > AUDIBLE_MIN_INT16_DELTA, (
        f"a wild HR-freq jump IS heard in the pre-TEST window (triangle); got {wild} "
        f"-- so absorb to the adjacent note's freq, not an arbitrary constant"
    )
