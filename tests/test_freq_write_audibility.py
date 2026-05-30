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
  * **The TEST bit (ctrl bit 3) resets the oscillator**, so a freq write while
    test is set does NOT reach the output -- the ONE freq write that is safe to
    absorb. Songs use the test bit per note as a HARD RESTART: it zeroes the
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

Conclusion for the encoder: only freq writes on **test-bit frames** may be
absorbed. Noise-frame, combined-waveform, and release-frame freqs must be encoded
(audible), even though they are not melodic pitch.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

pyresidfp = pytest.importorskip("pyresidfp")
SoundInterfaceDevice = pyresidfp.SoundInterfaceDevice
ChipModel = pyresidfp.sound_interface_device.ChipModel

PAL_FRAME_SECONDS = 1.0 / 50.123
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
    """Apply ``writes`` then clock exactly one PAL frame; return int16 samples."""
    for reg, val in writes:
        sid.write_register(reg, val)
    chunk = sid.clock(timedelta(seconds=PAL_FRAME_SECONDS))
    return np.asarray(chunk if chunk else [], dtype=np.int16)


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


def test_real_tune_test_bit_hr_freq_is_absorbable():
    """Derived from Wiklund *Facemorph*: its per-note hard restart is a test-bit
    frame (ctrl ``0x19`` = tri+gate+test, then ``0x09``). Replaying that real HR
    with two very different frequencies on the test frames yields the same audio --
    so those (and only those) freq writes are safe for the encoder to absorb."""

    def run(hr_freq):
        sid = _make_sid()
        # a settled triangle note (Facemorph's voice 0 is tri/pulse lead)
        sid.write_register(5, 0x00)
        sid.write_register(6, 0xB9)  # real Facemorph SR
        sid.write_register(24, 0x0F)
        sid.write_register(0, 0x80)
        sid.write_register(1, 0x10)
        _frame(sid, [(4, 0x11)])  # triangle + gate
        for _ in range(4):
            _frame(sid, [(4, 0x11)])
        out = []
        # real Facemorph HR: test-bit frames with a freq write we want to absorb
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

    mx = _wave_max_diff(run(0x0880), run(0xFFFF))
    assert (
        mx <= INAUDIBLE_MAX_INT16_DELTA
    ), f"real-tune HR test-bit freq moved the wave by {mx}; should be absorbable"
