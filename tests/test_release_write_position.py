"""Where may a release-nibble (SR low) write be placed without changing the audio?

The envelope consults R only while gate=0 -- but not only while the envelope is
still releasing. Whenever gate=0, R is the rate prescaler's ACTIVE compare value:
the 15-bit LFSR runs against it even after the envelope counter has clamped at
zero, and because the comparison is for EQUALITY (the ADSR bug, cf.
``test_register_canonicalization.test_adsr_bug_attack_depends_on_prior_envelope_
state``), the counter's position relative to the next phase's compare decides
whether that phase's first step fires promptly or stalls a full LFSR wrap (~33ms).

So the invariant a tokenizer may rely on is the COMPARE-VALUE TRAJECTORY over
gate=0 time, not the write's timestamp. Measured placements of a mid-note R
change (the rate its own note's gate-off will release at):

- anywhere in the gate-on span, INCLUDING the onset frame AFTER the gate write:
  equivalent (R is latent the moment gate=1) -- this licenses folding a note's
  release rate into its note-on, emitted gate-then-SR;
- the onset frame BEFORE the gate write: NOT inert -- the inter-write gap is
  gate=0 dwell at the new compare, enough to carry the counter past attack rate
  0's compare value of 9 and flip the attack's ADSR-bug stall state either way;
- frames earlier, prior tail still releasing: the tail re-rates immediately;
- frames earlier, prior tail provably dead (ENV3=00): the gap's compare
  trajectory changes, and the next attack stalls.

The sustain nibble has no such freedom: S is the live per-voice volume while the
gate is on, so S writes own their timestamp.

Methodology: variants are WRITE-COUNT MATCHED -- every variant performs an SR
write at all three candidate slots (same-value rewrites where it is not the
change point, audibly bounded per ``test_sid_same_value_writes``), so chunk
boundaries and write cycle offsets are identical and diffs are purely semantic.

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
INTER_WRITE_SECONDS = 32 / PAL_CLOCK_HZ
EQUIVALENT_MAX_INT16_DELTA = 32  # cycle-accounting floor (16, cf. same-value
# writes) + measured resampler nondeterminism (identical schedules differ +-8)
AUDIBLE_MIN_INT16_DELTA = 500

# Two-note saw line on voice 3 (ENV3 readable): note1 on f1 / off f5, note2 on
# f12 / off f16. The SR rewrite under test is intended for note2's release; the
# candidate slots are the dead/releasing gap (f8), note2's onset frame (f12),
# and mid-note2 (f14, the corpus placement).
NOTE1_ON, NOTE1_OFF, NOTE2_ON, NOTE2_OFF = 1, 5, 12, 16
GAP_SLOT, ONSET_SLOT, MID_SLOT = 8, 12, 14
N_FRAMES = 22
FREQ_LO, FREQ_HI, CTRL, AD, SR = 14, 15, 18, 19, 20
ENV3 = 0x1C


def _make_sid():
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    sid.reset()
    for r in range(25):
        sid.write_register(r, 0)
    sid.clock(timedelta(seconds=0.05))
    return sid


def _frame(sid, writes):
    """Clock each write through with a nominal inter-write gap (real per-write
    timing, not collapsed by one end-of-frame clock)."""
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


def _line(sr_init, sr_gap, sr_onset, sr_mid, onset_before_gate=False):
    """Render the two-note line with SR writes at ALL of f8/f12/f14 (same-value
    where not the change point: write-count matched). Returns (per-frame audio,
    per-frame ENV3 read at frame end)."""
    sid = _make_sid()
    for reg, val in [(24, 0x0F), (FREQ_LO, 0x80), (FREQ_HI, 0x10), (AD, 0x00)]:
        sid.write_register(reg, val)
    sid.write_register(SR, sr_init)
    frames, envs = [], []
    for f in range(N_FRAMES):
        gate = (
            [(CTRL, 0x21)]
            if f in (NOTE1_ON, NOTE2_ON)
            else [(CTRL, 0x20)] if f in (NOTE1_OFF, NOTE2_OFF) else []
        )
        if f == GAP_SLOT:
            writes = [(SR, sr_gap)]
        elif f == ONSET_SLOT:
            sr = [(SR, sr_onset)]
            writes = sr + gate if onset_before_gate else gate + sr
        elif f == MID_SLOT:
            writes = [(SR, sr_mid)]
        else:
            writes = gate
        frames.append(_frame(sid, writes))
        envs.append(sid.read_register(ENV3))
    return frames, envs


def _diff(a_frames, b_frames):
    return max(_max_diff(a, b) for a, b in zip(a_frames, b_frames, strict=False))


def _mid(sr_old, sr_new, **kw):
    """Ground truth: the R change at its corpus position, mid-note."""
    return _line(sr_old, sr_old, sr_old, sr_new, **kw)


def test_release_fold_to_onset_after_gate_is_equivalent():
    """A mid-note R change relocated to its note's onset frame AFTER the gate
    write is equivalent across the R_old x R_new grid (measured worst 10 vs the
    16 cycle-accounting floor), in both prior-tail regimes (fast R_old: tail
    dead at onset; slow R_old: tail still live). The moment gate=1 the active
    compare is the decay nibble's and R is pure latent state -- this is the
    relocation that lets a tokenizer fold a note's release rate into its
    note-on, PROVIDED the decode emits gate-then-SR."""
    for r_old in (0x0, 0x2, 0x4, 0x8, 0xF):
        for r_new in (0x0, 0x2, 0x4, 0x8, 0xF):
            if r_old == r_new:
                continue
            mid, _ = _mid(0xF0 | r_old, 0xF0 | r_new)
            fold, _ = _line(0xF0 | r_old, 0xF0 | r_old, 0xF0 | r_new, 0xF0 | r_new)
            d = _diff(mid, fold)
            assert d <= EQUIVALENT_MAX_INT16_DELTA, (
                f"R {r_old:X}->{r_new:X}: mid-note vs onset-after-gate should "
                f"be equivalent, got {d}"
            )


def test_release_fold_before_gate_arms_the_adsr_bug_stall():
    """The same fold emitted BEFORE the gate write is NOT a chip-inert reorder:
    the inter-write gap is gate=0 dwell at the new compare. With the prior gap
    at R=0 (compare 9, counter pinned <9) a dwell at a larger compare carries
    the counter past attack rate 0's compare, and the equality comparison makes
    the attack wrap the full LFSR: ENV3 reads 00 at the onset frame where the
    faithful stream reads attacked (measured 4077). The hazard is two-sided --
    a change TO a small compare can equally UN-stall an authentic stall -- so
    gate-then-SR emission order is load-bearing for R-changing onset writes."""
    mid, env_mid = _mid(0xF0, 0xF8, onset_before_gate=True)
    fold, env_fold = _line(0xF0, 0xF0, 0xF8, 0xF8, onset_before_gate=True)
    assert env_mid[NOTE2_ON] > 0, "faithful stream attacks within the onset frame"
    assert env_fold[NOTE2_ON] == 0, "before-gate fold should stall the attack"
    d = _diff(mid, fold)
    assert d > AUDIBLE_MIN_INT16_DELTA, f"before-gate fold should be audible: {d}"


def test_release_change_before_gate_on_rerates_a_live_prior_tail():
    """With a slow prior release (R=0xA, ~1.5s full-scale) the tail is still
    releasing when a gap-relocated write lands: R is the ACTIVE rate, so the
    prior note's decay slope changes immediately -- the relocation audibly
    corrupts the note the write was never meant for."""
    mid, env_mid = _mid(0xFA, 0xF0)
    moved, _ = _line(0xFA, 0xF0, 0xF0, 0xF0)
    assert env_mid[GAP_SLOT] > 0, "prior tail must still be live at the gap slot"
    d = _diff(mid, moved)
    assert d > AUDIBLE_MIN_INT16_DELTA, f"live-tail re-rate should be audible: {d}"


def test_release_change_in_dead_gap_stalls_next_attack():
    """With a fast prior release (R=0x0, 6ms full-scale) the envelope is provably
    dead across the gap (ENV3=00) -- yet the relocation is STILL audible, one
    note later: the gap runs the prescaler against the new, larger compare, so
    at note2's gate-on the counter sits past attack rate 0's compare value and
    the attack stalls on the equality comparison (~a full frame: ENV3 reads 00
    where the faithful stream reads attacked). 'The note is no longer live' does
    not make R writes relocatable -- R owns ALL gate=0 time."""
    mid, env_mid = _mid(0xF0, 0xF8)
    moved, env_moved = _line(0xF0, 0xF8, 0xF8, 0xF8)
    dead = env_mid[NOTE1_OFF:NOTE2_ON] + env_moved[NOTE1_OFF:NOTE2_ON]
    assert all(e == 0 for e in dead), f"prior tail must be dead in the gap: {dead}"
    assert env_mid[NOTE2_ON] > 0, "faithful stream attacks within the onset frame"
    assert env_moved[NOTE2_ON] == 0, "gap-relocated stream's attack should stall"
    d = _diff(mid, moved)
    assert d > AUDIBLE_MIN_INT16_DELTA, f"attack stall should be audible: {d}"


def test_sustain_change_is_live_not_relocatable_to_onset():
    """The S nibble is the held amplitude -- per-voice volume -- while the gate
    is on. Moving a mid-note sustain drop (F->4, R held) to the onset slot makes
    the note sound at the wrong level for the frames in between: S writes own
    their timestamp and must remain standalone events."""
    mid, _ = _mid(0xF4, 0x44)
    fold, _ = _line(0xF4, 0xF4, 0x44, 0x44)
    d = _diff(mid, fold)
    assert d > AUDIBLE_MIN_INT16_DELTA, f"sustain relocation should be audible: {d}"
