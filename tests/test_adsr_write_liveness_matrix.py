"""The complete AD/SR write-liveness matrix: which nibble may be rewritten when,
without changing the audio. Pinned so canonicalization questions are answered
from the model, not by chasing corner cases one probe at a time.

THE MECHANISM (pinned against pyresidfp):

1. State machine. gate 0->1 -> ATTACK: the counter rises linearly at the A rate
   from wherever it is; at FF -> DECAY_SUSTAIN: falls at the D rate toward the
   sustain level S*0x11 (exponentially stretched at low counter values); gate
   1->0 -> RELEASE: falls at the R rate to 0 and clamps. No register write ever
   moves the counter or the prescaler -- writes only retarget comparisons.

2. Active prescaler compare. The 15-bit rate LFSR free-runs ALWAYS and is
   compared for EQUALITY against one nibble's rate period: A during ATTACK, D
   during DECAY_SUSTAIN (even while clamped at sustain), R whenever gate=0
   (even after the counter is dead at 0). Overshooting the compare costs a full
   ~33ms LFSR wrap -- the ADSR bug.

3. The sustain hold is an equality too: the counter holds while == S*0x11.
   RAISING S mid-sustain un-matches the hold; the counter resumes falling (it
   can never rise outside ATTACK), passes the unreachable new level, and pins
   at ZERO -- raising sustain kills the note at the D rate
   (``test_raising_sustain_kills_the_note``).

LIVENESS. A nibble write is relocatable within an interval iff the nibble is
neither VALUE-LIVE (currently shaping the counter) nor PHASE-LIVE (the active
compare: rewriting it re-phases the LFSR and can flip equality-stall states at
the next transition -- magnitude value-dependent) anywhere in that interval:

    phase    | A          | D                    | S                  | R
    ATTACK   | LIVE value | latent               | latent             | latent
    DECAY    | latent     | LIVE value           | live (clamp, eq.)  | latent
    SUSTAIN  | latent     | LIVE phase (val-dep) | LIVE (both dirs)   | latent
    RELEASE  | latent     | latent               | latent             | LIVE value
    DEAD     | latent     | latent               | latent             | LIVE phase

Tokenizer consequences: R writes relocate freely across gate=1 time (foldable
into note-on, emitted AFTER the gate write -- see
``test_release_write_position``); A writes relocate freely outside ATTACK
(consumed at the next gate-on); D and S writes own their timestamps whenever
the gate is on -- the mid-note AD/SR events that change them are NOT foldable.

Methodology: write-count-matched variants (same-value pads at the unused slot,
audibly bounded per ``test_sid_same_value_writes``), so chunk boundaries and
write cycle offsets are identical and diffs are purely semantic. Each line ends
with a second note so latent values ARE consumed downstream -- a latent verdict
means "the position is free", never "the value never mattered".

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

FREQ_LO, FREQ_HI, CTRL, AD, SR = 14, 15, 18, 19, 20  # voice 3: ENV3 readable
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


def _render(ad0, sr0, gates, slots, n_frames):
    """gates: {frame: ctrl byte}; slots: {frame: (reg, val)} -- the slot write
    goes before the frame's gate write. Returns (frames, env3 per frame end)."""
    sid = _make_sid()
    for reg, val in [
        (24, 0x0F),
        (FREQ_LO, 0x80),
        (FREQ_HI, 0x10),
        (AD, ad0),
        (SR, sr0),
    ]:
        sid.write_register(reg, val)
    frames, envs = [], []
    for f in range(n_frames):
        writes = []
        if f in slots:
            writes.append(slots[f])
        if f in gates:
            writes.append((CTRL, gates[f]))
        frames.append(_frame(sid, writes))
        envs.append(sid.read_register(ENV3))
    return frames, envs


def _cell(ad0, sr0, gates, reg, old_b, new_b, early, late, n_frames):
    """Max sample diff between the change at the window start vs the window end
    (write-count matched: the other slot holds a same-value rewrite)."""
    a, _ = _render(ad0, sr0, gates, {early: (reg, new_b), late: (reg, new_b)}, n_frames)
    b, _ = _render(ad0, sr0, gates, {early: (reg, old_b), late: (reg, new_b)}, n_frames)
    return max(
        int(np.abs(x.astype(np.int32) - y.astype(np.int32)).max())
        for x, y in zip(a, b, strict=False)
    )


def _assert_latent(d, what):
    assert d <= EQUIVALENT_MAX_INT16_DELTA, f"{what} should be relocatable: {d}"


def _assert_live(d, what):
    assert d > AUDIBLE_MIN_INT16_DELTA, f"{what} should own its timestamp: {d}"


# ATTACK spans f1..~f6 (A=8, 100ms); window f2..f4; off f14; 2nd note f18.
ATK = dict(ad0=0x84, sr0=0x84, gates={1: 0x21, 14: 0x20, 18: 0x21}, n_frames=22)


def test_attack_window_liveness():
    """During ATTACK only A is live (the rising slope re-rates at the write);
    D, S and R are latent -- consumed at the decay handoff / clamp / gate-off,
    so their position within the attack is free."""
    _assert_live(
        _cell(reg=AD, old_b=0x84, new_b=0x24, early=2, late=4, **ATK),
        "A 8->2 in attack",
    )
    for reg, old_b, new_b, what in [
        (AD, 0x84, 0x89, "D 4->9 in attack"),
        (AD, 0x84, 0x8F, "D 4->F in attack"),
        (SR, 0x84, 0x44, "S 8->4 in attack"),
        (SR, 0x84, 0xC4, "S 8->C in attack"),
        (SR, 0x84, 0x8A, "R 4->A in attack"),
        (SR, 0x84, 0x80, "R 4->0 in attack"),
    ]:
        _assert_latent(
            _cell(reg=reg, old_b=old_b, new_b=new_b, early=2, late=4, **ATK), what
        )


# DECAY: A=0 (instant attack), D=B (2.4s full-scale) falling toward S=2;
# window f3..f8 (counter still far above the clamp); off f16; 2nd note f19.
DEC = dict(ad0=0x0B, sr0=0x24, gates={1: 0x21, 16: 0x20, 19: 0x21}, n_frames=24)


def test_decay_window_liveness():
    """During DECAY only D is live (the falling slope re-rates at the write);
    A and R are latent. (S during decay is the equality clamp target -- its own
    pathology is pinned by ``test_raising_sustain_kills_the_note``.)"""
    _assert_live(
        _cell(reg=AD, old_b=0x0B, new_b=0x03, early=3, late=8, **DEC), "D B->3 in decay"
    )
    for reg, old_b, new_b, what in [
        (AD, 0x0B, 0xFB, "A 0->F in decay"),
        (AD, 0x0B, 0x8B, "A 0->8 in decay"),
        (SR, 0x24, 0x20, "R 4->0 in decay"),
        (SR, 0x24, 0x2B, "R 4->B in decay"),
    ]:
        _assert_latent(
            _cell(reg=reg, old_b=old_b, new_b=new_b, early=3, late=8, **DEC), what
        )


# SUSTAIN: A=0, D=0 -> clamps at S=8 within ~a frame; window f4..f10;
# off f14 (R=2); 2nd note f18.
SUS = dict(ad0=0x00, sr0=0x82, gates={1: 0x21, 14: 0x20, 18: 0x21}, n_frames=24)


def test_sustain_window_liveness():
    """During clamped SUSTAIN: S is live in BOTH directions (lower = re-decay to
    the new level now; raise = the equality hold un-matches and the note dies --
    either way the write's timestamp is audible). D is PHASE-live: the clamped
    counter never moves, but D is the active prescaler compare, so moving the
    write re-phases the LFSR and shifts the release's first step at gate-off --
    magnitude is value-dependent (D 0->F audible here; 0->8 measured at the
    cycle-accounting floor), so D mid-note writes are NOT relocatable in
    general. A and R are latent."""
    _assert_live(
        _cell(reg=SR, old_b=0x82, new_b=0x32, early=4, late=10, **SUS),
        "S 8->3 (lower) in sustain",
    )
    _assert_live(
        _cell(reg=SR, old_b=0x82, new_b=0xC2, early=4, late=10, **SUS),
        "S 8->C (raise) in sustain",
    )
    _assert_live(
        _cell(reg=AD, old_b=0x00, new_b=0x0F, early=4, late=10, **SUS),
        "D 0->F in sustain (phase)",
    )
    for reg, old_b, new_b, what in [
        (AD, 0x00, 0xF0, "A 0->F in sustain"),
        (AD, 0x00, 0x80, "A 0->8 in sustain"),
        (SR, 0x82, 0x89, "R 2->9 in sustain"),
        (SR, 0x82, 0x80, "R 2->0 in sustain"),
    ]:
        _assert_latent(
            _cell(reg=reg, old_b=old_b, new_b=new_b, early=4, late=10, **SUS), what
        )


# RELEASE: S=8 held, off f4 at R=B (2.4s -- tail spans the window f6..f12);
# 2nd note f16.
REL = dict(ad0=0x00, sr0=0x8B, gates={1: 0x21, 4: 0x20, 16: 0x21}, n_frames=24)


def test_release_window_liveness():
    """During RELEASE only R is live (the tail re-rates at the write); A, D and
    S are latent -- consumed at the second note."""
    _assert_live(
        _cell(reg=SR, old_b=0x8B, new_b=0x82, early=6, late=12, **REL),
        "R B->2 in release",
    )
    for reg, old_b, new_b, what in [
        (AD, 0x00, 0xF0, "A 0->F in release"),
        (AD, 0x00, 0x09, "D 0->9 in release"),
        (AD, 0x00, 0x0F, "D 0->F in release"),
        (SR, 0x8B, 0x3B, "S 8->3 in release"),
    ]:
        _assert_latent(
            _cell(reg=reg, old_b=old_b, new_b=new_b, early=6, late=12, **REL), what
        )


# DEAD: off f4 at R=0 (6ms -- counter at 0 throughout the window f6..f12);
# 2nd note f16.
DEAD = dict(ad0=0x00, sr0=0x80, gates={1: 0x21, 4: 0x20, 16: 0x21}, n_frames=24)


def test_dead_window_liveness():
    """After the release has run to zero: A, D and S are latent, but R remains
    PHASE-live -- the dead counter cannot move, yet R is still the active
    compare, so moving the write re-phases the LFSR (above the cycle-accounting
    floor here; the audibly-stalled next attack is pinned by
    ``test_release_write_position.test_release_change_in_dead_gap_stalls_next_
    attack``)."""
    d = _cell(reg=SR, old_b=0x80, new_b=0x88, early=6, late=12, **DEAD)
    assert d > EQUIVALENT_MAX_INT16_DELTA, f"R in the dead zone is phase-live: {d}"
    for reg, old_b, new_b, what in [
        (AD, 0x00, 0xF0, "A 0->F in dead zone"),
        (AD, 0x00, 0x09, "D 0->9 in dead zone"),
        (SR, 0x80, 0x30, "S 8->3 in dead zone"),
    ]:
        _assert_latent(
            _cell(reg=reg, old_b=old_b, new_b=new_b, early=6, late=12, **DEAD), what
        )


def test_raising_sustain_kills_the_note():
    """The sustain hold is ``counter == S*0x11``, not >=. Raising S above the
    clamped counter un-matches the hold; the counter resumes the only direction
    it has outside ATTACK -- down -- sails past the unreachable new level and
    pins at ZERO at the D rate. ENV3: clamped 88 ... raise ... 00."""
    gates = {1: 0x21}
    _, env_raised = _render(0x00, 0x82, gates, {5: (SR, 0xC2)}, 12)
    _, env_held = _render(0x00, 0x82, gates, {5: (SR, 0x82)}, 12)
    assert env_held[4] == 0x88 and env_held[11] == 0x88, "control holds the clamp"
    assert env_raised[4] == 0x88, "clamped before the raise"
    assert env_raised[11] == 0x00, f"raise should kill the note: {env_raised}"
