"""The canonical gate/ADSR programming reference, as unit tests. Companion to
``test_adsr_write_liveness_matrix`` (write placement within phases) and
``test_release_write_position`` (the R-fold placement rules); this file pins the
ADSR (envelope delay) bug itself and the gate-edge mechanics built on it.

THE BUG, EXACTLY. The rate prescaler is a free-running 15-bit LFSR compared for
EQUALITY against the active nibble's rate period, and every envelope step resets
it. The bug is COMPARE-CHANGE associated, NOT gate associated: the running
phase stalls a full LFSR wrap (~33ms) whenever the ACTIVE compare drops below
the current count -- from either trigger:

1. INTERNAL handoffs never stall. The FF->decay handoff and the arrival at the
   sustain clamp coincide with an envelope step, so the LFSR is freshly reset
   and cannot already sit past the next compare
   (``test_phase_handoffs_never_stall``).

2. A WRITE to the active nibble reproduces the bug with NO gate edge anywhere:
   lowering the active compare freezes the running phase in attack, decay and
   release alike (``test_write_alone_freezes_attack/decay/release``). The bug
   is ONE-DIRECTIONAL -- raising the active compare never stalls, because the
   count is necessarily below the new, larger compare
   (``test_raising_the_active_compare_never_stalls``).

3. A GATE EDGE swaps WHICH nibble is the active compare, hitting the same
   mechanism with the old epoch's count:
   - attack stall at gate-on: armed iff count > A's compare at the edge; the
     count is the phase of the prior gate=0 epoch (cycling mod R's compare).
     Hard restart is the driver-side fix: force that epoch's compare small
     (HR-SR=0) so the count is pinned below any attack compare
     (``test_attack_stall_armed_by_prior_gap_compare``).
   - release stall at gate-off: armed iff count > R's compare; the count
     cycles mod D's compare while the gate is on, so large D arms a held-level
     hang and D=0 disarms (``test_release_stall_armed_by_decay_compare``).

4. EDGE POSITION IS CONTENT at single-write granularity: inserting even a
   SAME-VALUE write before a CTRL edge write delays the edge by one inter-write
   slot (~32 cycles), re-phases the LFSR, and can flip a later attack stall
   outright (``test_gate_edge_position_is_content``). Same-value-write removal
   is only safe when the surviving writes keep their recorded clocks -- never
   re-slot writes around gate edges.

5. EDGE-ADJACENT WRITE ORDER (edge slot held fixed): at gate-on all four
   nibbles may go immediately before or after the gate write (<= 1 envelope
   step) -- except R before the gate when A's compare is small, the
   before-gate hazard pinned in ``test_release_write_position``. At gate-off A
   and R are free, S leads by the few steps it decays early (real,
   sub-audible), and D's slot is CONTENT: the dwell at one compare vs the
   other diverges the LFSR phase and flips a downstream attack stall
   (``test_gate_on_edge_adjacent_order_is_free``,
   ``test_gate_off_edge_adjacent_order``).

6. GATE SEMANTICS: a CTRL write with the gate already 1 is not a retrigger
   (the envelope responds to edges, not levels); a re-attack resumes from the
   current counter, so the prior note's state is audible in the next attack
   (and the re-attack itself stalls per #3 if the gap compare was large); a
   short gate pulse yields a partial attack then release
   (``test_gate_rewrite_while_on_is_not_a_retrigger``,
   ``test_reattack_resumes_from_current_counter``,
   ``test_short_gate_pulse_releases_partial_attack``).

Methodology: write-count matched variants throughout (same-value pads), ENV3 on
voice 3 for state-level verdicts. Proven against pyresidfp.
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
ONE_ENV_STEP_MAX_INT16_DELTA = 128  # one envelope step at high counter/volume

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


def _run(ad0, sr0, sched, n_frames):
    """sched: {frame: [(reg, val), ...]} applied in order before the frame's
    remainder clocks out. Returns (per-frame audio, per-frame ENV3)."""
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
        frames.append(_frame(sid, sched.get(f, [])))
        envs.append(sid.read_register(ENV3))
    return frames, envs


def _diff(a_frames, b_frames):
    return max(
        int(np.abs(x.astype(np.int32) - y.astype(np.int32)).max())
        for x, y in zip(a_frames, b_frames, strict=False)
    )


def test_phase_handoffs_never_stall():
    """Internal phase handoffs cannot hit the ADSR bug: the FF->decay handoff
    coincides with an envelope step, which has just reset the rate LFSR, so the
    decay compare cannot already be overshot. A=9 attack (~12 frames) into D=4
    decay: the envelope never reads FF at a frame boundary and declines
    immediately after its peak -- a stall would hang ~1.6 frames at FF."""
    _, env = _run(0x94, 0x00, {1: [(CTRL, 0x21)]}, 20)
    peak = max(range(20), key=lambda f: env[f])
    assert env[peak] >= 0xF0, f"attack should near full scale: {env[peak]:02X}"
    assert sum(1 for e in env if e == 0xFF) <= 1, f"no FF hang: {env}"
    assert env[peak + 1] < env[peak], "decay must begin within a frame of the peak"


def test_attack_stall_armed_by_prior_gap_compare():
    """The attack stall is armed by the PRIOR gate=0 epoch's compare, not by
    anything audible: both variants here have a provably dead gap (ENV3=00),
    differing only in a mid-gap SR write -- a same-value pad (gap stays at R=0,
    compare 9, count pinned < 9 = the hard-restart principle) vs R=F (large
    compare, count grows past A=0's compare of 9). The armed attack starts a
    stall late; ENV3's frame-end sampling can hide a sub-frame stall (both read
    FF here) but the audio cannot. The frame-scale version of the same arming
    is ``test_release_write_position.test_release_change_in_dead_gap...``."""
    sched = {1: [(CTRL, 0x21)], 3: [(CTRL, 0x20)], 12: [(CTRL, 0x21)]}
    disarmed = {**sched, 6: [(SR, 0xF0)]}  # same-value pad
    armed = {**sched, 6: [(SR, 0xFF)]}  # R=F: re-comparate the dead gap
    a, env_a = _run(0x00, 0xF0, disarmed, 16)
    b, env_b = _run(0x00, 0xF0, armed, 16)
    gap = env_a[4:12] + env_b[4:12]
    assert all(e == 0 for e in gap), f"gap must be dead in both: {gap}"
    d = _diff(a, b)
    assert d > AUDIBLE_MIN_INT16_DELTA, f"armed attack should start audibly late: {d}"


def test_release_stall_armed_by_decay_compare():
    """The release stall is armed by D, the compare the LFSR cycled against
    while the gate was on. D=0 (compare 9): the count is < 9 at gate-off, never
    past R=0's compare -- the envelope releases to 0 within the off frame. D=F
    (large compare): the count sits far past it -- the envelope HANGS at its
    held level through the entire off frame before the first release step."""
    on_off = {1: [(CTRL, 0x21)], 6: [(CTRL, 0x20)]}
    _, env_fast = _run(0x00, 0x80, on_off, 12)  # D=0
    _, env_hang = _run(0x0F, 0x80, on_off, 12)  # D=F
    assert env_fast[6] == 0x00, f"D=0: released within the off frame: {env_fast}"
    assert env_hang[6] == env_hang[5] != 0x00, f"D=F: held-level hang: {env_hang}"
    assert env_hang[7] == 0x00, f"D=F: release completes next frame: {env_hang}"


def test_gate_edge_position_is_content():
    """Delaying a gate edge by ONE inter-write slot (~32 cycles) -- here by a
    same-value SR write before the second note's gate-on -- re-phases the LFSR
    mod the gap's compare and flips that note's attack stall: clean attack
    (decayed to the 88 clamp by frame end) vs a full stalled frame (00). The
    write-count differs by construction, so the verdict is ENV3 state, not an
    audio diff. Encoder consequence: never re-slot writes around gate edges;
    dropping same-value writes is safe only because the renderer keeps the
    surviving writes' recorded clocks."""
    plain = {1: [(CTRL, 0x21)], 10: [(CTRL, 0x20)], 16: [(CTRL, 0x21)]}
    padded = {1: [(CTRL, 0x21)], 10: [(CTRL, 0x20)], 16: [(SR, 0x82), (CTRL, 0x21)]}
    _, env_plain = _run(0x00, 0x82, plain, 22)
    _, env_padded = _run(0x00, 0x82, padded, 22)
    assert env_plain[16] == 0x88, f"unshifted edge attacks cleanly: {env_plain}"
    assert env_padded[16] == 0x00, f"shifted edge stalls the attack: {env_padded}"
    assert (
        env_padded[17] == 0x88
    ), f"stalled attack completes a frame late: {env_padded}"


# Edge-adjacent ordering cells: 3 writes, the CTRL edge ALWAYS at slot 1, the
# nibble's new value crossing the edge (before: change at slot 0, same-value at
# slot 2; after: same-value old at slot 0, change at slot 2).
def _edge_order_cell(ad0, sr0, sched, edge_f, ctrl_val, reg, old_b, new_b, n_frames):
    before = {**sched, edge_f: [(reg, new_b), (CTRL, ctrl_val), (reg, new_b)]}
    after = {**sched, edge_f: [(reg, old_b), (CTRL, ctrl_val), (reg, new_b)]}
    a = _run(ad0, sr0, before, n_frames)
    b = _run(ad0, sr0, after, n_frames)
    return _diff(a[0], b[0]), a[1], b[1]


def test_gate_on_edge_adjacent_order_is_free():
    """With the edge slot held fixed, every nibble may be written immediately
    before or after the gate-on write: at most one envelope step of drift. (The
    exception is R before the gate when A's compare is SMALL -- the before-gate
    dwell hazard pinned in ``test_release_write_position``; here A=8's compare
    dwarfs the dwell.)"""
    sched = {1: [(CTRL, 0x21)], 5: [(CTRL, 0x20)]}
    for reg, old_b, new_b, what in [
        (AD, 0x84, 0x24, "A 8->2"),
        (AD, 0x84, 0x8F, "D 4->F"),
        (SR, 0x80, 0x30, "S 8->3"),
        (SR, 0x80, 0x8A, "R 0->A"),
    ]:
        d, _, _ = _edge_order_cell(0x84, 0x80, sched, 12, 0x21, reg, old_b, new_b, 22)
        assert d <= ONE_ENV_STEP_MAX_INT16_DELTA, f"{what} across gate-on: {d}"


def test_gate_off_edge_adjacent_order():
    """At gate-off (edge slot fixed): A and R cross freely; S leads by the few
    steps it decays toward the new clamp before the edge (real but sub-audible
    here); D's slot is CONTENT -- one write slot of dwell at the old vs new
    compare diverges the LFSR phase (a reset fires in one variant only) and
    flips the SECOND note's attack stall outright (env FF vs 00)."""
    sched = {1: [(CTRL, 0x21)], 16: [(CTRL, 0x21)]}
    for reg, old_b, new_b, what in [
        (AD, 0x00, 0xF0, "A 0->F"),
        (SR, 0x82, 0x8A, "R 2->A"),
    ]:
        d, _, _ = _edge_order_cell(0x00, 0x82, sched, 10, 0x20, reg, old_b, new_b, 22)
        assert d <= EQUIVALENT_MAX_INT16_DELTA, f"{what} across gate-off: {d}"
    for old_b, new_b, what in [(0x82, 0x32, "S 8->3"), (0x82, 0xC2, "S 8->C")]:
        d, _, _ = _edge_order_cell(0x00, 0x82, sched, 10, 0x20, SR, old_b, new_b, 22)
        assert (
            EQUIVALENT_MAX_INT16_DELTA < d < AUDIBLE_MIN_INT16_DELTA
        ), f"{what} across gate-off should be a real but small lead: {d}"
    d, env_b, env_a = _edge_order_cell(0x00, 0x82, sched, 10, 0x20, AD, 0x00, 0x0F, 22)
    assert d > AUDIBLE_MIN_INT16_DELTA, f"D 0->F slot is content: {d}"
    assert (env_b[16] == 0x00) != (
        env_a[16] == 0x00
    ), f"the divergence is the 2nd note's stall flip: {env_b[16]:02X}/{env_a[16]:02X}"


def test_gate_rewrite_while_on_is_not_a_retrigger():
    """The envelope responds to gate EDGES, not levels: rewriting CTRL with the
    gate still 1 -- same value or a waveform change -- does not re-attack. (A
    'gate-on' mid-note is a contradiction in terms; only 1->0->1 retriggers,
    which is what hard restart choreographs.)"""
    sched = {1: [(CTRL, 0x21)], 6: [(CTRL, 0x21)], 7: [(CTRL, 0x25)]}
    _, env = _run(0x00, 0x82, sched, 12)
    assert all(e == 0x88 for e in env[5:10]), f"clamp must hold through rewrites: {env}"


def test_reattack_resumes_from_current_counter():
    """gate-on attacks from wherever the counter is -- there is no reset -- so
    the prior note's release depth is audible in the next attack: a 2-frame vs
    8-frame release gap leaves different counters, and both re-attacks rise
    from exactly those levels. (Both first re-attack frames also show the
    stall of #2: the gap ran at R=A's large compare.)"""
    levels = {}
    for gap in (2, 8):
        sched = {1: [(CTRL, 0x21)], 4: [(CTRL, 0x20)], 4 + gap: [(CTRL, 0x21)]}
        _, env = _run(0x84, 0xFA, sched, 18)
        f = 4 + gap
        assert env[f + 1] > env[f - 1], f"re-attack must rise from the tail: {env}"
        assert env[f + 2] > env[f + 1], f"and keep rising: {env}"
        levels[gap] = env[f - 1]
    assert levels[2] > levels[8] > 0, f"longer release -> deeper start: {levels}"


def test_short_gate_pulse_releases_partial_attack():
    """A 1-frame gate pulse attacks partway (A=8 reaches ~0x32 in one frame)
    and then releases from that partial level -- gates need not outlive their
    attacks, and the envelope carries the partial state forward."""
    sched = {1: [(CTRL, 0x21)], 2: [(CTRL, 0x20)]}
    _, env = _run(0x88, 0x8A, sched, 10)
    assert 0x00 < env[1] < 0x80, f"partial attack: {env}"
    assert env[3] < env[2] <= env[1], f"then release: {env}"


def test_write_alone_freezes_attack():
    """The bug with NO gate edge: mid-attack at A=B (3.1ms/step, count cycling
    far above 9), writing A->0 drops the active compare below the count -- a
    no-bug envelope would complete the 2ms attack within the write frame, but
    the phase FREEZES through it (~33ms wrap) and only then snaps to full scale
    and the sustain clamp."""
    _, env = _run(0xB4, 0x80, {1: [(CTRL, 0x21)], 6: [(AD, 0x04)]}, 12)
    assert env[5] > env[4] > 0, f"attack must be rising pre-write: {env}"
    assert env[6] == env[5], f"write frame must freeze, not finish in 2ms: {env}"
    assert env[8] == 0x88, f"then attack completes and clamps at S=8: {env}"


def test_write_alone_freezes_decay():
    """Mid-decay at D=B, writing D->0 (compare 9, far below the count): a
    no-bug envelope would crash to sustain in 6ms, but the decay FREEZES
    through the write frame, then crashes after the wrap."""
    _, env = _run(0x0B, 0x00, {1: [(CTRL, 0x21)], 8: [(AD, 0x00)]}, 14)
    assert 0 < env[7] < env[6], f"decay must be falling pre-write: {env}"
    assert env[8] == env[7], f"write frame must freeze, not crash in 6ms: {env}"
    assert env[9] == 0x00, f"then the crash to sustain completes: {env}"


def test_write_alone_freezes_release():
    """Mid-release at R=B, writing R->0: the no-bug 6ms crash to zero instead
    FREEZES through the write frame -- the same mechanism as the gate-off
    release stall, with no gate edge involved."""
    sched = {1: [(CTRL, 0x21)], 4: [(CTRL, 0x20)], 8: [(SR, 0xF0)]}
    _, env = _run(0x00, 0xFB, sched, 14)
    assert 0 < env[7] < env[6], f"release must be falling pre-write: {env}"
    assert env[8] == env[7], f"write frame must freeze, not die in 6ms: {env}"
    assert env[9] == 0x00, f"then the release completes: {env}"


def test_raising_the_active_compare_never_stalls():
    """The bug is one-directional: RAISING the active compare cannot stall,
    because the count is necessarily below the new, larger compare. Mid-decay
    D=4 -> D=B: the envelope keeps stepping down every frame -- merely slower
    -- with no freeze and no wrap."""
    _, env = _run(0x04, 0x00, {1: [(CTRL, 0x21)], 3: [(AD, 0x0B)]}, 12)
    assert all(env[f + 1] < env[f] for f in range(3, 10)), f"no freeze: {env}"
    assert env[10] > 0, f"and no crash -- just the slower rate: {env}"
