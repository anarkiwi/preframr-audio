# preframr-audio

SID audio rendering, render-fidelity gates, and audio fingerprinting for the
[preframr](https://github.com/anarkiwi/preframr) research codebase, built on
[pyresidfp](https://github.com/pyresidfp/pyresidfp). Published standalone so the
audio path installs without the training stack.

This README is the API reference for the package **and** the reference for
programming the SID through it: the chip-behavior facts the renderer and the
fidelity gates rely on are each pinned by a unit test in `tests/` (see
[SID programming facts](#sid-programming-facts-backed-by-unit-tests)).

## Install

```bash
pip install preframr-audio
```

System dependencies: ALSA + `libasound2` for the real-time driver (offline
rendering to WAV/PCM works without ALSA).

## Import contract

Everything public is re-exported from the package root
(`from preframr_audio import compare_renders`). Resolution is lazy (PEP 562):
`import preframr_audio` does not pull in `pyresidfp`/`scipy` — those load on
first access of a render-backed symbol, so the render-free path
(`compare_renders`, `mel_features`, …) works without them.

| Module | Role |
|---|---|
| `sidwav` | Chip construction + single register writes: `default_sid`, `sidq`, `write_reg` |
| `audio_driver` | Render pipeline: `render_to_samples`, `render_to_wav`, `render_per_voice`, `df_to_packets`, `FramePacket`/`FrameOp`, `AudioRenderBuffer`, `ResidWorker`, real-time drivers, ASID SysEx codec |
| `fidelity` | Equivalence gates: strict per-frame RMS (`compare_renders`, `dfs_render_equivalent`) and perceptual (`perceptual_distance`, `dfs_perceptually_equivalent`) |
| `features` | Deterministic audio feature extractors (`mel_features`, `spectral_features`, `band_power_features`, `raw_pcm_features`) |
| `fingerprint` | Render + featurize register-write sequences (`canonical_scaffold`, `fingerprint_writes`, `fingerprint_batch`) |
| `batch` | Process-parallel render / verify over many dfs (`render_batch`, `verify_equivalent_batch`, `verify_perceptually_equivalent_batch`) |
| `live_animator` | VT100 terminal voice-state visualiser (`LiveAnimator`, `VoiceState`) |

## The SID register model

Registers are addressed 0..24 (`MAX_REG = 24`). Per voice `v` (0..2) the base is
`v * 7` (`VOICE_REG_SIZE = 7`):

| Offset | Register |
|---|---|
| +0 | freq lo |
| +1 | freq hi |
| +2 | pulse width lo |
| +3 | pulse width hi (low nibble) |
| +4 | control: bit0 = GATE, bit1 = SYNC, bit2 = RING, bit3 = TEST, bits4–7 = waveform |
| +5 | AD (attack/decay nibbles) |
| +6 | SR (sustain/release nibbles) |

Globals: 21 = filter cutoff lo (`FC_LO_REG`), 22 = cutoff hi, 23 =
resonance/filter routing, 24 = mode/volume (`MODE_VOL_REG`).
`VOICE_CTRL_REG = {0: 4, 1: 11, 2: 18}`; `voice_of_reg(reg)` maps a register to
its owning voice (None for globals/markers). The 16-bit frequency is always
`(reg[v*7+1] << 8) | reg[v*7]` — never read a lo or hi byte in isolation.

### Input DataFrame and frame timing

The render entry points consume a *prepared-for-audio* DataFrame (produced by
`preframr_tokens.prepare_df_for_audio`: literal `reg`/`val` writes plus marker
rows, with a `delay` column in seconds). Two negative marker registers drive
frame packetization:

- `FRAME_REG = -128` — frame boundary; its delay carries the frame remainder.
- `DELAY_REG = -127` — multi-frame gap (silence packets).

A PAL frame is `19656` clock cycles (~50.12 Hz) at `clock_frequency = 985248`;
`sidq(sid)` returns seconds-per-cycle (`1 / sid.clock_frequency`). Each register
write is clocked individually with its own `delay_cycles` dwell
(`FrameOp(reg, val, delay_cycles)`) — **per-write clocking is mandatory**:
collapsed timing (clock once per frame) masks write-placement effects that are
audible on the chip (see the facts table). Negative `delay_cycles` offsets
prior positive dwell in the same frame rather than being clamped
(`test_audio_driver.py::TestNegativeDelayCarry`).

## Render pipeline

```
prepared df ──df_to_packets──> FramePacket stream ──ResidWorker──> int16 PCM ──> sink
                                                       (pyresidfp)        WavSampleSink / SampleRing / ALSA / ASID
```

| Function | Signature (key args) | Returns |
|---|---|---|
| `render_to_samples` | `(df, reg_widths, irq, cents=50, reg_start=None, chip_model="MOS8580", mute_voices=())` | `(np.int16 samples, sample_rate)` |
| `render_to_wav` | `(df, wav_path, reg_widths, irq, …, mute_voices=())` | sample count, writes WAV via scipy |
| `render_per_voice` | `(df, reg_widths, irq, …)` | `{voice: samples}` — each voice soloed; other voices' outputs muted but their **oscillators keep running**, so SYNC/RING coupling from a muted modulator is preserved (`test_render_per_voice_captures_sync_ring_modulator`) |
| `df_to_packets` | `(df, reg_widths, freq_mapper, irq_cycles=19656, clock_frequency=985248, frame_id_start=0)` | generator of `FramePacket`; `FRAME_REG`/`DELAY_REG` rows close frames, unknown regs are skipped, a trailing partial frame is still yielded |
| `write_reg` | `(sid, reg, val, reg_widths)` | applies one logical write; freq regs 0/7/14 map through `FreqMapper.if_map` and expand to lo+hi byte pairs, PW regs 2/9/16 and `FC_LO` 21 are 2-byte, all clamped to the legal key range |

`reg_start` (default: max sustain, 50% PWM, volume 15) primes the chip before
the first frame. `mute_voices` zeroes the listed voices' control output while
keeping their oscillators running.

Real-time sinks (not part of the offline API, `pragma: no cover`):
`ResidAlsaDriver` (PortAudio), `AsidMidiDriver`/`AsidWorker` (delta-encoded
ASID SysEx over MIDI; only changed registers are sent per frame), `AsidServer`
(receives ASID, renders on a local SID for round-trip validation), and
`play_via_aplay` / `play_via_wav` subprocess players. The ASID codec itself is
pure and tested: `encode_asid_update({reg: val}) -> sysex_body`,
`decode_asid_update(body) -> {reg: val}`, bijective over all 25 registers.

`AudioRenderBuffer(max_frames=200)` is the thread-safe FIFO between producer
and renderer: `push_frame` (blocks when full), `pop_frame`, and
`replace_frame(frame_id, ops)` which edits a still-queued frame and returns
False once the playhead has passed it.

## Fidelity gates

Two levels, both rendering through the same pipeline:

**Strict (per-frame RMS).** `compare_renders(a, b, sample_rate, tolerance=0.05,
max_frame_drift=1)` aligns by cross-correlation lag and compares per-PAL-frame
relative RMS, returning a frozen `AudioFidelityResult(passed, shape, diagnostic,
lag_samples, worst_frame_rel_rms, n_frames_compared)`. `shape` classifies the
divergence (`PASS`, `DURATION_MISMATCH`, `FRAME_CADENCE_BREAK`,
`INITIAL_STATE_DIVERGENCE`, `DRIFTING_DIVERGENCE`, `CONSTANT_DIVERGENCE`,
`SAMPLE_RATE_MISMATCH`, `NO_INFO`). `per_frame_rel_rms` returns the full
per-frame timeline to pinpoint divergent frames;
`compare_renders_per_voice` runs the gate over `render_per_voice` maps.
`dfs_render_equivalent(df_a, df_b)` renders both dfs in memory and compares;
`assert_dfs_render_equivalent` is the raising variant (renders via
`render_df_to_wav`, which requires `preframr-tokens` for
`prepare_df_for_audio`). Tolerance floor: `FRAME_RMS_TOLERANCE = 0.05`.

**Perceptual.** `perceptual_distance(a, b, sample_rate, win_s=1.0)` returns the
worst-window distance plus the per-window array; per window the metric is a
32-band linear band-power log-envelope (`band_power_features`) plus scaled
`spectral_features`. Linear bands (not mel) are deliberate: a sub-band-width
detune (≤50 cents) is invisible, while waveform/filter/envelope/voice-drop
changes are large — calibrated INERT/BAD separation ≥ 2× around
`PERCEPTUAL_DISTANCE_THRESHOLD = 0.257`
(`tests/test_perceptual_fidelity.py::test_separation_margin`; reproduce with
`python -m preframr_audio.fidelity`, the `calibrate()` entry point).
`dfs_perceptually_equivalent` / `assert_dfs_perceptually_equivalent` are the
df-level gate and raising variant, returning `PerceptualFidelityResult`.

Use the strict gate for must-be-inaudible rewrites (augmentation, tokenizer
canonicalization A/B); use the perceptual gate for lossy-tier changes that may
move samples but must not move the music.

## Features and fingerprinting

`features` extractors are pure numpy, deterministic, fixed-length:
`mel_features` (time-pooled log-mel mean+std, `(2*n_mels,)`),
`spectral_features` (centroid, bandwidth, rolloff-95, ZCR, RMS),
`band_power_features` (32 equal-width linear bands 0–8 kHz, pitch-tolerant),
`raw_pcm_features`. Registry: `FEATURE_FNS`.

`fingerprint_writes(test_writes, irq=19656, feature="mel", …)` wraps raw
`(clock, reg, val)` writes in `canonical_scaffold` (chip init + a default
3-voice triad + silence pre/post frames, irq stored in `df.attrs`), renders,
and returns the feature vector. `fingerprint_batch(sequences, n_workers=-1)`
parallelizes over a spawn pool and returns the stacked `(N, dim)` matrix
(parallel == sequential within the chip's nondeterminism floor,
`test_fingerprint.py`).

`batch` mirrors this for whole dfs: `render_batch`, `verify_equivalent_batch`,
`verify_perceptually_equivalent_batch` — all order-preserving, workers return
only the small result objects.

## SID programming facts (backed by unit tests)

The renderer and every canonicalization decision in the tokenizer rest on the
following chip behaviors, each pinned by a deterministic pyresidfp test in this
repo. The test is the citation; if a fact needs revisiting, change the test.

### Envelope (ADSR) mechanism — `tests/test_gate_adsr_reference.py`

The envelope rate prescaler is a free-running 15-bit LFSR compared for
**equality** against the active nibble's rate period; every envelope step
resets it. Consequences, each a test:

| Fact | Test |
|---|---|
| Writing a smaller period to the **active** nibble freezes the running phase ~33 ms with no gate edge involved (attack, decay and release alike) | `test_write_alone_freezes_attack/decay/release` |
| The bug is one-directional: raising the active compare never stalls | `test_raising_the_active_compare_never_stalls` |
| Internal phase handoffs (FF→decay, sustain-clamp arrival) never stall; stalls are compare-change-associated, not gate-associated | `test_phase_handoffs_never_stall` |
| A large release compare left over from a prior gate epoch arms the **next attack's** stall (this is what hard restart works around) | `test_attack_stall_armed_by_prior_gap_compare` |
| A large decay compare at gate-off hangs the release at the held level | `test_release_stall_armed_by_decay_compare` |
| Gate-edge **position** is content even at single-write granularity: a same-value write delaying the edge ~32 cycles can flip a stall | `test_gate_edge_position_is_content` |
| At gate-ON, adjacent nibble order is free (within one envelope step) | `test_gate_on_edge_adjacent_order_is_free` |
| At gate-OFF, D's side of the edge is content (flips the next attack); A and R are free; S is a small real lead | `test_gate_off_edge_adjacent_order` |
| Gate 1→1 rewrite is not a retrigger; re-attack resumes from the current counter; a short gate pulse releases a partial attack | `test_gate_rewrite_while_on_is_not_a_retrigger`, `test_reattack_resumes_from_current_counter`, `test_short_gate_pulse_releases_partial_attack` |

### Write-liveness matrix — `tests/test_adsr_write_liveness_matrix.py`

Which ADSR nibble writes are audible in which envelope phase (a write is
relocatable iff neither value-live nor phase-live):

| Phase | Live nibbles | Test |
|---|---|---|
| Attack | A only | `test_attack_window_liveness` |
| Decay | D only | `test_decay_window_liveness` |
| Sustain | S (both directions; D is phase-live) | `test_sustain_window_liveness` |
| Release | R only | `test_release_window_liveness` |
| Dead (counter 0, gate 0) | R is still phase-live | `test_dead_window_liveness` |

Sustain hold is itself an equality compare: **raising S above the clamped
counter kills the note** (`test_raising_sustain_kills_the_note`).

### Release-write placement — `tests/test_release_write_position.py`

R-nibble writes are foldable to the note onset **across a gate=1 span**, never
across gate=0: a live release tail re-rates audibly, and even over a provably
dead gap the changed compare stalls the next attack. S is live, not
relocatable.

### Write order and audibility — `tests/test_register_canonicalization.py`, `tests/test_freq_write_audibility.py`

| Fact | Test |
|---|---|
| Intra-frame write order is audible (gate before its freq → wrong attack pitch) | `test_intra_frame_write_order_is_audible` |
| Interleaved ADSR/CTRL order is audible (~17% of single-speed HVSC tunes have such frames) — never reg-sort within a voice | `test_interleaved_adsr_ctrl_order_is_audible` |
| Multiple CTRL writes per frame each take effect (TEST pulses, gate toggles) | `test_intra_frame_gate_toggles_take_effect` |
| A re-gated attack depends on prior envelope state | `test_adsr_bug_attack_depends_on_prior_envelope_state` |
| On a TEST-bit frame, PW is audible but freq is a near-don't-care; waveform bits are NOT don't-care (held DC level) | `test_test_bit_frame_pw_is_audible_but_freq_is_not`, `test_waveform_bits_during_test_are_NOT_dont_care` |
| RING with a silent source silences (not a no-op); SYNC with a non-oscillating source IS a no-op | `test_ring_with_a_silent_source_silences_not_noop`, `test_sync_with_a_non_oscillating_source_is_a_noop` |
| Release is not instant (R=0 still sounds for frames); freq changes during release are audible; noise freq is audible; combined-waveform freq is audible | `test_freq_write_audibility.py` |
| Pre-first-gate freq writes are inaudible in the pre-gate window, droppable **only when the first note hard-restarts via TEST** | `test_freq_write_before_first_gate_on_is_inaudible_in_the_pre_gate_window`, `test_pre_gate_freq_droppable_only_when_first_note_hard_restarts` |
| Same-value rewrites are chip no-ops within a small DC floor | `tests/test_sid_same_value_writes.py` |
| Sustain nibble = held amplitude (per-voice volume); filter routes/attenuates; test bit resets oscillator phase; SID Wizard hard-restart reduces prior-state dependence | `tests/test_sid_feature_behaviors.py` |

Two **distinct** "hard restart" mechanisms — don't conflate them:

1. **ADSR/envelope hard restart** (gate-based): force the prior epoch's compare
   small ~2 frames before the next gate-on so the ADSR-bug window elapses. No
   TEST bit involved; freq stays audible during those release frames.
2. **TEST-bit oscillator reset** (CTRL bit 3): silences and resets the
   oscillator; freq on the TEST frame is the only near-don't-care.

Players also exploit the bug deliberately ("sexy start": AD+SR written before
GATE with big R/small A to arm the ~32 ms attack delay) — write placement
around the gate edge is musical content, which is why the tokenizer records
gate-edge side per nibble instead of normalizing it.

### Scope

These tests and the renderer target single-speed, non-digi tunes (one player
call per IRQ frame). Multi-speed (~5%) and digi (~3%) playback are out of
scope.

## Testing

```bash
./run_tests.sh   # black, pylint, pyright, pytest with coverage gate >= 85%
```

The chip-fact suites run pyresidfp deterministically (fixed write scripts, ENV3
reads, ±8-count nondeterminism floor). A/B audibility tests are write-count
matched so the comparison isolates placement, not write count.

## Used by

- `preframr` (training/inference): `render_to_wav`, `sidq`, `play_samples`.
- `preframr-tokens`: `assert_dfs_render_equivalent` (transform gates),
  `play_via_aplay`, `LiveAnimator`.
- `preframr-xpt` (research): `render_df_to_wav`, `compare_renders`,
  `fingerprint_batch`, `df_to_packets`.

## Stability

Library follows semver from v1.0. Pre-1.0 releases may break API as the
preframr codebase evolves. The root re-exports above are the supported
surface; `_reg_mappers.FreqMapper` and `_sid_constants` are internal but
currently imported by sibling repos.

## License

Apache 2.0. See `LICENSE`.
