# preframr-audio

SID audio rendering primitives extracted from the
[preframr](https://github.com/anarkiwi/preframr) research codebase.

Provides offline + real-time playback of pre-parsed register-event
sequences through [pyresidfp](https://github.com/pyresidfp/pyresidfp).
Used by the preframr training/predict pipeline; published as a
standalone package so the audio path can be installed without the
full training stack.

## Install

```bash
pip install preframr-audio
```

System dependencies: ALSA + `libasound2` for the real-time driver
(offline rendering to WAV works without ALSA).

## Modules

- `preframr_audio.sidwav` -- `default_sid()`, `sidq()`,
  `write_reg(sid, reg, val, reg_widths)`.
- `preframr_audio.audio_driver` -- `render_to_wav()` (offline),
  `play_samples()` (real-time via ALSA), `ResidWorker`,
  `AudioRenderBuffer`, `FramePacket` / `FrameOp`.
- `preframr_audio.live_animator` -- terminal voice-state visualiser.

## Stability

Library follows semver from v1.0. Pre-1.0 releases may break API as
the preframr codebase evolves.

## License

Apache 2.0. See `LICENSE`.
