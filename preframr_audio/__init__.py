"""preframr-audio: SID audio rendering, fidelity comparison, fingerprinting.

Public surface is re-exported here so consumers do ``from preframr_audio import
compare_renders`` rather than reaching into submodules. Resolution is lazy
(PEP 562 ``__getattr__``): ``import preframr_audio`` stays cheap and does not pull
in ``pyresidfp``/``scipy`` -- those load only when a render/render-backed symbol
is first accessed, preserving the soft-dependency behaviour the render-free path
(``compare_renders``, ``mel_features``) relies on.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("preframr-audio")
except PackageNotFoundError:  # pragma: no cover - source tree without dist-info
    __version__ = "0.0.0"

_EXPORTS = {
    "render_to_samples": "audio_driver",
    "render_to_wav": "audio_driver",
    "render_per_voice": "audio_driver",
    "AudioFidelityResult": "fidelity",
    "FRAME_RMS_TOLERANCE": "fidelity",
    "compare_renders": "fidelity",
    "compare_renders_per_voice": "fidelity",
    "per_frame_rel_rms": "fidelity",
    "dfs_render_equivalent": "fidelity",
    "assert_dfs_render_equivalent": "fidelity",
    "PerceptualFidelityResult": "fidelity",
    "PERCEPTUAL_DISTANCE_THRESHOLD": "fidelity",
    "perceptual_distance": "fidelity",
    "dfs_perceptually_equivalent": "fidelity",
    "assert_dfs_perceptually_equivalent": "fidelity",
    "render_batch": "batch",
    "verify_equivalent_batch": "batch",
    "verify_perceptually_equivalent_batch": "batch",
    "mel_features": "features",
    "spectral_features": "features",
    "band_power_features": "features",
    "raw_pcm_features": "features",
    "fingerprint_writes": "fingerprint",
    "fingerprint_batch": "fingerprint",
    "canonical_scaffold": "fingerprint",
    "sidq": "sidwav",
    "default_sid": "sidwav",
    "write_reg": "sidwav",
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'preframr_audio' has no attribute {name!r}")
    return getattr(importlib.import_module(f"preframr_audio.{module}"), name)


def __dir__():
    return list(__all__)
