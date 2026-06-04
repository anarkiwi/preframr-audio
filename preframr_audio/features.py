"""Audio feature extractors. Pluggable feature_fn signatures for fingerprint and fidelity pipelines: ``f(samples: np.ndarray, sample_rate: int) -> np.ndarray``. Built-ins: ``mel_features`` (log-mel time-pooled), ``spectral_features`` (cheap classical descriptors), ``raw_pcm_features`` (pass-through for callers that want the unfingerprinted samples)."""

from __future__ import annotations

import numpy as np

DEFAULT_N_MELS = 64
DEFAULT_FMIN = 30.0
DEFAULT_FMAX_FRAC = 0.5
DEFAULT_WIN_MS = 25.0
DEFAULT_HOP_MS = 10.0


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    mel_lo = _hz_to_mel(np.asarray(fmin))
    mel_hi = _hz_to_mel(np.asarray(fmax))
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if mid > lo:
            fb[m, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def _stft_power(
    samples: np.ndarray, sample_rate: int, win_ms: float, hop_ms: float
) -> tuple[np.ndarray, int]:
    """Real-FFT power spectrogram via Hann-windowed framing. Returns (power, n_fft)."""
    n_win = max(1, int(round(sample_rate * win_ms / 1000.0)))
    n_fft = 1 << (n_win - 1).bit_length()
    n_hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_win) / max(1, n_win - 1))
    n_frames = max(1, 1 + (len(samples) - n_win) // n_hop)
    if len(samples) < n_win:
        padded = np.zeros(n_win, dtype=np.float64)
        padded[: len(samples)] = samples
        samples = padded
        n_frames = 1
    power = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.float64)
    for f in range(n_frames):
        start = f * n_hop
        frame = samples[start : start + n_win].astype(np.float64) * window
        if len(frame) < n_win:
            tmp = np.zeros(n_win, dtype=np.float64)
            tmp[: len(frame)] = frame
            frame = tmp
        spec = np.fft.rfft(frame, n=n_fft)
        power[f] = np.abs(spec) ** 2
    return power, n_fft


def mel_features(
    samples: np.ndarray,
    sample_rate: int,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = DEFAULT_FMIN,
    fmax: float | None = None,
    win_ms: float = DEFAULT_WIN_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> np.ndarray:
    """Log-mel spectrogram time-pooled to (2*n_mels,) via per-bin (mean, std). Deterministic; pure numpy. Fixed-length output regardless of input duration."""
    if samples.size == 0:
        return np.zeros(2 * n_mels, dtype=np.float64)
    if fmax is None:
        fmax = sample_rate * DEFAULT_FMAX_FRAC
    s = samples.astype(np.float64)
    peak = float(np.abs(s).max())
    if peak > 0:
        s = s / peak
    power, n_fft = _stft_power(s, sample_rate, win_ms, hop_ms)
    fb = _mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
    mel = power @ fb.T
    log_mel = np.log(np.maximum(mel, 1e-10))
    means = log_mel.mean(axis=0)
    stds = log_mel.std(axis=0)
    return np.concatenate([means, stds]).astype(np.float64)


def spectral_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Classical spectral descriptors: (centroid_hz, bandwidth_hz, rolloff_95_hz, zero_crossing_rate, rms). Time-averaged scalars. Fast; complements mel for cheap acoustic similarity."""
    if samples.size == 0:
        return np.zeros(5, dtype=np.float64)
    s = samples.astype(np.float64)
    peak = float(np.abs(s).max())
    if peak > 0:
        s = s / peak
    win_ms = DEFAULT_WIN_MS
    hop_ms = DEFAULT_HOP_MS
    power, _n_fft = _stft_power(s, sample_rate, win_ms, hop_ms)
    freqs = np.linspace(0.0, sample_rate / 2.0, power.shape[1])
    power_sum = power.sum(axis=1) + 1e-12
    centroid_per_frame = (power * freqs[None, :]).sum(axis=1) / power_sum
    centroid = float(centroid_per_frame.mean())
    bandwidth_per_frame = np.sqrt(
        (power * (freqs[None, :] - centroid_per_frame[:, None]) ** 2).sum(axis=1)
        / power_sum
    )
    bandwidth = float(bandwidth_per_frame.mean())
    cum_power = np.cumsum(power, axis=1) / power_sum[:, None]
    rolloff_idx = np.argmax(cum_power >= 0.95, axis=1)
    rolloff = float(freqs[rolloff_idx].mean())
    sign = np.signbit(s)
    zcr = float((sign[:-1] != sign[1:]).sum()) / max(1, len(s) - 1)
    rms = float(np.sqrt(np.mean(s**2)))
    return np.array([centroid, bandwidth, rolloff, zcr, rms], dtype=np.float64)


DEFAULT_N_BANDS = 32
DEFAULT_BAND_FMAX = 8000.0
DEFAULT_BAND_FLOOR = 1e-3


def band_power_features(
    samples: np.ndarray,
    sample_rate: int,
    n_bands: int = DEFAULT_N_BANDS,
    fmax: float = DEFAULT_BAND_FMAX,
    floor: float = DEFAULT_BAND_FLOOR,
) -> np.ndarray:
    """Coarse linear-band log power *distribution*: one sharp real-FFT over the
    whole input, summed into ``n_bands`` equal-width bands across ``[0, fmax]`` Hz,
    normalised to sum 1, then log-compressed. Returns ``(n_bands,)``.

    A deliberately pitch-tolerant spectral-envelope descriptor: a frequency shift
    smaller than a band's width (``fmax / n_bands`` Hz) leaves the per-band sums
    unchanged, so a sub-band detune reads as ~0, while timbre (waveform), filter
    (cutoff), and voicing (drop/envelope) changes move energy *between* bands and
    so read large. This is the complement of ``mel_features``: mel allocates
    narrow bins to low frequencies and so reacts strongly to sub-semitone pitch
    shifts of bass-heavy spectra (e.g. SID), which is exactly wrong for a
    perceptual render-equivalence gate. Deterministic; pure numpy."""
    if samples.size == 0:
        return np.zeros(n_bands, dtype=np.float64)
    s = samples.astype(np.float64)
    peak = float(np.abs(s).max())
    if peak > 0:
        s = s / peak
    power = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), 1.0 / sample_rate)
    edges = np.linspace(0.0, fmax, n_bands + 1)
    idx = np.clip(np.digitize(freqs, edges) - 1, 0, n_bands - 1)
    bands = np.zeros(n_bands, dtype=np.float64)
    np.add.at(bands, idx, power)
    total = bands.sum()
    if total > 0:
        bands = bands / total
    return np.log(bands + floor)


def raw_pcm_features(samples: np.ndarray, _sample_rate: int) -> np.ndarray:
    """Pass-through: returns samples as float64 (normalised to [-1, 1] by peak). Caller-defined comparison via the resulting full-length vector."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    s = samples.astype(np.float64)
    peak = float(np.abs(s).max())
    if peak > 0:
        s = s / peak
    return s


FEATURE_FNS = {
    "mel": mel_features,
    "spectral": spectral_features,
    "band_power": band_power_features,
    "raw_pcm": raw_pcm_features,
}
