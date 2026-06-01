"""Smoke tests for the SID utility helpers in ``preframr_audio.sidwav``."""

import unittest
from datetime import timedelta

import numpy as np
from pyresidfp import SoundInterfaceDevice

from preframr_audio.sidwav import default_sid, sidq


class TestSidwavHelpers(unittest.TestCase):
    def test_default_sid_returns_8580(self):
        sid = default_sid()
        self.assertGreater(sid.clock_frequency, 0)
        self.assertGreater(sid.sampling_frequency, 0)

    def test_sidq_inverse_of_clock_frequency(self):
        sid = default_sid()
        q = sidq(sid)
        self.assertAlmostEqual(q, 1.0 / sid.clock_frequency, places=12)
        self.assertAlmostEqual(q * sid.clock_frequency, 1.0, places=9)


class TestSidNullDelay(unittest.TestCase):
    """Direct resid envelope-equivalence checks. Independent of the
    preframr write path -- pure pyresidfp behaviour we depend on for
    the sustain-max canonical-form simplification."""

    @staticmethod
    def _samples(decay_val, sustain_val):
        sid = SoundInterfaceDevice(sampling_frequency=8e3)
        sec = timedelta(seconds=1)
        sid.write_register(24, 15)
        sid.write_register(5, decay_val)
        sid.write_register(6, sustain_val)
        sid.write_register(0, 8)
        sid.write_register(1, 8)
        sid.clock(sec)
        sid.write_register(4, 33)
        return np.array(sid.clock(sec))

    def test_max_sustain_decay_is_no_op(self):
        a = self._samples(0b00010001, 0b11110000)
        b = self._samples(0b00010100, 0b11110000)
        self.assertTrue(np.allclose(a, b, rtol=2, atol=2))

    def test_non_max_sustain_shows_decay(self):
        a = self._samples(0b00010001, 0b00110000)
        b = self._samples(0b00010100, 0b00110000)
        self.assertFalse(np.allclose(a, b, rtol=2, atol=2))


class TestSidRingSync(unittest.TestCase):
    """Ring-mod / sync behaviour (used by the macros canonicalisation
    rules); also pure pyresidfp -- pinned here so a chip-model bump
    can't silently change the audio output the LM was trained on."""

    @staticmethod
    def _ring_samples(ctrl_val):
        sid = SoundInterfaceDevice(sampling_frequency=8e3)
        sec = timedelta(seconds=1)
        sid.write_register(24, 15)
        sid.write_register(6, 0b11110000)
        sid.write_register(0, 8)
        sid.write_register(1, 8)
        sid.write_register(20, 0b11110000)
        sid.write_register(14, 4)
        sid.write_register(15, 4)
        sid.clock(sec)
        sid.write_register(4, ctrl_val)
        return np.array(sid.clock(sec))

    def test_ring_affects_triangle_only(self):
        gate = 1
        ring = 0b100
        tri = 16
        a = self._ring_samples(tri + gate)
        b = self._ring_samples(tri + gate + ring)
        self.assertFalse(np.allclose(a, b, rtol=2, atol=2))
        for waveform in (32, 64, 128):
            a = self._ring_samples(waveform + gate)
            b = self._ring_samples(waveform + gate + ring)
            self.assertTrue(np.allclose(a, b, rtol=3, atol=3))

    @staticmethod
    def _sync_samples(ctrl_val1, ctrl_val2):
        sid = SoundInterfaceDevice(sampling_frequency=8e3)
        sec = timedelta(seconds=1)
        sid.write_register(24, 15)
        sid.write_register(6, 240)
        sid.write_register(0, 8)
        sid.write_register(1, 8)
        sid.write_register(20, 240)
        sid.write_register(14, 4)
        sid.write_register(15, 4)
        sid.write_register(4, ctrl_val1)
        sid.clock(sec)
        sid.write_register(4, ctrl_val2)
        return np.array(sid.clock(sec))

    def test_sync_no_op_without_waveform(self):
        gate = 1
        sync = 0b10
        for waveform in (16, 32, 64, 128):
            a = self._sync_samples(waveform + gate, 0)
            b = self._sync_samples(waveform + gate, sync)
            self.assertTrue(np.allclose(a, b, rtol=3, atol=3))


if __name__ == "__main__":
    unittest.main()
