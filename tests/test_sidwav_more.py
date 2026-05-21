"""Coverage tests for ``preframr_audio.sidwav.write_reg`` branch behaviour."""

import unittest

from preframr_audio._reg_mappers import FreqMapper
from preframr_audio.sidwav import write_reg


class _FakeSid:
    def __init__(self):
        self.freq_mapper = FreqMapper(cents=10)
        self.writes = []

    def write_register(self, reg, val):
        self.writes.append((reg, val))


class TestWriteReg(unittest.TestCase):
    def test_freq_reg_uses_if_map(self):
        sid = _FakeSid()
        write_reg(sid, 0, 60, {})
        self.assertEqual(len(sid.writes), 2)

    def test_freq_reg_negative_clamps_to_zero(self):
        sid = _FakeSid()
        write_reg(sid, 7, -5, {})
        self.assertEqual(len(sid.writes), 2)

    def test_freq_reg_oversize_clamps_to_max(self):
        sid = _FakeSid()
        max_in = max(sid.freq_mapper.if_map.keys())
        write_reg(sid, 14, max_in + 1000, {})
        self.assertEqual(len(sid.writes), 2)

    def test_pwm_reg_two_bytes(self):
        sid = _FakeSid()
        write_reg(sid, 2, 256, {})
        self.assertEqual(len(sid.writes), 2)

    def test_fc_lo_reg_two_bytes(self):
        sid = _FakeSid()
        write_reg(sid, 21, 100, {})
        self.assertEqual(len(sid.writes), 2)

    def test_unknown_reg_uses_reg_widths(self):
        sid = _FakeSid()
        write_reg(sid, 23, 7, {23: 1})
        self.assertEqual(len(sid.writes), 1)

    def test_default_width_one(self):
        sid = _FakeSid()
        write_reg(sid, 24, 5, {})
        self.assertEqual(len(sid.writes), 1)


if __name__ == "__main__":
    unittest.main()
