"""Tests for ``preframr_audio.live_animator``."""

import io
import unittest

from preframr_audio._sid_constants import VOICE_REG_SIZE, VOICES
from preframr_audio.audio_driver import FrameOp, FramePacket
from preframr_audio.live_animator import (
    VOICE_CTRL_REG,
    LiveAnimator,
    VoiceState,
    _BpsMeter,
    _MacroBurst,
    voice_of_reg,
)


class TestVoiceState(unittest.TestCase):
    def test_absorb_ctrl_decodes_gate_and_waveform(self):
        vs = VoiceState()
        relevant = vs.absorb_op(op_reg=4, op_val=0x21, voice_index=0)
        self.assertTrue(relevant)
        self.assertTrue(vs.gate)
        self.assertEqual(vs.waveform_nibble, 0x2)
        self.assertEqual(vs.waveform_name(), "saw")

    def test_absorb_freq_combines_lo_hi(self):
        vs = VoiceState()
        vs.absorb_op(op_reg=0, op_val=0x12, voice_index=0)
        vs.absorb_op(op_reg=1, op_val=0x34, voice_index=0)
        self.assertEqual(vs.freq(), 0x3412)

    def test_absorb_ignores_other_voices_regs(self):
        vs = VoiceState()
        relevant = vs.absorb_op(op_reg=11, op_val=0x21, voice_index=0)
        self.assertFalse(relevant)
        self.assertFalse(vs.gate)

    def test_voice_1_block(self):
        vs = VoiceState()
        self.assertTrue(vs.absorb_op(op_reg=11, op_val=0x41, voice_index=1))
        self.assertTrue(vs.gate)
        self.assertEqual(vs.waveform_nibble, 0x4)
        self.assertEqual(vs.waveform_name(), "pul")


class TestLiveAnimator(unittest.TestCase):
    def test_open_emits_clear_and_hide_cursor(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        out = buf.getvalue()
        self.assertIn("\x1b[2J", out)
        self.assertIn("\x1b[?25l", out)
        self.assertIn("preframr live render", out)

    def test_close_shows_cursor(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        before = len(buf.getvalue())
        anim.close()
        after = buf.getvalue()
        self.assertIn("\x1b[?25h", after[before:])

    def test_on_frame_updates_voice_state(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        ops = [
            FrameOp(reg=4, val=0x21),
            FrameOp(reg=0, val=0x12),
            FrameOp(reg=1, val=0x34),
        ]
        anim.on_frame(FramePacket(frame_id=0, ops=ops))
        self.assertEqual(anim.frame_count, 1)
        self.assertTrue(anim.state[0].gate)
        self.assertEqual(anim.state[0].waveform_name(), "saw")
        self.assertEqual(anim.state[0].freq(), 0x3412)

    def test_on_frame_writes_to_each_voice_row(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        ops = [
            FrameOp(reg=4, val=0x21),
            FrameOp(reg=11, val=0x11),
        ]
        anim.on_frame(FramePacket(frame_id=0, ops=ops))
        out = buf.getvalue()
        self.assertIn("V1 [G] saw", out)
        self.assertIn("V2 [G] tri", out)
        self.assertIn("V3 [.]", out)

    def test_negative_reg_marker_is_skipped(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        ops = [
            FrameOp(reg=-128, val=0),
            FrameOp(reg=4, val=0x21),
        ]
        anim.on_frame(FramePacket(frame_id=0, ops=ops))
        self.assertTrue(anim.state[0].gate)

    def test_sparkline_handles_constant_history(self):
        spark = LiveAnimator._sparkline([100] * 10)
        self.assertEqual(len(spark), 10)
        self.assertEqual(set(spark), {spark[0]})

    def test_voice_count_configurable(self):
        anim = LiveAnimator(out=io.StringIO(), voices=2)
        self.assertEqual(len(anim.state), 2)

    def test_title_appears_in_header(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf, title="Robocop_3.1")
        anim.open()
        self.assertIn("Robocop_3.1", buf.getvalue())

    def test_header_includes_frame_counter_and_bps(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        anim.on_frame(FramePacket(frame_id=0, ops=[]))
        self.assertIn("frame      1", buf.getvalue())
        self.assertIn("bps:", buf.getvalue())

    def test_status_row_reflects_mute_state(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        self.assertIn("mute: 123", buf.getvalue())
        anim.muted_voices.add(1)
        anim.on_frame(FramePacket(frame_id=0, ops=[]))
        self.assertIn("mute: 1-3", buf.getvalue())

    def test_help_row_lists_hotkeys(self):
        buf = io.StringIO()
        anim = LiveAnimator(out=buf)
        anim.open()
        out = buf.getvalue()
        self.assertIn("[1/2/3]", out)
        self.assertIn("[q]", out)


class TestVoiceOfReg(unittest.TestCase):
    def test_voice_regs_map_to_voice_index(self):
        for v in range(VOICES):
            for off in range(VOICE_REG_SIZE):
                self.assertEqual(voice_of_reg(v * VOICE_REG_SIZE + off), v)

    def test_filter_and_neg_regs_are_none(self):
        self.assertIsNone(voice_of_reg(21))
        self.assertIsNone(voice_of_reg(24))
        self.assertIsNone(voice_of_reg(-128))

    def test_voice_ctrl_reg_table(self):
        self.assertEqual(VOICE_CTRL_REG, {0: 4, 1: 11, 2: 18})


class TestMacroBurst(unittest.TestCase):
    def test_gate_on_fires_label(self):
        vs = VoiceState()
        burst = _MacroBurst()
        vs.absorb_op(op_reg=4, op_val=0x21, voice_index=0)
        burst.update(vs)
        self.assertGreater(burst.flash_remaining, 0)
        self.assertEqual(burst.label.strip(), "GATE+")

    def test_waveform_change_fires(self):
        vs = VoiceState()
        vs.absorb_op(op_reg=4, op_val=0x21, voice_index=0)
        vs.snapshot_prev()
        vs.absorb_op(op_reg=4, op_val=0x41, voice_index=0)
        burst = _MacroBurst()
        burst.update(vs)
        self.assertEqual(burst.label.strip(), "PUL")

    def test_render_includes_reverse_video_when_flashing(self):
        burst = _MacroBurst()
        burst.fire("GATE+")
        rendered = burst.render()
        self.assertIn("\x1b[7m", rendered)
        self.assertIn("GATE+", rendered)

    def test_decays_after_flash_frames(self):
        burst = _MacroBurst()
        burst.fire("X")
        steady = VoiceState()
        steady.snapshot_prev()
        for _ in range(_MacroBurst.FLASH_FRAMES):
            burst.update(steady)
        self.assertEqual(burst.flash_remaining, 0)


class TestQuitFlag(unittest.TestCase):
    def test_is_quit_reflects_quit_requested(self):
        anim = LiveAnimator(out=io.StringIO())
        self.assertFalse(anim.is_quit())
        anim.quit_requested = True
        self.assertTrue(anim.is_quit())


class TestBpsMeter(unittest.TestCase):
    def test_records_bytes_and_caps(self):
        m = _BpsMeter(baud=115200)
        m.add(50)
        m.add(75)
        self.assertEqual(m.now(), 125)
        self.assertEqual(m.cap, 11520)


if __name__ == "__main__":
    unittest.main()
