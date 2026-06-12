"""Unit tests for ``preframr_audio.audio_driver`` testable surface."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from preframr_audio._sid_constants import DELAY_REG, FRAME_REG, MODE_VOL_REG
from preframr_audio.audio_driver import (
    ASID_MANID,
    ASID_REG_TO_SLOT,
    ASID_SLOT_TO_REG,
    ASID_START,
    ASID_UPDATE,
    FrameOp,
    FramePacket,
    SampleRing,
    _default_reg_start,
    _OpCollector,
    _reg_start_init_ops,
    decode_asid_update,
    df_to_packets,
    encode_asid_update,
)

_NO_FREQ_MAPPER = None


class TestEncodeDecodeAsidUpdate(unittest.TestCase):
    def test_empty_input_returns_only_header(self):
        body = encode_asid_update({})
        self.assertEqual(body[:2], [ASID_MANID, ASID_UPDATE])
        self.assertEqual(body[2:6], [0, 0, 0, 0])
        self.assertEqual(body[6:10], [0, 0, 0, 0])
        self.assertEqual(body[10:], [])

    def test_unknown_reg_silently_dropped(self):
        body = encode_asid_update({99: 0x55})
        self.assertEqual(body[2:10], [0, 0, 0, 0, 0, 0, 0, 0])

    def test_round_trip_low_byte(self):
        body = encode_asid_update({0: 0x42})
        decoded = decode_asid_update(body)
        self.assertEqual(decoded, {0: 0x42})

    def test_round_trip_high_byte_sets_msb(self):
        body = encode_asid_update({4: 0x84})
        decoded = decode_asid_update(body)
        self.assertEqual(decoded, {4: 0x84})

    def test_round_trip_multi_reg(self):
        regs = {0: 0x12, 5: 0x84, 24: 0xFF}
        body = encode_asid_update(regs)
        decoded = decode_asid_update(body)
        self.assertEqual(decoded, regs)

    def test_decode_short_packet_returns_empty(self):
        self.assertEqual(decode_asid_update([ASID_MANID, ASID_UPDATE]), {})

    def test_decode_wrong_manid(self):
        body = [0x00] * 12
        self.assertEqual(decode_asid_update(body), {})

    def test_decode_wrong_cmd(self):
        body = [ASID_MANID, ASID_START] + [0] * 8
        self.assertEqual(decode_asid_update(body), {})

    def test_decode_truncated_values_stops_cleanly(self):
        body = [ASID_MANID, ASID_UPDATE, 0b11, 0, 0, 0, 0, 0, 0, 0, 0x33]
        decoded = decode_asid_update(body)
        self.assertEqual(decoded, {ASID_SLOT_TO_REG[0]: 0x33})


class TestSampleRing(unittest.TestCase):
    def test_write_then_read_round_trip(self):
        ring = SampleRing(capacity_samples=16)
        samples = np.arange(8, dtype=np.int16)
        self.assertTrue(ring.write(samples))
        self.assertEqual(ring.occupancy(), (8, 16))
        out = np.empty(8, dtype=np.int16)
        n = ring.read(8, out)
        self.assertEqual(n, 8)
        np.testing.assert_array_equal(out, samples)
        self.assertEqual(ring.occupancy(), (0, 16))

    def test_write_oversized_returns_false(self):
        ring = SampleRing(capacity_samples=4)
        too_big = np.zeros(8, dtype=np.int16)
        self.assertFalse(ring.write(too_big))

    def test_read_wraparound(self):
        ring = SampleRing(capacity_samples=8)
        first = np.arange(6, dtype=np.int16)
        ring.write(first)
        out = np.empty(4, dtype=np.int16)
        ring.read(4, out)
        np.testing.assert_array_equal(out, first[:4])

        more = np.array([100, 101, 102, 103], dtype=np.int16)
        ring.write(more)
        out6 = np.empty(6, dtype=np.int16)
        n = ring.read(6, out6)
        self.assertEqual(n, 6)
        np.testing.assert_array_equal(
            out6, np.array([4, 5, 100, 101, 102, 103], dtype=np.int16)
        )

    def test_short_read_increments_underruns(self):
        ring = SampleRing(capacity_samples=4)
        ring.write(np.array([1, 2], dtype=np.int16))
        out = np.empty(4, dtype=np.int16)
        n = ring.read(4, out)
        self.assertEqual(n, 2)
        np.testing.assert_array_equal(out, np.array([1, 2, 0, 0], dtype=np.int16))
        self.assertEqual(ring.underruns, 1)


class TestOpCollector(unittest.TestCase):
    def test_collects_write_register_calls(self):
        c = _OpCollector(freq_mapper=None)
        c.write_register(5, 0xAB)
        c.write_register(7, 0x12)
        self.assertEqual(c.collected, [(5, 0xAB), (7, 0x12)])


class TestDfToPackets(unittest.TestCase):
    def _build_df(self, rows):
        return pd.DataFrame(rows)

    def test_empty_df_yields_nothing(self):
        df = self._build_df([])
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual(out, [])

    def test_marker_before_writes_emits_silence_packet(self):
        df = self._build_df([{"reg": FRAME_REG, "val": 0, "delay": 0.01}])
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual(len(out), 1)
        pkt = out[0]
        self.assertEqual(pkt.frame_id, 0)
        self.assertEqual(len(pkt.ops), 1)
        self.assertEqual(pkt.ops[0].reg, -1)
        self.assertGreater(pkt.ops[0].delay_cycles, 0)

    def test_normal_frame(self):
        df = self._build_df(
            [
                {"reg": MODE_VOL_REG, "val": 0x0F, "delay": 0.001},
                {"reg": FRAME_REG, "val": 0, "delay": 0.0},
            ]
        )
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual(len(out), 1)
        pkt = out[0]
        self.assertEqual(pkt.frame_id, 0)
        regs_written = [op.reg for op in pkt.ops if op.reg >= 0]
        self.assertIn(MODE_VOL_REG, regs_written)

    def test_trailing_partial_frame_emitted(self):
        df = self._build_df([{"reg": MODE_VOL_REG, "val": 0x0A, "delay": 0.001}])
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].frame_id, 0)
        self.assertTrue(out[0].ops)

    def test_bad_row_skipped(self):
        df = self._build_df(
            [
                {"reg": 999, "val": 0, "delay": 0.0},
                {"reg": MODE_VOL_REG, "val": 5, "delay": 0.0},
                {"reg": FRAME_REG, "val": 0, "delay": 0.0},
            ]
        )
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual(len(out), 1)

    def test_frame_id_increments(self):
        df = self._build_df(
            [
                {"reg": MODE_VOL_REG, "val": 1, "delay": 0.0},
                {"reg": FRAME_REG, "val": 0, "delay": 0.0},
                {"reg": MODE_VOL_REG, "val": 2, "delay": 0.0},
                {"reg": FRAME_REG, "val": 0, "delay": 0.0},
            ]
        )
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual([p.frame_id for p in out], [0, 1])

    def test_frame_id_start_offsets_frame_ids(self):
        df = self._build_df(
            [
                {"reg": MODE_VOL_REG, "val": 1, "delay": 0.0},
                {"reg": FRAME_REG, "val": 0, "delay": 0.0},
                {"reg": MODE_VOL_REG, "val": 2, "delay": 0.0},
                {"reg": FRAME_REG, "val": 0, "delay": 0.0},
            ]
        )
        out = list(
            df_to_packets(
                df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER, frame_id_start=7
            )
        )
        self.assertEqual([p.frame_id for p in out], [7, 8])

    def test_delay_reg_marker_advances_frame_id(self):
        df = self._build_df(
            [
                {"reg": MODE_VOL_REG, "val": 1, "delay": 0.0},
                {"reg": DELAY_REG, "val": 0, "delay": 0.01},
            ]
        )
        out = list(df_to_packets(df, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER))
        self.assertEqual(len(out), 1)


class TestDefaultRegStart(unittest.TestCase):
    def test_layout(self):
        rs = _default_reg_start()
        self.assertEqual(rs[MODE_VOL_REG], 15)
        from preframr_audio._sid_constants import VOICE_REG_SIZE, VOICES

        for v in range(VOICES):
            offset = v * VOICE_REG_SIZE
            self.assertEqual(rs[6 + offset], 240)
            self.assertEqual(rs[3 + offset], 16)


class TestRegStartInitOps(unittest.TestCase):
    def test_default_start_emits_at_least_one_op_per_reg(self):
        rs = _default_reg_start()
        ops = _reg_start_init_ops(rs, reg_widths={}, freq_mapper=_NO_FREQ_MAPPER)
        self.assertTrue(ops)
        for op in ops:
            self.assertEqual(op.delay_cycles, 0)


class TestFrameTypesAreUsable(unittest.TestCase):
    def test_frame_op_default_delay(self):
        op = FrameOp(reg=1, val=2)
        self.assertEqual(op.delay_cycles, 0)

    def test_frame_packet_defaults(self):
        pkt = FramePacket(frame_id=42)
        self.assertEqual(pkt.frame_id, 42)
        self.assertEqual(pkt.ops, [])
        self.assertEqual(pkt.total_cycles, 19656)
        self.assertFalse(pkt.edited)


class TestSlotRegRoundTrip(unittest.TestCase):
    """Cheap consistency check on the ASID slot map -- catches a future
    edit that breaks the slot<->reg invariant."""

    def test_slot_reg_bijection(self):
        self.assertEqual(len(ASID_SLOT_TO_REG), len(ASID_REG_TO_SLOT))
        for slot, reg in ASID_SLOT_TO_REG.items():
            self.assertEqual(ASID_REG_TO_SLOT[reg], slot)


class TestPlayViaWav(unittest.TestCase):
    """``play_via_wav`` renders the df to a temp wav file and execs
    aplay/paplay against it. All heavy collaborators -- render_to_wav,
    subprocess, threading, tempfile -- are mocked; we assert the
    branches and cleanup invariants.
    """

    def _proc_stub(self, returncode=0, stderr=b""):
        proc = mock.MagicMock()
        proc.stderr = mock.MagicMock()
        proc.stderr.read.return_value = stderr
        proc.wait.return_value = None
        proc.returncode = returncode
        return proc

    def _common_patches(self, stack, *, returncode=0, n_samples=4800):
        import preframr_audio.audio_driver as ad

        stack.enter_context(mock.patch.object(ad, "_HAVE_RESID", True))
        stack.enter_context(
            mock.patch.object(ad, "render_to_wav", return_value=n_samples)
        )
        stack.enter_context(
            mock.patch(
                "tempfile.mkstemp",
                return_value=(3, "/tmp/preframr_play_xyz.wav"),
            )
        )
        stack.enter_context(mock.patch("os.close"))
        unlink = stack.enter_context(mock.patch("os.unlink"))
        popen = stack.enter_context(
            mock.patch("subprocess.Popen", return_value=self._proc_stub(returncode))
        )
        return popen, unlink

    def test_pyresid_missing_raises(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ad, "_HAVE_RESID", False))
            with self.assertRaises(RuntimeError) as ctx:
                ad.play_via_wav(df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656)
        self.assertIn("pyresidfp", str(ctx.exception))

    def test_default_aplay_command_with_cleanup(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            popen, unlink = self._common_patches(stack)
            n = ad.play_via_wav(df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656)
        self.assertEqual(n, 4800)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:2], ["aplay", "-q"])
        self.assertEqual(cmd[-1], "/tmp/preframr_play_xyz.wav")
        unlink.assert_called_once_with("/tmp/preframr_play_xyz.wav")

    def test_paplay_path(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            popen, _ = self._common_patches(stack)
            ad.play_via_wav(
                df=pd.DataFrame({"reg": [0]}),
                reg_widths={},
                irq=19656,
                use_paplay=True,
            )
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd, ["paplay", "/tmp/preframr_play_xyz.wav"])

    def test_audio_device_threaded_into_aplay_cmd(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            popen, _ = self._common_patches(stack)
            ad.play_via_wav(
                df=pd.DataFrame({"reg": [0]}),
                reg_widths={},
                irq=19656,
                audio_device="plughw:1,0",
            )
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:4], ["aplay", "-D", "plughw:1,0", "-q"])

    def test_keep_wav_skips_cleanup(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            _, unlink = self._common_patches(stack)
            ad.play_via_wav(
                df=pd.DataFrame({"reg": [0]}),
                reg_widths={},
                irq=19656,
                keep_wav="/tmp/keep.wav",
            )
        unlink.assert_not_called()

    def test_nonzero_returncode_raises_runtime_error(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._common_patches(stack, returncode=2)
            popen_patch = mock.patch(
                "subprocess.Popen",
                return_value=self._proc_stub(returncode=2, stderr=b"some failure"),
            )
            stack.enter_context(popen_patch)
            with self.assertRaises(RuntimeError) as ctx:
                ad.play_via_wav(df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656)
        self.assertIn("rc=2", str(ctx.exception))
        self.assertIn("some failure", str(ctx.exception))


class TestPlayViaAplayPrerendered(unittest.TestCase):
    """``play_via_aplay_prerendered`` renders the entire df into a
    sample buffer first, then pipes it to aplay/paplay over stdin.
    Mocks the WavSampleSink + ResidWorker + AudioRenderBuffer +
    _drive chain so the test stays in unit-test budget.
    """

    def _proc_stub(self, returncode=0, stderr=b"", timeout=False):
        proc = mock.MagicMock()
        proc.stdin = mock.MagicMock()
        proc.stdin.write.return_value = None
        proc.stdin.close.return_value = None
        proc.stderr = mock.MagicMock()
        proc.stderr.read.return_value = stderr
        if timeout:
            import subprocess

            proc.wait.side_effect = [
                subprocess.TimeoutExpired(["aplay"], 30),
                None,
            ]
        else:
            proc.wait.return_value = None
        proc.returncode = returncode
        return proc

    def _patch_chain(self, stack, sample_rate=48000, n_samples=4):
        import preframr_audio.audio_driver as ad

        stack.enter_context(mock.patch.object(ad, "_HAVE_RESID", True))

        sink = mock.MagicMock()
        sink.to_array.return_value = np.zeros(n_samples, dtype=np.int16)
        stack.enter_context(mock.patch.object(ad, "WavSampleSink", return_value=sink))

        buffer_mock = mock.MagicMock()
        stack.enter_context(
            mock.patch.object(ad, "AudioRenderBuffer", return_value=buffer_mock)
        )

        worker = mock.MagicMock()
        worker.clock_frequency = 985248
        worker.sampling_frequency = sample_rate
        stack.enter_context(mock.patch.object(ad, "ResidWorker", return_value=worker))

        stack.enter_context(mock.patch.object(ad, "_drive"))

    def test_pyresid_missing_raises(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ad, "_HAVE_RESID", False))
            stack.enter_context(mock.patch.object(ad, "WavSampleSink"))
            with self.assertRaises(RuntimeError) as ctx:
                ad.play_via_aplay_prerendered(
                    df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656
                )
        self.assertIn("pyresidfp", str(ctx.exception))

    def test_default_aplay_cmd_shape(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._patch_chain(stack, sample_rate=48000)
            popen = stack.enter_context(
                mock.patch("subprocess.Popen", return_value=self._proc_stub())
            )
            n = ad.play_via_aplay_prerendered(
                df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656
            )
        self.assertEqual(n, 4)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[0], "aplay")
        self.assertIn("-f", cmd)
        self.assertIn("S16_LE", cmd)
        self.assertIn("-r", cmd)
        self.assertIn("48000", cmd)
        self.assertIn("-c", cmd)
        self.assertEqual(cmd[-1], "-")

    def test_audio_device_threaded_into_default_cmd(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._patch_chain(stack)
            popen = stack.enter_context(
                mock.patch("subprocess.Popen", return_value=self._proc_stub())
            )
            ad.play_via_aplay_prerendered(
                df=pd.DataFrame({"reg": [0]}),
                reg_widths={},
                irq=19656,
                audio_device="plughw:2,0",
            )
        cmd = popen.call_args.args[0]
        self.assertIn("-D", cmd)
        self.assertIn("plughw:2,0", cmd)
        self.assertEqual(cmd[-1], "-")

    def test_custom_aplay_cmd_used_verbatim(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._patch_chain(stack)
            popen = stack.enter_context(
                mock.patch("subprocess.Popen", return_value=self._proc_stub())
            )
            custom = ["paplay", "--raw", "--rate=48000"]
            ad.play_via_aplay_prerendered(
                df=pd.DataFrame({"reg": [0]}),
                reg_widths={},
                irq=19656,
                aplay_cmd=custom,
            )
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd, custom)

    def test_nonzero_returncode_raises_runtime_error(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._patch_chain(stack)
            stack.enter_context(
                mock.patch(
                    "subprocess.Popen",
                    return_value=self._proc_stub(returncode=3, stderr=b"alsa bad"),
                )
            )
            with self.assertRaises(RuntimeError) as ctx:
                ad.play_via_aplay_prerendered(
                    df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656
                )
        self.assertIn("rc=3", str(ctx.exception))
        self.assertIn("alsa bad", str(ctx.exception))

    def test_timeout_kills_subprocess(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._patch_chain(stack)
            proc = self._proc_stub(timeout=True, returncode=0)
            stack.enter_context(mock.patch("subprocess.Popen", return_value=proc))
            ad.play_via_aplay_prerendered(
                df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656
            )
        proc.kill.assert_called_once()


class TestPlayViaAplay(unittest.TestCase):
    """``play_via_aplay`` is the live-stream variant: clocks the SID
    inline and pipes samples to aplay over stdin. We mock the SID
    chip + subprocess so the test stays hermetic.
    """

    def _sid_stub(self, sample_rate=48000, clock_frequency=985248):
        sid = mock.MagicMock()
        sid.sampling_frequency = sample_rate
        sid.clock_frequency = clock_frequency
        sid.write_register.return_value = None
        sid.reset.return_value = None
        sid.clock.return_value = np.zeros(64, dtype=np.int16)
        return sid

    def _proc_stub(self, returncode=0, stderr=b""):
        proc = mock.MagicMock()
        proc.stdin = mock.MagicMock()
        proc.stdin.write.return_value = None
        proc.stdin.close.return_value = None
        proc.stderr = mock.MagicMock()
        proc.stderr.read.return_value = stderr
        proc.wait.return_value = None
        proc.returncode = returncode
        return proc

    def _patch_chain(self, stack, sid=None, popen_proc=None):
        import preframr_audio.audio_driver as ad

        stack.enter_context(mock.patch.object(ad, "_HAVE_RESID", True))
        sid = sid or self._sid_stub()
        stack.enter_context(
            mock.patch.object(ad, "SoundInterfaceDevice", return_value=sid)
        )
        stack.enter_context(
            mock.patch.object(
                ad, "ChipModel", mock.MagicMock(MOS8580=mock.sentinel.MOS8580)
            )
        )
        popen_proc = popen_proc or self._proc_stub()
        popen = stack.enter_context(
            mock.patch("subprocess.Popen", return_value=popen_proc)
        )
        return sid, popen, popen_proc

    def test_pyresid_missing_raises(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ad, "_HAVE_RESID", False))
            with self.assertRaises(RuntimeError) as ctx:
                ad.play_via_aplay(
                    df=pd.DataFrame({"reg": [0]}), reg_widths={}, irq=19656
                )
        self.assertIn("pyresidfp", str(ctx.exception))

    def test_default_aplay_cmd_threads_sample_rate(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        empty_df = pd.DataFrame({"reg": pd.Series([], dtype=int)})
        with contextlib.ExitStack() as stack:
            sid, popen, _ = self._patch_chain(
                stack, sid=self._sid_stub(sample_rate=44100)
            )
            ad.play_via_aplay(df=empty_df, reg_widths={}, irq=19656)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[0], "aplay")
        self.assertIn("44100", cmd)
        self.assertEqual(cmd[-1], "-")
        sid.reset.assert_called()

    def test_audio_device_threaded_into_default_cmd(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            _, popen, _ = self._patch_chain(stack)
            ad.play_via_aplay(
                df=pd.DataFrame({"reg": pd.Series([], dtype=int)}),
                reg_widths={},
                irq=19656,
                audio_device="plughw:0,3",
            )
        cmd = popen.call_args.args[0]
        self.assertIn("-D", cmd)
        self.assertIn("plughw:0,3", cmd)

    def test_custom_aplay_cmd_used_verbatim(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            _, popen, _ = self._patch_chain(stack)
            custom = ["paplay", "--raw"]
            ad.play_via_aplay(
                df=pd.DataFrame({"reg": pd.Series([], dtype=int)}),
                reg_widths={},
                irq=19656,
                aplay_cmd=custom,
            )
        self.assertEqual(popen.call_args.args[0], custom)

    def test_nonzero_returncode_raises_with_hint(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        with contextlib.ExitStack() as stack:
            self._patch_chain(
                stack,
                popen_proc=self._proc_stub(returncode=1, stderr=b"audio open error"),
            )
            with self.assertRaises(RuntimeError) as ctx:
                ad.play_via_aplay(
                    df=pd.DataFrame({"reg": pd.Series([], dtype=int)}),
                    reg_widths={},
                    irq=19656,
                )
        msg = str(ctx.exception)
        self.assertIn("rc=1", msg)
        self.assertIn("audio open error", msg)
        self.assertIn("aplay -l", msg)
        self.assertIn("--audio-device", msg)

    def test_should_quit_breaks_loop(self):
        import contextlib

        import preframr_audio.audio_driver as ad

        on_frame = mock.MagicMock()
        df = pd.DataFrame(
            [
                {"reg": MODE_VOL_REG, "val": 15, "delay": 0.0},
                {"reg": FRAME_REG, "val": 0, "delay": 0.01},
            ]
        )
        with contextlib.ExitStack() as stack:
            self._patch_chain(stack)
            ad.play_via_aplay(
                df=df,
                reg_widths={},
                irq=19656,
                on_frame=on_frame,
                should_quit=lambda: True,
            )
        on_frame.assert_not_called()


if __name__ == "__main__":
    unittest.main()
