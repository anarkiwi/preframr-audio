"""Tests for ``preframr_audio.audio_driver``."""

import os
import platform
import tempfile
import threading
import time
import unittest

import numpy as np
import pandas as pd
import scipy.io.wavfile

from preframr_audio._reg_mappers import FreqMapper
from preframr_audio.audio_driver import (
    _HAVE_RESID,
    _HAVE_RTMIDI,
    ASID_MANID,
    ASID_REG_TO_SLOT,
    ASID_SLOT_TO_REG,
    ASID_START,
    ASID_STOP,
    ASID_UPDATE,
    AsidMidiDriver,
    AsidServer,
    AsidWorker,
    AudioRenderBuffer,
    FrameOp,
    FramePacket,
    ResidWorker,
    SampleRing,
    WavSampleSink,
    _default_reg_start,
    _OpCollector,
    _reg_start_init_ops,
    decode_asid_update,
    df_to_packets,
    encode_asid_update,
    render_per_voice,
    render_to_wav,
)
from preframr_audio.sidwav import sidq


@unittest.skipUnless(_HAVE_RESID, "pyresidfp required")
class TestNegativeDelayCarry(unittest.TestCase):
    """A skeleton-owned stream emits negative per-frame delay corrections (reset_diffs subtracts the
    intra-frame total); the worker must clock the running NET, not drop negatives. Dropping them clocks
    the positives only and over-runs the render (the ~3.5x stretch)."""

    def _clocked_samples(self, delays):
        buf = AudioRenderBuffer(max_frames=64)
        sink = WavSampleSink()
        worker = ResidWorker(buf, sink=sink, chip_model="MOS8580")
        worker.start()
        for i, d in enumerate(delays):
            buf.push_frame(
                FramePacket(
                    frame_id=i,
                    ops=[FrameOp(reg=-1, val=0, delay_cycles=d)],
                    total_cycles=0,
                )
            )
        buf.close()
        worker.join(timeout=30.0)
        return len(sink.to_array()), worker.clock_frequency, worker.sampling_frequency

    def test_negative_delay_offsets_prior_positive(self):
        n_pos, clk, sf = self._clocked_samples([20000, 20000])
        n_net, _, _ = self._clocked_samples([20000, -10000, 20000])
        self.assertAlmostEqual(n_net, int(30000 / clk * sf), delta=sf // 200)
        self.assertLess(
            n_net, n_pos
        )  # the negative correction reduced clocked time, not ignored

    def test_negative_only_clocks_nothing(self):
        n, _, _ = self._clocked_samples([-5000, -5000])
        self.assertEqual(n, 0)


class TestFramePacket(unittest.TestCase):
    def test_default_total_cycles_is_pal(self):
        pkt = FramePacket(frame_id=5)
        self.assertEqual(pkt.total_cycles, 19656)
        self.assertEqual(pkt.ops, [])
        self.assertFalse(pkt.edited)

    def test_op_carries_delay(self):
        op = FrameOp(reg=4, val=0x21, delay_cycles=19656)
        self.assertEqual((op.reg, op.val, op.delay_cycles), (4, 0x21, 19656))


class TestAudioRenderBuffer(unittest.TestCase):
    def test_push_pop_in_order(self):
        buf = AudioRenderBuffer(max_frames=8)
        for i in range(5):
            self.assertTrue(
                buf.push_frame(FramePacket(frame_id=i, ops=[FrameOp(0, i)]))
            )
        out = [buf.pop_frame(timeout=0.1).frame_id for _ in range(5)]
        self.assertEqual(out, list(range(5)))

    def test_replace_lands_when_queued(self):
        buf = AudioRenderBuffer(max_frames=8)
        buf.push_frame(FramePacket(frame_id=0, ops=[FrameOp(0, 1)]))
        buf.push_frame(FramePacket(frame_id=1, ops=[FrameOp(0, 2)]))
        new_ops = [FrameOp(0, 0xFF, delay_cycles=19656)]
        self.assertTrue(buf.replace_frame(1, new_ops))
        pkt0 = buf.pop_frame()
        self.assertEqual(pkt0.frame_id, 0)
        pkt1 = buf.pop_frame()
        self.assertEqual(pkt1.ops, new_ops)
        self.assertTrue(pkt1.edited)

    def test_replace_misses_after_playhead(self):
        buf = AudioRenderBuffer(max_frames=8)
        buf.push_frame(FramePacket(frame_id=0))
        buf.pop_frame()
        self.assertFalse(buf.replace_frame(0, [FrameOp(0, 0)]))
        st = buf.state()
        self.assertEqual(st["edits_missed"], 1)
        self.assertEqual(st["edits_landed"], 0)

    def test_replace_misses_for_nonexistent_id(self):
        buf = AudioRenderBuffer(max_frames=8)
        buf.push_frame(FramePacket(frame_id=0))
        self.assertFalse(buf.replace_frame(99, [FrameOp(0, 0)]))

    def test_concurrent_drain_preserves_order(self):
        buf = AudioRenderBuffer(max_frames=20)
        popped = []

        def consumer():
            while True:
                pkt = buf.pop_frame(timeout=0.5)
                if pkt is None:
                    return
                popped.append(pkt.frame_id)

        t = threading.Thread(target=consumer)
        t.start()
        for i in range(50):
            buf.push_frame(FramePacket(frame_id=i))
        buf.close()
        t.join(timeout=2.0)
        self.assertEqual(popped, list(range(50)))

    def test_backpressure_blocks_producer(self):
        buf = AudioRenderBuffer(max_frames=2)

        def slow_consumer():
            for _ in range(5):
                time.sleep(0.02)
                buf.pop_frame(timeout=1.0)

        t = threading.Thread(target=slow_consumer)
        t.start()
        t0 = time.monotonic()
        for i in range(5):
            buf.push_frame(FramePacket(frame_id=i))
        elapsed = time.monotonic() - t0
        t.join()
        self.assertGreater(elapsed, 0.04)
        self.assertGreater(buf.state()["push_blocked_ms"], 0)


class TestSampleRing(unittest.TestCase):
    def test_write_then_read(self):
        ring = SampleRing(capacity_samples=100)
        ring.write(np.arange(40, dtype=np.int16))
        out = np.zeros(40, dtype=np.int16)
        n = ring.read(40, out)
        self.assertEqual(n, 40)
        self.assertTrue((out == np.arange(40)).all())

    def test_short_read_underrun_pads_zero(self):
        ring = SampleRing(capacity_samples=100)
        ring.write(np.full(20, 7, dtype=np.int16))
        out = np.zeros(50, dtype=np.int16)
        ring.read(50, out)
        self.assertEqual(ring.underruns, 1)
        self.assertTrue((out[:20] == 7).all())
        self.assertTrue((out[20:] == 0).all())

    def test_wraparound(self):
        ring = SampleRing(capacity_samples=100)
        ring.write(np.full(80, 1, dtype=np.int16))
        out = np.zeros(80, dtype=np.int16)
        ring.read(80, out)
        self.assertTrue(ring.write(np.full(50, 2, dtype=np.int16)))
        out2 = np.zeros(50, dtype=np.int16)
        n = ring.read(50, out2)
        self.assertEqual(n, 50)
        self.assertTrue((out2 == 2).all())


class TestWavSampleSink(unittest.TestCase):
    def test_write_collects_chunks(self):
        sink = WavSampleSink()
        sink.write(np.array([1, 2, 3], dtype=np.int16))
        sink.write(np.array([4, 5], dtype=np.int16))
        out = sink.to_array()
        self.assertEqual(out.tolist(), [1, 2, 3, 4, 5])

    def test_empty_returns_empty_array(self):
        sink = WavSampleSink()
        self.assertEqual(len(sink.to_array()), 0)

    def test_zero_length_write_is_noop(self):
        sink = WavSampleSink()
        sink.write(np.array([], dtype=np.int16))
        self.assertEqual(len(sink.to_array()), 0)


class TestDfToPackets(unittest.TestCase):
    def setUp(self):
        self.fm = FreqMapper(cents=50)
        self.reg_widths = {0: 2, 1: 2, 4: 1, 5: 1, 6: 1, 24: 1}

    def test_yields_packet_per_marker(self):
        df = pd.DataFrame(
            [
                (0, 0, 24, 15),
                (0, 0, 4, 0x21),
                (0, 19656, -128, 0),
                (0, 0, 4, 0x20),
                (0, 19656, -128, 0),
            ],
            columns=["op", "diff", "reg", "val"],
        )
        df["delay"] = df["diff"] * sidq()
        pkts = list(df_to_packets(df, self.reg_widths, self.fm))
        self.assertEqual(len(pkts), 2)
        self.assertEqual([p.frame_id for p in pkts], [0, 1])

    def test_two_byte_reg_expands(self):
        df = pd.DataFrame(
            [(0, 19656, 0, 60), (0, 19656, -128, 0)],
            columns=["op", "diff", "reg", "val"],
        )
        df["delay"] = df["diff"] * sidq()
        (pkt,) = df_to_packets(df, self.reg_widths, self.fm)
        self.assertEqual(len(pkt.ops), 2)
        self.assertEqual({op.reg for op in pkt.ops}, {0, 1})

    def test_trailing_partial_frame_yielded(self):
        df = pd.DataFrame([(0, 19656, 24, 15)], columns=["op", "diff", "reg", "val"])
        df["delay"] = df["diff"] * sidq()
        pkts = list(df_to_packets(df, self.reg_widths, self.fm))
        self.assertEqual(len(pkts), 1)


class TestRegStartInitOps(unittest.TestCase):
    def test_default_matches_legacy(self):
        rs = _default_reg_start()
        self.assertEqual(rs[6], 240)
        self.assertEqual(rs[3], 16)
        self.assertEqual(rs[24], 15)

    def test_init_ops_use_zero_delay(self):
        fm = FreqMapper(cents=50)
        ops = _reg_start_init_ops(_default_reg_start(), {}, fm)
        self.assertGreater(len(ops), 0)
        self.assertTrue(all(op.delay_cycles == 0 for op in ops))


class TestOpCollector(unittest.TestCase):
    def test_collects_writes(self):
        fm = FreqMapper(cents=10)
        sink = _OpCollector(fm)
        sink.write_register(4, 0x21)
        sink.write_register(5, 0x09)
        self.assertEqual(sink.collected, [(4, 0x21), (5, 0x09)])
        self.assertIs(sink.freq_mapper, fm)


class TestRenderToWav(unittest.TestCase):
    def test_round_trip_writes_valid_wav(self):
        df = pd.DataFrame(
            [
                (0, 0, 24, 15),
                (0, 0, 5, 0x09),
                (0, 0, 6, 0xA4),
                (0, 0, 0, 0x1500),
                (0, 0, 4, 0x21),
                (0, 19656, -128, 0),
                (0, 0, 0, 0x1700),
                (0, 19656, -128, 0),
            ],
            columns=["op", "diff", "reg", "val"],
        )
        df["delay"] = df["diff"] * sidq()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "render.wav")
            n = render_to_wav(
                df,
                path,
                reg_widths={0: 2, 1: 2, 4: 1, 5: 1, 6: 1, 24: 1},
                irq=19656,
                cents=50,
            )
            self.assertGreater(n, 0)
            rate, samples = scipy.io.wavfile.read(path)
            self.assertEqual(rate, 48000)
            self.assertEqual(len(samples), n)
            self.assertGreater(int(np.abs(samples).max()), 100)

    def test_render_per_voice_isolates_active_voices(self):
        # Only voice 1 is gated; voices 2 and 3 have no writes.
        df = pd.DataFrame(
            [
                (0, 0, 24, 15),
                (0, 0, 5, 0x09),
                (0, 0, 6, 0xF4),
                (0, 0, 0, 0x2000),
                (0, 0, 4, 0x21),
            ]
            + [(0, 19656, -128, 0)] * 6,
            columns=["op", "diff", "reg", "val"],
        )
        df["delay"] = df["diff"] * sidq()
        rw = {0: 2, 4: 1, 5: 1, 6: 1, 24: 1}
        voices = render_per_voice(df, reg_widths=rw, irq=19656, cents=50)
        self.assertEqual(set(voices), {0, 1, 2})

        # Difference cancels the chip's common-mode DC/idle floor. Soloing
        # voice 1 keeps its gated tone; soloing voice 2 or 3 mutes voice 1's
        # control reg (gate+waveform), so those render to the bare floor. A
        # large solo1-vs-solo2 difference with a near-zero solo2-vs-solo3
        # difference proves the per-voice mute isolates output.
        def dstd(x, y):
            n = min(len(x), len(y))
            return float(np.std(x[:n].astype(np.float64) - y[:n].astype(np.float64)))

        self.assertGreater(dstd(voices[0], voices[1]), 100.0)
        self.assertGreater(dstd(voices[0], voices[2]), 100.0)
        self.assertLess(dstd(voices[1], voices[2]), 10.0)

    def test_default_reg_start_used_when_none(self):
        df = pd.DataFrame(
            [(0, 19656, 24, 15), (0, 19656, -128, 0)],
            columns=["op", "diff", "reg", "val"],
        )
        df["delay"] = df["diff"] * sidq()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "default.wav")
            n = render_to_wav(df, path, reg_widths={24: 1}, irq=19656, cents=50)
            self.assertGreater(n, 0)
            self.assertTrue(os.path.exists(path))


class TestResidWorkerWithWavSink(unittest.TestCase):
    """End-to-end worker -> sink integration via the offline path."""

    def test_worker_drives_sink_to_completion(self):
        buf = AudioRenderBuffer(max_frames=10)
        sink = WavSampleSink()
        worker = ResidWorker(buf, sink=sink)
        worker.start()
        try:
            ops = [
                FrameOp(reg=24, val=15),
                FrameOp(reg=5, val=0x09),
                FrameOp(reg=6, val=0xA4),
                FrameOp(reg=0, val=0x00),
                FrameOp(reg=1, val=0x15),
                FrameOp(reg=4, val=0x21, delay_cycles=19656),
            ]
            buf.push_frame(FramePacket(frame_id=0, ops=ops))
        finally:
            buf.close()
            worker.join(timeout=5.0)
        self.assertEqual(worker.frames_rendered, 1)
        out = sink.to_array()
        self.assertGreater(len(out), 800)
        self.assertLess(len(out), 1000)


class TestPlayViaAplay(unittest.TestCase):
    """Exercise ``play_via_aplay``'s end-to-end frame loop with a
    captured-stdin stub aplay so no real audio device is needed."""

    def test_stub_aplay_receives_samples_and_on_frame_fires(self):
        from preframr_audio.audio_driver import play_via_aplay

        rows = [
            (0, 0, 24, 15),
            (0, 0, 5, 0x09),
            (0, 0, 6, 0xA4),
            (0, 0, 0, 0x1500),
            (0, 0, 4, 0x21),
            (0, 19656, -128, 0),
            (0, 19656, -128, 0),
            (0, 19656, -128, 0),
        ]
        df = pd.DataFrame(rows, columns=["op", "diff", "reg", "val"])
        df["delay"] = df["diff"] * sidq()
        reg_widths = {0: 2, 1: 2, 4: 1, 5: 1, 6: 1, 24: 1}

        with tempfile.TemporaryDirectory() as td:
            captured = os.path.join(td, "captured.raw")
            stub_aplay = ["sh", "-c", f"cat > {captured}"]
            seen = []

            def on_frame(pkt):
                seen.append(pkt.frame_id)

            n = play_via_aplay(
                df,
                reg_widths=reg_widths,
                irq=19656,
                cents=50,
                on_frame=on_frame,
                aplay_cmd=stub_aplay,
            )
            self.assertGreater(n, 0)
            self.assertGreater(len(seen), 0)
            with open(captured, "rb") as f:
                captured_bytes = f.read()
            self.assertEqual(len(captured_bytes), n * 2)
            self.assertGreater(n, 2500)
            self.assertLess(n, 3500)

    def test_mute_voices_silences_voice(self):
        """Muting voice 0 should null the captured audio for a song
        that only drives voice 0 (CTRL=0 is forced + voice writes are
        dropped, so the chip emits silence)."""
        from preframr_audio.audio_driver import play_via_aplay

        n_marker_frames = 30
        rows = [
            (0, 0, 24, 15),
            (0, 0, 5, 0x09),
            (0, 0, 6, 0xA4),
            (0, 0, 0, 0x1500),
            (0, 0, 4, 0x21),
        ] + [(0, 19656, -128, 0)] * n_marker_frames
        df = pd.DataFrame(rows, columns=["op", "diff", "reg", "val"])
        df["delay"] = df["diff"] * sidq()
        reg_widths = {0: 2, 1: 2, 4: 1, 5: 1, 6: 1, 24: 1}

        skip = 10000

        def _capture_rms(mute):
            with tempfile.TemporaryDirectory() as td:
                captured = os.path.join(td, "captured.raw")
                stub_aplay = ["sh", "-c", f"cat > {captured}"]
                play_via_aplay(
                    df,
                    reg_widths=reg_widths,
                    irq=19656,
                    cents=50,
                    aplay_cmd=stub_aplay,
                    mute_voices=mute,
                )
                raw = np.frombuffer(open(captured, "rb").read(), dtype=np.int16)
                body = raw[skip:] if len(raw) > skip else raw
                return float(np.sqrt(np.mean(body.astype(float) ** 2)))

        baseline = _capture_rms(mute=None)
        muted = _capture_rms(mute={0})
        self.assertGreater(baseline, 1000)
        self.assertLess(muted, baseline / 2)

    def test_should_quit_aborts_producer_loop(self):
        """Passing should_quit -> True after the first frame should
        cut off the stream early."""
        from preframr_audio.audio_driver import play_via_aplay

        rows = [
            (0, 0, 24, 15),
            (0, 0, 5, 0x09),
            (0, 0, 6, 0xA4),
            (0, 0, 0, 0x1500),
            (0, 0, 4, 0x21),
        ] + [(0, 19656, -128, 0)] * 50
        df = pd.DataFrame(rows, columns=["op", "diff", "reg", "val"])
        df["delay"] = df["diff"] * sidq()
        reg_widths = {0: 2, 1: 2, 4: 1, 5: 1, 6: 1, 24: 1}

        seen = []

        def on_frame(pkt):
            seen.append(pkt.frame_id)

        def should_quit():
            return len(seen) >= 2

        with tempfile.TemporaryDirectory() as td:
            captured = os.path.join(td, "captured.raw")
            stub_aplay = ["sh", "-c", f"cat > {captured}"]
            play_via_aplay(
                df,
                reg_widths=reg_widths,
                irq=19656,
                cents=50,
                on_frame=on_frame,
                should_quit=should_quit,
                aplay_cmd=stub_aplay,
            )
        self.assertLess(len(seen), 10)


class TestAsidEncoding(unittest.TestCase):
    def test_round_trip(self):
        regs = {0: 0x40, 1: 0x12, 4: 0x21, 24: 0x0F}
        body = encode_asid_update(regs)
        self.assertEqual(body[0], ASID_MANID)
        self.assertEqual(body[1], ASID_UPDATE)
        self.assertEqual(decode_asid_update(body), regs)

    def test_high_bit_round_trips(self):
        regs = {0: 0xFF, 1: 0x80, 2: 0x7F}
        body = encode_asid_update(regs)
        self.assertEqual(decode_asid_update(body), regs)

    def test_unmapped_reg_dropped(self):
        body = encode_asid_update({99: 0x42, 0: 0x40})
        self.assertEqual(decode_asid_update(body), {0: 0x40})

    def test_empty(self):
        body = encode_asid_update({})
        self.assertEqual(body[2:6], [0, 0, 0, 0])
        self.assertEqual(body[6:10], [0, 0, 0, 0])
        self.assertEqual(decode_asid_update(body), {})

    def test_decode_rejects_non_update_cmd(self):
        self.assertEqual(decode_asid_update([ASID_MANID, ASID_START]), {})
        self.assertEqual(decode_asid_update([ASID_MANID, ASID_STOP]), {})

    def test_slot_map_covers_full_sid_register_space(self):
        self.assertEqual(set(ASID_SLOT_TO_REG.values()), set(range(25)))
        self.assertEqual(set(ASID_REG_TO_SLOT.keys()), set(range(25)))


class _FakeMidiOut:
    """In-process stand-in for ``rtmidi.MidiOut`` so we can unit-test
    the worker's deltas + framing without touching real MIDI."""

    def __init__(self):
        self.sent = []

    def send_message(self, msg):
        self.sent.append(list(msg))


class TestAsidWorker(unittest.TestCase):
    def _drive_worker(self, frames):
        buf = AudioRenderBuffer(max_frames=20)
        out = _FakeMidiOut()
        w = AsidWorker(buf, out)
        w.start()
        try:
            for fid, ops in frames:
                buf.push_frame(FramePacket(frame_id=fid, ops=ops))
        finally:
            buf.close()
            w.join(timeout=2.0)
        return out.sent

    def test_first_frame_sends_all_writes(self):
        sent = self._drive_worker(
            [(0, [FrameOp(reg=0, val=0x40), FrameOp(reg=4, val=0x21)])]
        )
        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertEqual(msg[0], 0xF0)
        self.assertEqual(msg[-1], 0xF7)
        decoded = decode_asid_update(msg[1:-1])
        self.assertEqual(decoded, {0: 0x40, 4: 0x21})

    def test_only_changed_regs_in_subsequent_frame(self):
        sent = self._drive_worker(
            [
                (0, [FrameOp(reg=0, val=0x40), FrameOp(reg=4, val=0x21)]),
                (1, [FrameOp(reg=0, val=0x60), FrameOp(reg=4, val=0x21)]),
            ]
        )
        self.assertEqual(len(sent), 2)
        decoded1 = decode_asid_update(sent[1][1:-1])
        self.assertEqual(decoded1, {0: 0x60})

    def test_unchanged_frame_emits_no_packet(self):
        sent = self._drive_worker(
            [
                (0, [FrameOp(reg=0, val=0x40)]),
                (1, [FrameOp(reg=0, val=0x40)]),
            ]
        )
        self.assertEqual(len(sent), 1)


_E2E_REASON = (
    "ASID end-to-end test requires Linux + python-rtmidi + virtual MIDI "
    "port support. Run inside the docker image with --device /dev/snd."
)


@unittest.skipUnless(_HAVE_RTMIDI and platform.system() == "Linux", _E2E_REASON)
class TestAsidEndToEnd(unittest.TestCase):
    """Linux-only round-trip: ``AsidMidiDriver`` -> virtual MIDI port
    -> ``AsidServer`` (which renders via local resid). Compare server
    samples against the ``render_to_wav`` reference. ASID is per-frame
    so we use a static-note fixture where intra-frame timing doesn't
    diverge much from per-op clocking; FFT correlation tolerance
    """

    def test_round_trip_via_virtual_port(self):
        port_name = f"preframr-asid-test-{os.getpid()}"
        reg_widths = {0: 2, 1: 2, 4: 1, 5: 1, 6: 1, 24: 1}
        irq = 19656
        cents = 50
        rows = [
            (0, 0, 24, 15),
            (0, 0, 5, 0x09),
            (0, 0, 6, 0xA4),
            (0, 0, 0, 0x1500),
            (0, 0, 4, 0x21),
        ]
        for _ in range(6):
            rows.append((0, irq, -128, 0))
        df = pd.DataFrame(rows, columns=["op", "diff", "reg", "val"])
        df["delay"] = df["diff"] * sidq()

        with tempfile.TemporaryDirectory() as td:
            ref_path = os.path.join(td, "ref.wav")
            render_to_wav(
                df.copy(),
                ref_path,
                reg_widths=reg_widths,
                irq=irq,
                cents=cents,
            )
            _ref_rate, ref_samples = scipy.io.wavfile.read(ref_path)

        driver = AsidMidiDriver(port_name=port_name, virtual=True)
        try:
            driver.open()
        except (RuntimeError, OSError) as e:
            self.skipTest(f"virtual MIDI output port unavailable: {e}")

        time.sleep(0.1)

        try:
            server = AsidServer(port_pattern=port_name, cycles_per_frame=irq)
            try:
                server.open()
            except RuntimeError as e:
                self.skipTest(f"server cannot find driver's port: {e}")

            try:
                fm = FreqMapper(cents=cents)
                init_ops = _reg_start_init_ops(_default_reg_start(), reg_widths, fm)
                driver.push_frame(-1, init_ops, total_cycles=0)
                clock_freq = 985248
                for pkt in df_to_packets(
                    df,
                    reg_widths,
                    fm,
                    irq_cycles=irq,
                    clock_frequency=clock_freq,
                ):
                    driver.push_frame(
                        pkt.frame_id, pkt.ops, total_cycles=pkt.total_cycles
                    )
                deadline = time.monotonic() + 5.0
                while driver.lookahead_frames() > 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                server.drain(max_wait=5.0)
            finally:
                server.close()
        finally:
            driver.close()

        srv_samples = server.collected_samples()
        self.assertGreater(len(srv_samples), 0, "server received no audio")

        n = min(len(ref_samples), len(srv_samples))
        ref_fft = np.abs(np.fft.rfft(ref_samples[:n].astype(np.float32)))
        srv_fft = np.abs(np.fft.rfft(srv_samples[:n].astype(np.float32)))
        if ref_fft.std() == 0 or srv_fft.std() == 0:
            self.fail("one of the renders produced silence")
        corr = float(np.corrcoef(ref_fft, srv_fft)[0, 1])
        self.assertGreater(
            corr,
            0.85,
            f"ASID round-trip diverged (FFT corr={corr:.3f}); "
            f"ref={len(ref_samples)} srv={len(srv_samples)} "
            f"packets={server.received_packets} writes={server.applied_writes}",
        )


if __name__ == "__main__":
    unittest.main()
