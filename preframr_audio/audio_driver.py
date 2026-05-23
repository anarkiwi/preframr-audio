"""Audio driver: ``AudioRenderBuffer`` + resid worker + two output sinks."""

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from pyresidfp import SoundInterfaceDevice
    from pyresidfp.sound_interface_device import ChipModel

    _HAVE_RESID = True
except ImportError:
    _HAVE_RESID = False
    SoundInterfaceDevice = None  # type: ignore
    ChipModel = None  # type: ignore

try:
    import sounddevice as sd

    _HAVE_SD = True
except ImportError:
    _HAVE_SD = False
    sd = None  # type: ignore

try:
    import rtmidi  # noqa: WPS433  -- conditional optional dep

    _HAVE_RTMIDI = hasattr(rtmidi, "MidiOut") and hasattr(rtmidi, "MidiIn")
    if not _HAVE_RTMIDI:
        rtmidi = None  # type: ignore
except ImportError:
    _HAVE_RTMIDI = False
    rtmidi = None  # type: ignore


ASID_MANID = 0x2D
ASID_START = 0x4C
ASID_STOP = 0x4D
ASID_UPDATE = 0x4E

ASID_SLOT_TO_REG = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 12,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 19,
    17: 20,
    18: 21,
    19: 22,
    20: 23,
    21: 24,
    22: 4,
    23: 11,
    24: 18,
}
ASID_REG_TO_SLOT = {reg: slot for slot, reg in ASID_SLOT_TO_REG.items()}


def encode_asid_update(reg_to_val: Dict[int, int]) -> List[int]:
    """Pack a ``{reg: val}`` dict into an ASID UpdateReg SysEx body
    (the bytes between the framing ``0xF0`` and ``0xF7``).
    """
    masks = [0, 0, 0, 0]
    msbs = [0, 0, 0, 0]
    vals: List[int] = []
    for slot in sorted(ASID_SLOT_TO_REG):
        reg = ASID_SLOT_TO_REG[slot]
        if reg not in reg_to_val:
            continue
        val = int(reg_to_val[reg]) & 0xFF
        meta_byte = slot // 7
        meta_bit = slot % 7
        masks[meta_byte] |= 1 << meta_bit
        if val & 0x80:
            msbs[meta_byte] |= 1 << meta_bit
        vals.append(val & 0x7F)
    return [ASID_MANID, ASID_UPDATE] + masks + msbs + vals


def decode_asid_update(sysex_body) -> Dict[int, int]:
    """Inverse of ``encode_asid_update``. ``sysex_body`` is the bytes
    between ``0xF0`` and ``0xF7`` (i.e. starting with ``ASID_MANID``).
    Returns ``{reg: val}`` for the slots present in the masks.
    Returns ``{}`` for non-UpdateReg cmds.
    """
    if len(sysex_body) < 10 or sysex_body[0] != ASID_MANID:
        return {}
    if sysex_body[1] != ASID_UPDATE:
        return {}
    masks = sysex_body[2:6]
    msbs = sysex_body[6:10]
    vals = sysex_body[10:]
    reg_to_val: Dict[int, int] = {}
    val_idx = 0
    for slot in sorted(ASID_SLOT_TO_REG):
        meta_byte = slot // 7
        meta_bit = slot % 7
        if not (masks[meta_byte] & (1 << meta_bit)):
            continue
        if val_idx >= len(vals):
            break
        v = vals[val_idx] & 0x7F
        if msbs[meta_byte] & (1 << meta_bit):
            v |= 0x80
        reg_to_val[ASID_SLOT_TO_REG[slot]] = v
        val_idx += 1
    return reg_to_val


@dataclass
class FrameOp:
    """One register write within a frame plus the cycle gap that
    follows before the next op. Sub-frame timing matters for audio
    fidelity: filter sweeps, gate edges, and PWM modulation all
    depend on when within the frame each write lands. ``reg < 0``
    is permitted (e.g. ``DELAY_REG`` carrier rows) and treated as
    """

    reg: int
    val: int
    delay_cycles: int = 0


@dataclass
class FramePacket:
    """One PAL / NTSC frame's worth of register writes. ``frame_id``
    is monotonic across the run and stable across edits -- it's the
    key the tracker view uses to address a specific upcoming frame.
    """

    frame_id: int
    ops: List[FrameOp] = field(default_factory=list)
    total_cycles: int = 19656
    edited: bool = False


class AudioRenderBuffer:
    """Thread-safe per-frame queue with addressable replace."""

    def __init__(self, max_frames: int = 200):
        self.max_frames = max_frames
        self._deque: Deque[FramePacket] = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.playhead_frame_id = -1
        self.pushed = 0
        self.popped = 0
        self.edits_landed = 0
        self.edits_missed = 0
        self.push_blocked_ns = 0

    def push_frame(self, packet: FramePacket, timeout: float = 1.0) -> bool:
        """Append a frame. Blocks up to ``timeout`` seconds if full.
        Returns False if the buffer is closed or push timed out.
        """
        t0 = time.monotonic_ns()
        with self._not_full:
            while len(self._deque) >= self.max_frames and not self._closed:
                if not self._not_full.wait(timeout=timeout):
                    return False
            if self._closed:
                return False
            self._deque.append(packet)
            self.pushed += 1
            self._not_empty.notify()
        self.push_blocked_ns += time.monotonic_ns() - t0
        return True

    def pop_frame(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Pop oldest frame. Blocks up to ``timeout`` seconds. Returns
        None if buffer is empty + closed, or if timed out.
        """
        with self._not_empty:
            while not self._deque and not self._closed:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if not self._deque:
                return None
            pkt = self._deque.popleft()
            self.playhead_frame_id = pkt.frame_id
            self.popped += 1
            self._not_full.notify()
            return pkt

    def replace_frame(self, frame_id: int, ops: List["FrameOp"]) -> bool:
        """Mutate a still-buffered frame in place. Returns True if the
        edit landed (frame found in queue), False if the frame already
        passed the playhead (or the frame_id isn't queued).
        """
        with self._lock:
            if frame_id <= self.playhead_frame_id:
                self.edits_missed += 1
                return False
            for pkt in self._deque:
                if pkt.frame_id == frame_id:
                    pkt.ops = list(ops)
                    pkt.edited = True
                    self.edits_landed += 1
                    return True
            self.edits_missed += 1
            return False

    def close(self):
        """Signal end-of-stream; pop_frame will return None once
        drained."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def state(self) -> Dict:
        """Snapshot for telemetry / UI display."""
        with self._lock:
            return {
                "queued": len(self._deque),
                "max": self.max_frames,
                "pushed": self.pushed,
                "popped": self.popped,
                "playhead_id": self.playhead_frame_id,
                "edits_landed": self.edits_landed,
                "edits_missed": self.edits_missed,
                "push_blocked_ms": self.push_blocked_ns / 1e6,
                "closed": self._closed,
            }


class WavSampleSink:
    """Sample sink that just appends to a list; ``to_array`` concatenates
    for one-shot wav writes. Used by ``render_to_wav`` to take the same
    ``ResidWorker`` -> sink path as the real-time driver but without the
    SampleRing's threading + backpressure -- the whole thing runs
    sequentially in offline mode.
    """

    def __init__(self):
        self._chunks: List[np.ndarray] = []
        self.underruns = 0

    def write(self, samples: np.ndarray, timeout: float = 0.5) -> bool:
        del timeout
        if len(samples):
            self._chunks.append(np.asarray(samples, dtype=np.int16))
        return True

    def occupancy(self) -> Tuple[int, int]:
        n = sum(len(c) for c in self._chunks)
        return (n, n)

    def to_array(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._chunks)


class SampleRing:
    """Thread-safe int16 sample ring with backpressure."""

    def __init__(self, capacity_samples: int):
        self.cap = capacity_samples
        self.buf = np.zeros(self.cap, dtype=np.int16)
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0
        self._lock = threading.Lock()
        self._space_available = threading.Condition(self._lock)
        self.underruns = 0

    def write(self, samples: np.ndarray, timeout: float = 0.5) -> bool:
        n = len(samples)
        if n > self.cap:
            return False
        with self._space_available:
            while self.cap - self.size < n:
                if not self._space_available.wait(timeout=timeout):
                    return False
            end = (self.write_pos + n) % self.cap
            if end > self.write_pos:
                self.buf[self.write_pos : end] = samples
            else:
                tail = self.cap - self.write_pos
                self.buf[self.write_pos :] = samples[:tail]
                self.buf[:end] = samples[tail:]
            self.write_pos = end
            self.size += n
        return True

    def read(self, n: int, out: np.ndarray) -> int:
        """Non-blocking read up to ``n`` samples into ``out``. Returns
        actual sample count read; pads with 0 if short."""
        with self._lock:
            avail = min(n, self.size)
            end = (self.read_pos + avail) % self.cap
            if end > self.read_pos:
                out[:avail] = self.buf[self.read_pos : end]
            elif avail > 0:
                tail = self.cap - self.read_pos
                out[:tail] = self.buf[self.read_pos :]
                out[tail:avail] = self.buf[:end]
            if avail < n:
                out[avail:n] = 0
                self.underruns += 1
            self.read_pos = end
            self.size -= avail
            self._space_available.notify()
        return avail

    def occupancy(self) -> Tuple[int, int]:
        with self._lock:
            return (self.size, self.cap)


class ResidWorker(threading.Thread):
    """Pops frames from AudioRenderBuffer, applies register writes,
    clocks resid for the frame's cycles, writes resulting samples to
    the SampleRing.
    """

    def __init__(
        self,
        buffer: AudioRenderBuffer,
        sink,
        chip_model: str = "MOS8580",
        on_frame_done: Optional[Callable[[FramePacket, int], None]] = None,
    ):
        super().__init__(daemon=True)
        if not _HAVE_RESID:
            raise RuntimeError("pyresidfp not installed; pip install pyresidfp")
        self.buffer = buffer
        self.sink = sink
        self.sid = SoundInterfaceDevice(model=getattr(ChipModel, chip_model))
        self.sid.reset()
        for _r in range(25):
            self.sid.write_register(_r, 0)
        self.sid.clock(timedelta(seconds=0.05))
        self.on_frame_done = on_frame_done
        self.stop_evt = threading.Event()
        self.frames_rendered = 0

    @property
    def sampling_frequency(self) -> int:
        return int(self.sid.sampling_frequency)

    @property
    def clock_frequency(self) -> int:
        return int(self.sid.clock_frequency)

    def run(self):
        while not self.stop_evt.is_set():
            pkt = self.buffer.pop_frame(timeout=0.1)
            if pkt is None:
                if self.buffer.state()["closed"]:
                    return
                continue
            frame_samples = 0
            for op in pkt.ops:
                if op.reg >= 0:
                    self.sid.write_register(int(op.reg), int(op.val))
                if op.delay_cycles > 0:
                    seconds = op.delay_cycles / self.clock_frequency
                    samples = self.sid.clock(timedelta(seconds=seconds))
                    samples = np.asarray(samples, dtype=np.int16)
                    if len(samples):
                        self.sink.write(samples)
                        frame_samples += len(samples)
            self.frames_rendered += 1
            if self.on_frame_done:
                self.on_frame_done(pkt, frame_samples)

    def stop(self):
        self.stop_evt.set()


class ResidAlsaDriver:  # pragma: no cover
    """sounddevice (PortAudio) output stream driven by a ResidWorker."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        device: Optional[int] = None,
        block_size: int = 512,
        ring_seconds: float = 1.5,
        max_frames: int = 100,
        chip_model: str = "MOS8580",
    ):
        if not _HAVE_SD:
            raise RuntimeError("sounddevice not installed; pip install sounddevice")
        if not _HAVE_RESID:
            raise RuntimeError("pyresidfp not installed; pip install pyresidfp")
        self.sample_rate = sample_rate
        self.device = device
        self.block_size = block_size
        self.ring_seconds = ring_seconds
        self.max_frames = max_frames
        self.chip_model = chip_model
        self.buffer: Optional[AudioRenderBuffer] = None
        self.ring: Optional[SampleRing] = None
        self.worker: Optional[ResidWorker] = None
        self.stream: Optional["sd.OutputStream"] = None

    def open(self):
        self.buffer = AudioRenderBuffer(max_frames=self.max_frames)
        probe = SoundInterfaceDevice(model=getattr(ChipModel, self.chip_model))
        sr = self.sample_rate or int(probe.sampling_frequency)
        self.sample_rate = sr
        self.ring = SampleRing(int(sr * self.ring_seconds))
        self.worker = ResidWorker(
            self.buffer, sink=self.ring, chip_model=self.chip_model
        )
        self.worker.start()

        def _audio_cb(outdata, frames, _time, status):
            if status:
                pass
            mono = np.empty(frames, dtype=np.int16)
            self.ring.read(frames, mono)
            if outdata.dtype == np.int16:
                if outdata.shape[1] == 1:
                    outdata[:, 0] = mono
                else:
                    outdata[:, :] = mono[:, None]
            else:
                norm = mono.astype(np.float32) / 32768.0
                if outdata.shape[1] == 1:
                    outdata[:, 0] = norm
                else:
                    outdata[:, :] = norm[:, None]

        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=self.block_size,
            device=self.device,
            callback=_audio_cb,
            latency="low",
        )
        self.stream.start()

    def close(self):
        if self.buffer:
            self.buffer.close()
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=1.0)
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stream = None
        self.worker = None

    def push_frame(
        self, frame_id: int, ops: List["FrameOp"], total_cycles: int = 19656
    ) -> bool:
        return self.buffer.push_frame(
            FramePacket(frame_id=frame_id, ops=list(ops), total_cycles=total_cycles)
        )

    def replace_frame(self, frame_id: int, ops: List["FrameOp"]) -> bool:
        return self.buffer.replace_frame(frame_id, ops)

    def lookahead_frames(self) -> int:
        return self.buffer.state()["queued"]

    def latency_ms(self) -> float:
        st = self.buffer.state()
        ring_size, _ = self.ring.occupancy()
        ring_ms = ring_size / self.sample_rate * 1000.0
        frames_ms = st["queued"] * (19656 / 985248) * 1000.0
        out_ms = (self.stream.latency * 1000.0) if self.stream else 0.0
        return frames_ms + ring_ms + out_ms

    def status(self) -> Dict:
        st = dict(self.buffer.state()) if self.buffer else {}
        if self.ring:
            sz, cap = self.ring.occupancy()
            st["ring_samples"] = sz
            st["ring_cap_samples"] = cap
            st["ring_pct"] = round(100 * sz / cap, 1) if cap else 0.0
            st["ring_underruns"] = self.ring.underruns
        if self.worker:
            st["worker_frames"] = self.worker.frames_rendered
            st["sample_rate"] = self.worker.sampling_frequency
        if self.stream:
            st["pa_latency_ms"] = round(self.stream.latency * 1000.0, 1)
        return st


class AsidWorker(threading.Thread):  # pragma: no cover
    """Pops frames from the buffer, packs only *changed* registers into
    an ASID UpdateReg SysEx packet, sends via MIDI. Per-reg last-sent
    state is kept so the SysEx carries deltas (matches what an Asid
    cart on a C64 expects -- the IRQ handler only wants the bytes
    that actually changed since the previous frame).
    """

    def __init__(self, buffer, midi_out, on_frame_done=None):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.midi_out = midi_out
        self.last_reg: Dict[int, int] = {}
        self.stop_evt = threading.Event()
        self.frames_rendered = 0
        self.packets_sent = 0
        self.on_frame_done = on_frame_done

    def run(self):
        while not self.stop_evt.is_set():
            pkt = self.buffer.pop_frame(timeout=0.1)
            if pkt is None:
                if self.buffer.state()["closed"]:
                    return
                continue
            frame_regs = dict(self.last_reg)
            for op in pkt.ops:
                if op.reg < 0 or op.reg not in ASID_REG_TO_SLOT:
                    continue
                frame_regs[op.reg] = int(op.val) & 0xFF
            deltas = {r: v for r, v in frame_regs.items() if self.last_reg.get(r) != v}
            if deltas:
                body = encode_asid_update(deltas)
                self.midi_out.send_message([0xF0] + body + [0xF7])
                self.packets_sent += 1
            self.last_reg = frame_regs
            self.frames_rendered += 1
            if self.on_frame_done:
                self.on_frame_done(pkt)


class AsidMidiDriver:  # pragma: no cover
    """Real-time ASID-over-MIDI driver. Same shape as ``ResidAlsaDriver``
    but emits SysEx packets to a MIDI port instead of audio samples to
    a sounddevice stream.
    """

    def __init__(
        self,
        port_name: str = "preframr-asid",
        virtual: bool = False,
        port_index: Optional[int] = None,
        max_frames: int = 200,
    ):
        if not _HAVE_RTMIDI:
            raise RuntimeError(
                "python-rtmidi not installed (or shadowed by another "
                "rtmidi flavour); ASID driver unavailable"
            )
        self.port_name = port_name
        self.virtual = virtual
        self.port_index = port_index
        self.max_frames = max_frames
        self.buffer: Optional[AudioRenderBuffer] = None
        self.midi_out = None
        self.worker: Optional[AsidWorker] = None
        self.opened_port_name: Optional[str] = None

    def open(self):
        self.buffer = AudioRenderBuffer(max_frames=self.max_frames)
        self.midi_out = rtmidi.MidiOut()  # pylint: disable=no-member
        if self.virtual:
            self.midi_out.open_virtual_port(self.port_name)
            self.opened_port_name = self.port_name
        else:
            ports = self.midi_out.get_ports()
            target_idx = self.port_index
            if target_idx is None:
                for i, p in enumerate(ports):
                    if self.port_name in p:
                        target_idx = i
                        break
                else:
                    raise RuntimeError(
                        f"no MIDI output port matching {self.port_name!r}; "
                        f"available: {ports}"
                    )
            self.midi_out.open_port(target_idx)
            self.opened_port_name = ports[target_idx]
        self.midi_out.send_message([0xF0, ASID_MANID, ASID_START, 0xF7])
        self.worker = AsidWorker(self.buffer, self.midi_out)
        self.worker.start()

    def close(self):
        if self.buffer is not None:
            self.buffer.close()
        if self.worker is not None:
            self.worker.stop_evt.set()
            self.worker.join(timeout=1.0)
            self.worker = None
        if self.midi_out is not None:
            try:
                self.midi_out.send_message([0xF0, ASID_MANID, ASID_STOP, 0xF7])
            except (RuntimeError, OSError):
                pass
            try:
                self.midi_out.close_port()
            except (RuntimeError, OSError):
                pass
            del self.midi_out
            self.midi_out = None

    def push_frame(
        self, frame_id: int, ops: List["FrameOp"], total_cycles: int = 19656
    ) -> bool:
        if self.buffer is None:
            return False
        return self.buffer.push_frame(
            FramePacket(frame_id=frame_id, ops=list(ops), total_cycles=total_cycles)
        )

    def replace_frame(self, frame_id: int, ops: List["FrameOp"]) -> bool:
        if self.buffer is None:
            return False
        return self.buffer.replace_frame(frame_id, ops)

    def lookahead_frames(self) -> int:
        return self.buffer.state()["queued"] if self.buffer else 0

    def status(self) -> Dict:
        st = dict(self.buffer.state()) if self.buffer else {}
        if self.worker:
            st["worker_frames"] = self.worker.frames_rendered
            st["packets_sent"] = self.worker.packets_sent
        st["port"] = self.opened_port_name
        return st


class AsidServer:  # pragma: no cover
    """Receives ASID SysEx via MIDI, decodes, applies the resulting
    register writes to a local resid SID, captures samples to a
    ``WavSampleSink``. The "asid server application" reference
    implementation: the same logic an Asid cart on a C64 runs in its
    IRQ handler, but sample-rendered via pyresidfp instead of by the
    """

    def __init__(
        self,
        port_pattern: str = "preframr-asid",
        cycles_per_frame: int = 19656,
        chip_model: str = "MOS8580",
        idle_timeout: float = 1.0,
    ):
        if not _HAVE_RTMIDI:
            raise RuntimeError("python-rtmidi not installed")
        if not _HAVE_RESID:
            raise RuntimeError("pyresidfp not installed")
        self.port_pattern = port_pattern
        self.cycles_per_frame = cycles_per_frame
        self.chip_model = chip_model
        self.idle_timeout = idle_timeout
        self.midi_in = None
        self.queue: "Queue" = None  # type: ignore  -- set in open()
        self.worker = None
        self.sink: Optional[WavSampleSink] = None
        self.sid = None
        self.opened_port_name: Optional[str] = None
        self.received_packets = 0
        self.applied_writes = 0
        self._stop_evt = threading.Event()

    def open(self):
        import queue as _queue  # noqa: WPS433 -- avoid module-name shadow

        self.queue = _queue.Queue()
        self.midi_in = rtmidi.MidiIn()  # pylint: disable=no-member
        self.midi_in.ignore_types(sysex=False, timing=True, active_sense=True)
        ports = self.midi_in.get_ports()
        target_idx = None
        for i, p in enumerate(ports):
            if self.port_pattern in p:
                target_idx = i
                break
        if target_idx is None:
            raise RuntimeError(
                f"no MIDI input port matching {self.port_pattern!r}; "
                f"available: {ports}"
            )
        self.midi_in.open_port(target_idx)
        self.opened_port_name = ports[target_idx]
        self.sink = WavSampleSink()
        self.sid = SoundInterfaceDevice(model=getattr(ChipModel, self.chip_model))
        self.midi_in.set_callback(self._on_midi)
        self.worker = threading.Thread(target=self._render_loop, daemon=True)
        self.worker.start()

    def close(self):
        self._stop_evt.set()
        if self.queue is not None:
            self.queue.put(None)
        if self.midi_in is not None:
            try:
                self.midi_in.close_port()
            except (RuntimeError, OSError):
                pass
            del self.midi_in
            self.midi_in = None
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            self.worker = None

    def drain(self, max_wait: float = 5.0) -> bool:
        """Wait until the queue has been idle for ``idle_timeout``
        (no new SysEx received) or ``max_wait`` elapses. Use after
        the driver's last push to be sure all packets landed.
        """
        deadline = time.monotonic() + max_wait
        last_count = -1
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if self.received_packets != last_count:
                last_count = self.received_packets
                last_change = time.monotonic()
            elif time.monotonic() - last_change > self.idle_timeout:
                return True
        return False

    def _on_midi(self, message_and_dt, _data=None):
        msg = message_and_dt[0]
        if not msg or msg[0] != 0xF0 or msg[-1] != 0xF7:
            return
        if len(msg) < 4 or msg[1] != ASID_MANID:
            return
        cmd = msg[2]
        if cmd == ASID_START:
            self.queue.put(("START", None))
        elif cmd == ASID_STOP:
            self.queue.put(("STOP", None))
        elif cmd == ASID_UPDATE:
            body = msg[1:-1]
            reg_to_val = decode_asid_update(body)
            self.queue.put(("UPDATE", reg_to_val))
            self.received_packets += 1

    def _render_loop(self):
        seconds_per_frame = self.cycles_per_frame / self.sid.clock_frequency
        while not self._stop_evt.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if item is None:
                return
            kind, payload = item
            if kind == "STOP":
                return
            if kind == "START":
                self.sid = SoundInterfaceDevice(
                    model=getattr(ChipModel, self.chip_model)
                )
                continue
            if kind == "UPDATE":
                for reg, val in payload.items():
                    self.sid.write_register(int(reg), int(val))
                    self.applied_writes += 1
                samples = self.sid.clock(timedelta(seconds=seconds_per_frame))
                samples = np.asarray(samples, dtype=np.int16)
                if len(samples):
                    self.sink.write(samples)

    def collected_samples(self) -> np.ndarray:
        return (
            self.sink.to_array()
            if self.sink is not None
            else np.zeros(0, dtype=np.int16)
        )

    @property
    def sample_rate(self) -> int:
        return int(self.sid.sampling_frequency) if self.sid else 0


_FRAME_REG = -128
_DELAY_REG = -127
_PAL_CYCLES_PER_SEC = 985248


class _OpCollector:
    """SID-shaped sink used with ``preframr_audio.sidwav.write_reg`` so the
    producer reuses the same 2-byte-reg / freq_mapper expansion path
    as the wav-render reference. Captures the expanded raw writes
    instead of poking a real chip.
    """

    def __init__(self, freq_mapper):
        self.freq_mapper = freq_mapper
        self.collected: List[Tuple[int, int]] = []

    def write_register(self, reg, val):
        self.collected.append((int(reg), int(val)))


def df_to_packets(
    df,
    reg_widths,
    freq_mapper,
    irq_cycles: int = 19656,
    clock_frequency: int = _PAL_CYCLES_PER_SEC,
    frame_id_start: int = 0,
):
    """Walk a prepared-for-audio df and yield ``FramePacket`` objects."""
    from preframr_audio.sidwav import write_reg  # noqa: WPS433

    fid = frame_id_start
    ops: List[FrameOp] = []
    for row in df.itertuples():
        reg = int(row.reg)
        delay_seconds = float(getattr(row, "delay", 0.0) or 0.0)
        delay_cycles = int(round(delay_seconds * clock_frequency))
        if reg == _FRAME_REG or reg == _DELAY_REG:
            if ops:
                ops[-1].delay_cycles += delay_cycles
                yield FramePacket(frame_id=fid, ops=ops, total_cycles=irq_cycles)
                fid += 1
                ops = []
            else:
                yield FramePacket(
                    frame_id=fid,
                    ops=[FrameOp(reg=-1, val=0, delay_cycles=delay_cycles)],
                    total_cycles=irq_cycles,
                )
                fid += 1
            continue
        sink = _OpCollector(freq_mapper)
        try:
            write_reg(sink, reg, int(row.val), reg_widths)
        except (KeyError, ValueError, IndexError):
            continue
        if not sink.collected:
            continue
        last_idx = len(sink.collected) - 1
        for i, (raw_reg, raw_val) in enumerate(sink.collected):
            ops.append(
                FrameOp(
                    reg=raw_reg,
                    val=raw_val,
                    delay_cycles=delay_cycles if i == last_idx else 0,
                )
            )
    if ops:
        yield FramePacket(frame_id=fid, ops=ops, total_cycles=irq_cycles)


def _default_reg_start():
    """Match the prior ``sidwav.write_samples`` defaults: max sustain
    on all voices, 50% PWM, ``MODE_VOL=15``. Used by both wav and
    real-time entries so initial chip state is identical to the
    legacy renderer.
    """
    from preframr_audio._sid_constants import (  # noqa: WPS433
        MODE_VOL_REG,
        VOICES,
        VOICE_REG_SIZE,
    )

    rs: Dict[int, int] = {}
    for v in range(VOICES):
        offset = v * VOICE_REG_SIZE
        rs[6 + offset] = 240
        rs[3 + offset] = 16
    rs[MODE_VOL_REG] = 15
    return rs


def _reg_start_init_ops(reg_start, reg_widths, freq_mapper):
    """Build the FrameOps that apply ``reg_start`` writes to the SID
    via the same ``write_reg`` expansion the legacy wav-render path
    used. Used as the contents of a synthetic ``frame_id=-1`` packet
    so downstream the chip sees identical initial state.
    """
    from preframr_audio.sidwav import write_reg  # noqa: WPS433

    sink = _OpCollector(freq_mapper)
    for reg, val in sorted(reg_start.items()):
        write_reg(sink, reg, int(val), reg_widths)
    return [FrameOp(reg=r, val=v, delay_cycles=0) for r, v in sink.collected]


def _drive(
    buffer,
    df,
    reg_widths,
    freq_mapper,
    irq,
    reg_start,
    clock_frequency: int,
    progress_label: str = "",
):
    """Shared producer loop used by both wav and real-time entries.
    Pushes a synthetic ``frame_id=-1`` init packet, then walks the
    df via ``df_to_packets``. Returns the number of frames pushed.
    """
    init_ops = _reg_start_init_ops(reg_start, reg_widths, freq_mapper)
    if init_ops:
        buffer.push_frame(FramePacket(frame_id=-1, ops=init_ops, total_cycles=0))
    n_frames = 0
    for pkt in df_to_packets(
        df,
        reg_widths,
        freq_mapper,
        irq_cycles=int(irq),
        clock_frequency=clock_frequency,
    ):
        if not buffer.push_frame(pkt):
            break
        n_frames += 1
        if progress_label and n_frames % 200 == 0:
            print(f"{progress_label}: {n_frames} frames", file=sys.stderr)
    return n_frames


def render_to_samples(
    df,
    reg_widths,
    irq,
    cents: int = 50,
    reg_start: Optional[Dict[int, int]] = None,
    chip_model: str = "MOS8580",
    progress_label: str = "",
) -> Tuple[np.ndarray, int]:
    """In-memory offline render: prepared-for-audio df -> (samples_int16, sample_rate). Shares the wav and real-time entries' worker plumbing without the file write or threading-buffer backpressure."""
    from preframr_audio._reg_mappers import FreqMapper  # noqa: WPS433

    if not _HAVE_RESID:
        raise RuntimeError("pyresidfp not installed; pip install pyresidfp")

    freq_mapper = FreqMapper(cents=cents)
    if reg_start is None:
        reg_start = _default_reg_start()
    buffer = AudioRenderBuffer(max_frames=max(1024, 1))
    sink = WavSampleSink()
    worker = ResidWorker(buffer, sink=sink, chip_model=chip_model)
    worker.start()
    try:
        _drive(
            buffer,
            df,
            reg_widths,
            freq_mapper,
            irq,
            reg_start,
            clock_frequency=worker.clock_frequency,
            progress_label=progress_label,
        )
    finally:
        buffer.close()
        worker.join(timeout=60.0)
    return sink.to_array(), int(worker.sampling_frequency)


def render_to_wav(
    df,
    wav_path: str,
    reg_widths,
    irq,
    cents: int = 50,
    reg_start: Optional[Dict[int, int]] = None,
    chip_model: str = "MOS8580",
    descriptions: Optional[List[str]] = None,
    progress: bool = False,
):
    """Offline render: prepared-for-audio df -> wav file. Thin wrapper over ``render_to_samples`` + ``scipy.io.wavfile.write``."""
    from scipy.io import wavfile  # noqa: WPS433

    samples, sample_rate = render_to_samples(
        df,
        reg_widths,
        irq,
        cents=cents,
        reg_start=reg_start,
        chip_model=chip_model,
        progress_label=(
            (descriptions[0] if descriptions else "render") if progress else ""
        ),
    )
    wavfile.write(wav_path, sample_rate, samples)
    return len(samples)


def play_samples(  # pragma: no cover
    df,
    reg_widths,
    irq,
    cents: int = 50,
    reg_start: Optional[Dict[int, int]] = None,
    device: Optional[int] = None,
    chip_model: str = "MOS8580",
    max_frames: int = 200,
    ring_seconds: float = 1.5,
    progress: bool = False,
):
    """Real-time playback: prepared-for-audio df -> sounddevice output."""
    from preframr_audio._reg_mappers import FreqMapper  # noqa: WPS433

    freq_mapper = FreqMapper(cents=cents)
    if reg_start is None:
        reg_start = _default_reg_start()
    drv = ResidAlsaDriver(
        device=device,
        max_frames=max_frames,
        ring_seconds=ring_seconds,
        chip_model=chip_model,
    )
    drv.open()
    try:
        clk = drv.worker.clock_frequency if drv.worker else _PAL_CYCLES_PER_SEC
        _drive(
            drv.buffer,
            df,
            reg_widths,
            freq_mapper,
            irq,
            reg_start,
            clock_frequency=clk,
            progress_label="play" if progress else "",
        )
        while drv.lookahead_frames() > 0:
            time.sleep(0.05)
        if drv.ring is not None and drv.sample_rate:
            sz, _ = drv.ring.occupancy()
            time.sleep(sz / drv.sample_rate + 0.1)
    finally:
        drv.close()
    return drv


def play_via_wav(  # pragma: no cover
    df,
    reg_widths,
    irq,
    cents: int = 50,
    reg_start: Optional[Dict[int, int]] = None,
    chip_model: str = "MOS8580",
    on_frame=None,
    on_frame_fps: float = 50.0,
    audio_device: Optional[str] = None,
    use_paplay: bool = False,
    keep_wav: Optional[str] = None,
):
    """Render the entire df to a temporary wav file, then exec
    ``paplay`` (or ``aplay``) against the file.
    """
    import os  # noqa: WPS433
    import subprocess  # noqa: WPS433
    import tempfile  # noqa: WPS433

    if not _HAVE_RESID:
        raise RuntimeError("pyresidfp not installed")

    if keep_wav:
        wav_path = keep_wav
        cleanup = False
    else:
        fd, wav_path = tempfile.mkstemp(prefix="preframr_play_", suffix=".wav")
        os.close(fd)
        cleanup = True

    try:
        n_samples = render_to_wav(
            df,
            wav_path,
            reg_widths=reg_widths,
            irq=irq,
            cents=cents,
            reg_start=reg_start,
            chip_model=chip_model,
        )
        if use_paplay:
            cmd = ["paplay", wav_path]
        else:
            cmd = ["aplay", "-q", wav_path]
            if audio_device:
                cmd[1:1] = ["-D", audio_device]
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        ticker_stop = threading.Event()
        ticker_thread = None
        if on_frame is not None:
            sample_rate = SoundInterfaceDevice(
                model=getattr(ChipModel, chip_model)
            ).sampling_frequency
            duration = n_samples / sample_rate

            def _tick():
                period = 1.0 / on_frame_fps
                n_frames = int(duration * on_frame_fps)
                next_tick = time.monotonic()
                for idx in range(n_frames):
                    if ticker_stop.is_set():
                        return
                    on_frame(FramePacket(frame_id=idx, ops=[]))
                    next_tick += period
                    sleep_for = next_tick - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            ticker_thread = threading.Thread(target=_tick, daemon=True)
            ticker_thread.start()
        try:
            stderr = proc.stderr.read() if proc.stderr else b""
            proc.wait()
        finally:
            ticker_stop.set()
            if ticker_thread is not None:
                ticker_thread.join(timeout=1.0)
        if proc.returncode not in (0, None):
            msg = stderr.decode(errors="replace").strip() or "(no stderr)"
            raise RuntimeError(f"{cmd[0]} rc={proc.returncode}: {msg}")
    finally:
        if cleanup:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
    return n_samples


def play_via_aplay_prerendered(  # pragma: no cover
    df,
    reg_widths,
    irq,
    cents: int = 50,
    reg_start: Optional[Dict[int, int]] = None,
    chip_model: str = "MOS8580",
    on_frame=None,
    on_frame_fps: float = 50.0,
    audio_device: Optional[str] = None,
    aplay_cmd=None,
):
    """Diagnostic baseline: render the *entire* df to a sample buffer
    in memory, then stream the buffer to aplay/paplay in one shot.
    """
    import subprocess  # noqa: WPS433

    sink = WavSampleSink()
    if not _HAVE_RESID:
        raise RuntimeError("pyresidfp not installed")
    from preframr_audio._reg_mappers import FreqMapper  # noqa: WPS433

    freq_mapper = FreqMapper(cents=cents)
    if reg_start is None:
        reg_start = _default_reg_start()
    buffer = AudioRenderBuffer(max_frames=max(2048, 1))
    worker = ResidWorker(buffer, sink=sink, chip_model=chip_model)
    worker.start()
    try:
        _drive(
            buffer,
            df,
            reg_widths,
            freq_mapper,
            irq,
            reg_start,
            clock_frequency=worker.clock_frequency,
        )
    finally:
        buffer.close()
        worker.join(timeout=60.0)
    samples = sink.to_array()
    sample_rate = worker.sampling_frequency

    if aplay_cmd is None:
        aplay_cmd = [
            "aplay",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(int(sample_rate)),
            "-c",
            "1",
            "--buffer-time=500000",
            "--period-time=100000",
        ]
        if audio_device:
            aplay_cmd += ["-D", audio_device]
        aplay_cmd.append("-")
    proc = subprocess.Popen(aplay_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    ticker_stop = threading.Event()

    def _tick():
        from preframr_audio.audio_driver import FramePacket as _FP  # noqa: WPS433

        period = 1.0 / on_frame_fps
        n_packets = len(samples) / sample_rate * on_frame_fps
        next_tick = time.monotonic()
        idx = 0
        while not ticker_stop.is_set() and idx < n_packets:
            on_frame(_FP(frame_id=idx, ops=[]))
            idx += 1
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    ticker_thread = None
    if on_frame is not None:
        ticker_thread = threading.Thread(target=_tick, daemon=True)
        ticker_thread.start()
    try:
        try:
            proc.stdin.write(samples.tobytes())
        except BrokenPipeError:
            pass
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            stderr = proc.stderr.read() if proc.stderr else b""
            proc.wait(timeout=int(len(samples) / sample_rate) + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stderr = b""
    finally:
        ticker_stop.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=1.0)
    if proc.returncode not in (0, None):
        msg = stderr.decode(errors="replace").strip() or "(no stderr)"
        raise RuntimeError(f"aplay rc={proc.returncode}: {msg}")
    return len(samples)


def play_via_aplay(  # pragma: no cover
    df,
    reg_widths,
    irq,
    cents: int = 50,
    reg_start: Optional[Dict[int, int]] = None,
    chip_model: str = "MOS8580",
    on_frame=None,
    audio_device: Optional[str] = None,
    aplay_cmd=None,
    mute_voices: Optional[set] = None,
    should_quit: Optional[Callable[[], bool]] = None,
):
    """Synchronous live audio playback via the ``aplay`` subprocess."""
    import subprocess  # noqa: WPS433  -- only needed by this function

    if not _HAVE_RESID:
        raise RuntimeError("pyresidfp not installed")
    from preframr_audio._reg_mappers import FreqMapper  # noqa: WPS433
    from preframr_audio._sid_constants import (  # noqa: WPS433
        VOICE_CTRL_REG,
        voice_of_reg,
    )

    freq_mapper = FreqMapper(cents=cents)
    if reg_start is None:
        reg_start = _default_reg_start()

    sid = SoundInterfaceDevice(model=getattr(ChipModel, chip_model))
    sid.reset()
    for _warm_reg in range(25):
        sid.write_register(_warm_reg, 0)
    sid.clock(timedelta(seconds=0.05))
    sample_rate = int(sid.sampling_frequency)
    if aplay_cmd is None:
        aplay_cmd = [
            "aplay",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(sample_rate),
            "-c",
            "1",
            "--buffer-time=500000",
            "--period-time=100000",
        ]
        if audio_device:
            aplay_cmd += ["-D", audio_device]
        aplay_cmd.append("-")
    proc = subprocess.Popen(aplay_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    ring = SampleRing(capacity_samples=sample_rate)
    aplay_died = threading.Event()

    def _writer():
        chunk = np.empty(4096, dtype=np.int16)
        while not aplay_died.is_set():
            n = ring.read(len(chunk), chunk)
            if n == 0:
                if writer_done.is_set() and ring.occupancy()[0] == 0:
                    return
                time.sleep(0.005)
                continue
            try:
                proc.stdin.write(chunk[:n].tobytes())
            except (BrokenPipeError, ValueError):
                aplay_died.set()
                return

    writer_done = threading.Event()
    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    init_ops = _reg_start_init_ops(reg_start, reg_widths, freq_mapper)
    n_samples = 0
    try:
        for op in init_ops:
            if op.reg >= 0:
                sid.write_register(int(op.reg), int(op.val))
        for pkt in df_to_packets(
            df,
            reg_widths,
            freq_mapper,
            irq_cycles=int(irq),
            clock_frequency=sid.clock_frequency,
        ):
            if aplay_died.is_set():
                break
            if should_quit is not None and should_quit():
                break
            muted_now = frozenset(mute_voices) if mute_voices else frozenset()
            for v in muted_now:
                if v in VOICE_CTRL_REG:
                    sid.write_register(VOICE_CTRL_REG[v], 0)
            for op in pkt.ops:
                if op.reg >= 0:
                    op_voice = voice_of_reg(int(op.reg))
                    if op_voice is not None and op_voice in muted_now:
                        pass
                    else:
                        sid.write_register(int(op.reg), int(op.val))
                if op.delay_cycles > 0:
                    seconds = op.delay_cycles / sid.clock_frequency
                    samples = sid.clock(timedelta(seconds=seconds))
                    samples = np.asarray(samples, dtype=np.int16)
                    if len(samples):
                        ring.write(samples)
                        n_samples += len(samples)
            if on_frame is not None:
                on_frame(pkt)
        writer_done.set()
        writer_thread.join(timeout=10.0)
    finally:
        writer_done.set()
        aplay_died.set()
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            stderr = proc.stderr.read() if proc.stderr else b""
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stderr = b""
    if proc.returncode not in (0, None):
        msg = stderr.decode(errors="replace").strip() or "(no stderr)"
        hint = (
            "aplay exited with non-zero status. Common cause: ALSA device "
            "not reachable inside the container. Run ``aplay -l`` to "
            "list playback devices, then pass ``--audio-device "
            "plughw:CARD,DEV`` (e.g. plughw:0,3)."
        )
        raise RuntimeError(f"{hint}\naplay rc={proc.returncode}: {msg}")
    return n_samples


def demo_frame_producer(driver: ResidAlsaDriver, seconds: float):  # pragma: no cover
    """Emit a sweep on voice 0 for the requested duration, exercising
    the driver end-to-end. PAL timing: 50 frames/sec, ~19656 cycles
    per frame; the demo packs each frame as a single op carrying the
    full frame's cycle count after the writes (sufficient for a sweep
    where intra-frame ordering doesn't matter).
    """
    fps = 50
    n_frames = int(seconds * fps)
    setup_regs = [(24, 15), (5, 0x09), (6, 0xA4)]
    base_freq = 0x1500
    for i in range(n_frames):
        ops: List[FrameOp] = []
        if i == 0:
            for reg, val in setup_regs:
                ops.append(FrameOp(reg=reg, val=val, delay_cycles=0))
        f = base_freq + (i * 60)
        ops.append(FrameOp(reg=0, val=f & 0xFF))
        ops.append(FrameOp(reg=1, val=(f >> 8) & 0xFF))
        if i == 0:
            ops.append(FrameOp(reg=4, val=0x21))
        if i == n_frames - 5:
            ops.append(FrameOp(reg=4, val=0x20))
        ops[-1].delay_cycles = 19656
        ok = driver.push_frame(frame_id=i, ops=ops, total_cycles=19656)
        if not ok:
            print(f"push_frame timeout at i={i}", file=sys.stderr)
            break


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice output device id (sd.query_devices())",
    )
    ap.add_argument("--sec", type=float, default=3.0)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        if not _HAVE_SD:
            print("sounddevice not installed", file=sys.stderr)
            return 1
        print(sd.query_devices())
        return 0

    if not _HAVE_RESID:
        print("pyresidfp not installed", file=sys.stderr)
        return 1
    if not _HAVE_SD:
        print("sounddevice not installed", file=sys.stderr)
        return 1

    drv = ResidAlsaDriver(device=args.device)
    drv.open()
    try:
        demo_frame_producer(drv, args.sec)
        while drv.lookahead_frames() > 0:
            time.sleep(0.05)
        sz, _ = drv.ring.occupancy()
        time.sleep(sz / drv.sample_rate + 0.1)
        print("done. status:", drv.status())
    finally:
        drv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
