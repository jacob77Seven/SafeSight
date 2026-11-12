"""Short spatialized warning tones for the PCM5102.

The DAC is brought up by ``dtoverlay=hifiberry-dac`` and shows up as the
default ALSA playback device. A tone is a few tens of milliseconds, so it
does not need a mixer graph.

Stereo pan follows the cluster azimuth. Negative azimuth is to the
wearer's left (column 0 of the ToF image). Equal-power pan keeps the
loudness steady as the obstacle moves across the cone: full left at
−30°, full right at +30°, both channels at the centre.

Pitch and level carry urgency. A distant 2 s TTC is a low, quiet beep.
A sub-second TTC is higher and louder. A step-down is a lower tone than
any of those so it is not confused with a closing obstacle.

Playback is non-blocking. ``sounddevice`` is used when it imports.
Otherwise the same PCM is handed to ``aplay`` on stdin, which is already
on Raspberry Pi OS and does not need a pip install or internet. If both
fail, warnings are still logged and published; the loop does not stop.
"""

import logging
import math
import shutil
import struct
import subprocess
import threading

log = logging.getLogger("audio_warn")

SAMPLE_RATE = 22050
# Half-angle of the VL53L7CX square field. Azimuth outside this saturates
# the pan instead of wrapping around the head.
PAN_HALF_DEG = 30.0


def _envelope(i, n, ramp):
    if i < ramp:
        return i / float(ramp)
    if i > n - ramp:
        return max(0.0, (n - i) / float(ramp))
    return 1.0


def render_tone(azimuth_deg, ttc_s, step_down, sample_rate=SAMPLE_RATE, duration_s=0.12):
    """Return ``(pcm_bytes, meta)``. PCM is S16_LE stereo, interleaved."""
    if step_down:
        freq = 280.0
        amplitude = 0.45
    elif ttc_s is None:
        freq = 700.0
        amplitude = 0.30
    elif ttc_s < 0.6:
        freq = 1400.0
        amplitude = 0.55
    elif ttc_s < 1.2:
        freq = 950.0
        amplitude = 0.40
    else:
        freq = 620.0
        amplitude = 0.25
    pan = azimuth_deg / PAN_HALF_DEG
    if pan < -1.0:
        pan = -1.0
    elif pan > 1.0:
        pan = 1.0
    # angle 0 is full left, pi/2 is full right.
    angle = (pan + 1.0) * (math.pi / 4.0)
    left_gain = math.cos(angle) * amplitude
    right_gain = math.sin(angle) * amplitude
    n = max(1, int(sample_rate * duration_s))
    ramp = max(1, int(sample_rate * 0.01))
    frames = []
    for i in range(n):
        env = _envelope(i, n, ramp)
        sample = math.sin(2.0 * math.pi * freq * (i / float(sample_rate)))
        left = int(max(-32767, min(32767, sample * env * left_gain * 32767.0)))
        right = int(max(-32767, min(32767, sample * env * right_gain * 32767.0)))
        frames.append(struct.pack("<hh", left, right))
    meta = {
        "freq_hz": freq,
        "pan": round(pan, 3),
        "amplitude": amplitude,
        "azimuth_deg": azimuth_deg,
        "ttc_s": ttc_s,
        "step_down": bool(step_down),
        "sample_rate": sample_rate,
    }
    return b"".join(frames), meta


class TonePlayer:
    """Play at most one tone at a time.

    A tone that arrives while one is already playing is dropped. Waiting
    for the DAC, or killing the current beep to start another, would stall
    the sensor loop. The runtime cooldown spaces warnings out so a drop
    here is rare.
    """

    def __init__(self, device=None):
        self.device = device
        self._lock = threading.Lock()
        self._busy = False
        self._proc = None
        self._backend = None
        self._sd = None
        self._select_backend()

    def _select_backend(self):
        try:
            import sounddevice as sd
            self._sd = sd
            self._backend = "sounddevice"
            log.info("audio backend: sounddevice device=%s", self.device or "default")
            return
        except Exception as exc:
            log.info("sounddevice not available (%s)", exc)
        if shutil.which("aplay"):
            self._backend = "aplay"
            log.info("audio backend: aplay device=%s", self.device or "default")
            return
        self._backend = None
        log.warning("no audio backend. Warnings will be logged only.")

    def close(self):
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _finished(self):
        if self._backend == "sounddevice":
            try:
                return not self._sd.get_stream().active
            except Exception:
                return True
        if self._proc is None:
            return True
        return self._proc.poll() is not None

    def play(self, pcm, meta):
        """Start ``pcm`` if the DAC is free. Returns True when playback started."""
        if self._backend is None:
            return False
        with self._lock:
            if self._busy and not self._finished():
                return False
            self._busy = True
            try:
                self._start(pcm)
            except Exception:
                self._busy = False
                log.exception("audio playback failed")
                return False
        log.info(
            "tone %s %.0f Hz pan %.2f ttc=%s",
            "step-down" if meta.get("step_down") else "obstacle",
            meta["freq_hz"],
            meta["pan"],
            meta.get("ttc_s"),
        )
        return True

    def _start(self, pcm):
        if self._backend == "sounddevice":
            # sounddevice wants float frames in −1..1, shape (n, 2).
            n = len(pcm) // 4
            frames = []
            for i in range(n):
                left, right = struct.unpack_from("<hh", pcm, i * 4)
                frames.append([left / 32767.0, right / 32767.0])
            self._sd.play(frames, SAMPLE_RATE, device=self.device, blocking=False)
            return
        if self._backend == "aplay":
            cmd = ["aplay", "-q", "-f", "S16_LE", "-c", "2", "-r", str(SAMPLE_RATE)]
            if self.device:
                cmd.extend(["-D", self.device])
            cmd.append("-")
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # The pipe buffer is usually larger than one beep (~10 KB), but
            # a full buffer would block this thread for the whole tone.
            # Hand the bytes to a side thread so the pipeline keeps polling.
            proc = self._proc

            def _feed():
                try:
                    proc.stdin.write(pcm)
                    proc.stdin.close()
                except BrokenPipeError:
                    log.warning("aplay closed the pipe")
                except Exception:
                    log.exception("aplay write failed")

            threading.Thread(target=_feed, name="aplay-write", daemon=True).start()
            return
        raise RuntimeError("no audio backend")


def cooldown_s(ttc_s, step_down):
    """Minimum gap between tones. Urgent contacts may repeat sooner."""
    if step_down:
        return 0.45
    if ttc_s is not None and ttc_s < 0.8:
        return 0.18
    return 0.40
