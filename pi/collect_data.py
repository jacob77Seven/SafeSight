#!/usr/bin/env python3
"""Log LIS3DH and VL53L7CX frames for offline gait training.

Headless and meant for a battery pack. The hot path is the same sensor
read as ``publish_sensors.py``. Records go to a local JSONL file. MQTT
is off unless ``--mqtt`` is passed, so a session does not depend on the
broker or the USB link.

Label events from a second SSH session while this one is running:

    echo walking >> data/<session>.ctl
    echo still >> data/<session>.ctl
    echo obstacle >> data/<session>.ctl
    echo step-down >> data/<session>.ctl
    echo arm-wave >> data/<session>.ctl

Any other text is stored as-is. If stdin is a terminal, a line typed
here is stored the same way. Recommended labels for the gait notebook
are ``walking`` and ``still``. The others are kept so a later obstacle
model can use the same files.

``mono`` in each record is the clock to train on. ``ts`` is the wall
clock and may be wrong until NTP exists.

Stop ``publish_sensors.py`` before a real session. It holds I²C, and
the two processes cannot share the bus. ``--fake`` skips the sensors
and only checks the file format.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time

from mqtt_debug import MqttDebug
from sensor_session import SensorSession
from session_io import JsonlWriter, default_out_path

log = logging.getLogger("collect_data")

LABELS = ("walking", "still", "obstacle", "step-down", "arm-wave")


def _control_path(out_path, explicit):
    if explicit:
        return explicit
    root, _ext = os.path.splitext(out_path)
    return root + ".ctl"


class LabelSource:
    """Labels from a control file and, when present, the terminal."""

    def __init__(self, control_path, stop):
        self.path = control_path
        self.stop = stop
        self._pending = []
        self._lock = threading.Lock()
        parent = os.path.dirname(control_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Empty the file so a previous session's labels are not replayed
        # and are not left sitting in the file the operator tails.
        with open(control_path, "w", encoding="utf-8"):
            pass
        self._fh = open(control_path, "r", encoding="utf-8")
        self._thread = None
        if sys.stdin is not None and sys.stdin.isatty():
            self._thread = threading.Thread(target=self._stdin_loop, name="labels", daemon=True)
            self._thread.start()
            log.info("type a label and press enter. Also appending to %s", control_path)
        else:
            log.info("headless. Label with: echo walking >> %s", control_path)

    def _stdin_loop(self):
        while not self.stop.is_set():
            line = sys.stdin.readline()
            if not line:
                return
            self._add(line)

    def _add(self, line):
        label = line.strip()
        if not label or label.startswith("#"):
            return
        with self._lock:
            self._pending.append(label)

    def poll(self):
        # Bytes appended after we opened the file. A handle that has
        # already seen EOF does not notice a later append until it seeks
        # (Windows, and some buffered readers). The second SSH session
        # only ever appends, so seeking to the current offset is enough.
        pos = self._fh.tell()
        self._fh.seek(pos)
        while True:
            line = self._fh.readline()
            if not line:
                break
            self._add(line)
        with self._lock:
            rows = self._pending
            self._pending = []
        return rows

    def close(self):
        try:
            self._fh.close()
        except OSError:
            pass


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Record SafeSight IMU and ToF frames to a JSONL file.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. 0 runs until Ctrl+C.")
    parser.add_argument("--out", default="", help="JSONL path. Default is data/session_<time>.jsonl")
    parser.add_argument("--imu-hz", type=float, default=50.0)
    parser.add_argument("--tof-hz", type=int, default=15)
    parser.add_argument("--mqtt", action="store_true", help="Also publish sensors/imu and sensors/tof.")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--fake", action="store_true", help="Synthetic sensors, for a dry run.")
    parser.add_argument("--no-fallback", action="store_true", help="Exit if a real sensor fails to open.")
    parser.add_argument("--rotate-s", type=float, default=0.0, help="Open a new file after this many seconds. 0 keeps one file.")
    parser.add_argument("--control", default="", help="Label file. Default is the output path with a .ctl suffix.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    out_path = args.out or default_out_path("session")
    control_path = _control_path(out_path, args.control)
    stop = threading.Event()

    def _handle(signum, _frame):
        log.info("signal %s, stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    session = SensorSession(
        imu_hz=args.imu_hz,
        tof_hz=args.tof_hz,
        fake=args.fake,
        no_fallback=args.no_fallback,
        side=8,
    )
    writer = None
    labels = None
    mqtt = None
    try:
        writer = JsonlWriter(out_path, rotate_s=args.rotate_s)
        labels = LabelSource(control_path, stop)
        session.start()
        if args.mqtt:
            mqtt = MqttDebug(args.broker, args.port, "safesight-collect")
            mqtt.start()
        writer.write({
            "rec": "meta",
            "ts": time.time(),
            "mono": time.monotonic(),
            "imu_hz": args.imu_hz,
            "tof_hz": args.tof_hz,
            "resolution": "8x8",
            "labels": list(LABELS),
            "note": "SafeSight collect. Use mono for intervals.",
        }, force_flush=True)
        log.info("writing %s", writer.path)
        log.info("known labels: %s", ", ".join(LABELS))
        started = time.monotonic()
        last_report = started
        report_imu = 0
        report_tof = 0
        while not stop.is_set():
            for sample in session.drain_imu():
                mono, ts, x, y, z, fake = sample
                writer.write({
                    "rec": "imu",
                    "mono": round(mono, 4),
                    "ts": round(ts, 3),
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": round(z, 4),
                    "fake": bool(fake),
                })
                if mqtt is not None:
                    mqtt.publish_imu(sample)
                report_imu += 1
            for frame in session.drain_tof():
                mono, ts, distances, status, temp, fake, _read_ms = frame
                writer.write({
                    "rec": "tof",
                    "mono": round(mono, 4),
                    "ts": round(ts, 3),
                    "resolution": "8x8",
                    "distances": [int(round(v)) for v in distances],
                    "status": [int(v) for v in status],
                    "silicon_temp_c": temp,
                    "fake": bool(fake),
                })
                if mqtt is not None:
                    mqtt.publish_tof(frame, "8x8")
                report_tof += 1
            for label in labels.poll():
                writer.write({
                    "rec": "event",
                    "mono": round(time.monotonic(), 4),
                    "ts": round(time.time(), 3),
                    "label": label,
                }, force_flush=True)
                log.info("event %s", label)
            now = time.monotonic()
            if now - last_report >= 5.0:
                span = now - last_report
                log.info(
                    "collect %.0fs imu %.1f Hz tof %.1f Hz events-file=%s bytes=%d",
                    now - started,
                    report_imu / span,
                    report_tof / span,
                    control_path,
                    writer.bytes,
                )
                report_imu = 0
                report_tof = 0
                last_report = now
            if args.duration and (now - started) >= args.duration:
                log.info("duration %.1f s elapsed", args.duration)
                stop.set()
                break
            stop.wait(0.02)
    finally:
        stop.set()
        session.close()
        if labels is not None:
            labels.close()
        if writer is not None:
            writer.close()
            log.info("closed %s", writer.path)
        if mqtt is not None:
            mqtt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
