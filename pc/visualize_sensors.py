#!/usr/bin/env python3
"""Live IMU plot and VL53L7CX zone view for SafeSight debugging.

Subscribes to the Pi publisher:

    sensors/imu   {"ts", "x", "y", "z", "fake"}
    sensors/tof   {"ts", "resolution", "distances", "status", "silicon_temp_c", "fake"}

The broker address is a constant below and a --broker argument.
Times on the plot are seconds before the newest sample, measured when
the message arrived on this PC. The status line also shows the timestamp
from the payload, which uses the Pi's clock.

ToF zone 0 is drawn at the bottom-left, matching the publisher (ST's
zone order: left to right, then bottom to top).
"""

import argparse
import json
import math
import sys
import threading
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# USB-Ethernet address of the Pi when this was set up. Pass --broker if it changes.
DEFAULT_BROKER = "169.254.239.255"
DEFAULT_PORT = 1883
IMU_TOPIC = "sensors/imu"
TOF_TOPIC = "sensors/tof"

# How much IMU history to keep on the scrolling plot.
IMU_WINDOW_S = 10.0
# Fixed colour scale for the zone map, in millimetres. The sensor's useful
# range is about 0-3500 mm. Keeping the scale fixed makes colours comparable
# from frame to frame.
TOF_LEVELS_MM = (0, 3500)
STATUS_VALID = (5, 9)


class StreamState:
    """Written by the paho thread, read by the Qt timer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.broker = ""
        self.last_error = ""

        self.imu_t = deque()
        self.imu_x = deque()
        self.imu_y = deque()
        self.imu_z = deque()
        self.imu_arrivals = deque()
        self.imu_ts = None
        self.imu_fake = False
        self.imu_rate = 0.0

        self.tof = None  # latest decoded dict
        self.tof_arrivals = deque()
        self.tof_received_at = None
        self.tof_rate = 0.0

    def _rate(self, arrivals, now):
        while arrivals and now - arrivals[0] > 2.0:
            arrivals.popleft()
        if len(arrivals) < 2:
            return 0.0
        span = arrivals[-1] - arrivals[0]
        if span <= 0:
            return 0.0
        return (len(arrivals) - 1) / span

    def add_imu(self, payload):
        now = time.perf_counter()
        with self.lock:
            self.imu_t.append(now)
            self.imu_x.append(float(payload["x"]))
            self.imu_y.append(float(payload["y"]))
            self.imu_z.append(float(payload["z"]))
            cutoff = now - IMU_WINDOW_S
            while self.imu_t and self.imu_t[0] < cutoff:
                self.imu_t.popleft()
                self.imu_x.popleft()
                self.imu_y.popleft()
                self.imu_z.popleft()
            self.imu_arrivals.append(now)
            self.imu_rate = self._rate(self.imu_arrivals, now)
            self.imu_ts = float(payload.get("ts", 0.0))
            self.imu_fake = bool(payload.get("fake", False))

    def add_tof(self, payload):
        now = time.perf_counter()
        distances = [float(value) for value in payload["distances"]]
        status = [int(value) for value in payload.get("status", [])]
        with self.lock:
            self.tof = {
                "distances": distances,
                "status": status,
                "resolution": str(payload.get("resolution", "")),
                "ts": float(payload.get("ts", 0.0)),
                "temp": payload.get("silicon_temp_c"),
                "fake": bool(payload.get("fake", False)),
            }
            self.tof_received_at = now
            self.tof_arrivals.append(now)
            self.tof_rate = self._rate(self.tof_arrivals, now)

    def snapshot(self):
        with self.lock:
            tof = None if self.tof is None else dict(self.tof)
            return {
                "connected": self.connected,
                "broker": self.broker,
                "error": self.last_error,
                "imu_t": np.asarray(self.imu_t, dtype=float),
                "imu_x": np.asarray(self.imu_x, dtype=float),
                "imu_y": np.asarray(self.imu_y, dtype=float),
                "imu_z": np.asarray(self.imu_z, dtype=float),
                "imu_ts": self.imu_ts,
                "imu_fake": self.imu_fake,
                "imu_rate": self.imu_rate,
                "imu_age": None if not self.imu_arrivals else time.perf_counter() - self.imu_arrivals[-1],
                "tof": tof,
                "tof_rate": self.tof_rate,
                "tof_age": None if self.tof_received_at is None else time.perf_counter() - self.tof_received_at,
            }


def _age(seconds):
    if seconds is None:
        return "no data"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms ago"
    return f"{seconds:.1f} s ago"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, state, broker, port):
        super().__init__()
        self.state = state
        self.broker = broker
        self.port = port
        self.setWindowTitle("SafeSight sensors")
        self.resize(1100, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.status = QtWidgets.QLabel("connecting...")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-family: Consolas, 'Courier New', monospace; padding: 4px;")
        layout.addWidget(self.status)

        self.plot = pg.PlotWidget(title="LIS3DH acceleration")
        self.plot.addLegend(offset=(8, 8))
        self.plot.setLabel("left", "acceleration", units="m/s^2")
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve_x = self.plot.plot(pen=pg.mkPen("#e74c3c", width=2), name="x")
        self.curve_y = self.plot.plot(pen=pg.mkPen("#2ecc71", width=2), name="y")
        self.curve_z = self.plot.plot(pen=pg.mkPen("#3498db", width=2), name="z")
        layout.addWidget(self.plot, stretch=3)

        tof_row = QtWidgets.QHBoxLayout()
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # ImageView builds a histogram the first time it sees an image.
        # Seed it so the colour scale is already in millimetres.
        self.image_view.setImage(np.zeros((8, 8)), autoLevels=False, levels=TOF_LEVELS_MM)
        tof_row.addWidget(self.image_view, stretch=3)

        self.grid = QtWidgets.QLabel("waiting for ToF")
        self.grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.grid.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 13px; padding: 8px;"
        )
        self.grid.setMinimumWidth(280)
        tof_row.addWidget(self.grid, stretch=2)
        layout.addLayout(tof_row, stretch=2)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    def refresh(self):
        snap = self.state.snapshot()
        self._update_status(snap)
        self._update_imu(snap)
        self._update_tof(snap)

    def _update_status(self, snap):
        link = "connected" if snap["connected"] else "disconnected"
        imu_flag = " FAKE" if snap["imu_fake"] else ""
        tof = snap["tof"]
        if tof is None:
            tof_text = "ToF: no data"
        else:
            fake = " FAKE" if tof["fake"] else ""
            temp = "" if tof["temp"] is None else f" temp={tof['temp']}C"
            tof_text = (
                f"ToF {tof['resolution']}{fake} {snap['tof_rate']:.1f} Hz "
                f"ts={tof['ts']:.3f} ({_age(snap['tof_age'])}){temp}"
            )
        imu_ts = "none" if snap["imu_ts"] is None else f"{snap['imu_ts']:.3f}"
        error = f"  error: {snap['error']}" if snap["error"] else ""
        self.status.setText(
            f"MQTT {link} {self.broker}:{self.port}{error}\n"
            f"IMU{imu_flag} {snap['imu_rate']:.1f} Hz ts={imu_ts} ({_age(snap['imu_age'])})    {tof_text}"
        )

    def _update_imu(self, snap):
        t = snap["imu_t"]
        if t.size == 0:
            return
        # 0 is "now", negative is older, so the trace scrolls left.
        t = t - t[-1]
        self.curve_x.setData(t, snap["imu_x"])
        self.curve_y.setData(t, snap["imu_y"])
        self.curve_z.setData(t, snap["imu_z"])

    def _update_tof(self, snap):
        tof = snap["tof"]
        if tof is None or not tof["distances"]:
            return
        count = len(tof["distances"])
        side = int(round(math.sqrt(count)))
        if side * side != count:
            self.grid.setText(f"unexpected zone count {count}")
            return
        distances = np.asarray(tof["distances"], dtype=float).reshape(side, side)
        status = np.zeros(count, dtype=int)
        raw_status = tof["status"]
        if raw_status:
            status[:min(count, len(raw_status))] = raw_status[:count]
        status = status.reshape(side, side)

        # ImageItem's origin is the bottom-left, and row 0 of this array is
        # zone row 0, so zone 0 lands on the bottom-left of the heatmap.
        self.image_view.setImage(
            distances,
            autoRange=False,
            autoLevels=False,
            levels=TOF_LEVELS_MM,
        )

        # Text grid is printed with the top row first, so it matches looking
        # at the heatmap. Invalid ranges (status not 5 or 9) are parenthesised.
        lines = [f"VL53L7CX {tof['resolution']}  mm", "top of grid = far row", ""]
        for row in range(side - 1, -1, -1):
            cells = []
            for col in range(side):
                value = int(round(distances[row, col]))
                if int(status[row, col]) in STATUS_VALID:
                    cells.append(f"{value:4d}")
                else:
                    cells.append(f"({value:4d})")
            lines.append(" ".join(cells))
        valid = int(np.isin(status, STATUS_VALID).sum())
        lines.append("")
        lines.append(f"valid {valid}/{count}   (paren = status not 5 or 9)")
        lines.append("zone 0 is bottom-left")
        self.grid.setText("\n".join(lines))

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


def start_mqtt(state, broker, port):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="safesight-visualizer")
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    state.broker = f"{broker}:{port}"

    def on_connect(client, userdata, flags, reason_code, properties):
        with state.lock:
            state.connected = reason_code == 0
            state.last_error = "" if reason_code == 0 else f"connect refused: {reason_code}"
        if reason_code == 0:
            client.subscribe(IMU_TOPIC)
            client.subscribe(TOF_TOPIC)

    def on_disconnect(client, userdata, flags, reason_code, properties):
        with state.lock:
            state.connected = False
            if reason_code != 0:
                state.last_error = f"disconnected: {reason_code}"

    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if message.topic == IMU_TOPIC:
                state.add_imu(payload)
            elif message.topic == TOF_TOPIC:
                state.add_tof(payload)
        except Exception as exc:
            with state.lock:
                state.last_error = f"{message.topic}: {exc}"

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect_async(broker, port, keepalive=30)
    client.loop_start()
    return client


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Plot SafeSight MQTT sensor streams.")
    parser.add_argument("--broker", default=DEFAULT_BROKER,
                        help=f"Pi Mosquitto host (default {DEFAULT_BROKER})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Mosquitto port")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    app = QtWidgets.QApplication(sys.argv)
    state = StreamState()
    client = start_mqtt(state, args.broker, args.port)
    window = MainWindow(state, args.broker, args.port)

    def _shutdown():
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass

    app.aboutToQuit.connect(_shutdown)
    window.show()
    exec_fn = app.exec if hasattr(app, "exec") else app.exec_
    return exec_fn()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
