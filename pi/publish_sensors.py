#!/usr/bin/env python3
"""Publish LIS3DH IMU and VL53L7CX ToF frames to Mosquitto.

Debugging tool for the Raspberry Pi. Both sensors share I2C bus 1
(GPIO2 SDA, GPIO3 SCL). A lock wraps every bus transaction so the IMU
thread and the ToF thread do not interleave a transfer.

Topics (JSON, QoS 0):

    sensors/imu
        {"ts": <unix seconds>, "x": <m/s^2>, "y": <m/s^2>, "z": <m/s^2>, "fake": <bool>}

    sensors/tof
        {"ts": <unix seconds>, "resolution": "8x8"|"4x4",
         "distances": [<mm>, ...], "status": [<int>, ...],
         "silicon_temp_c": <int|null>, "fake": <bool>}

Zone 0 is the sensor's first zone (bottom-left in ST's numbering).
The list is row-major: index = row * width + column, row 0 at the bottom.

If a sensor or its library fails to initialize, that stream falls back
to synthetic data so the MQTT path can still be checked. --fake forces
both streams to synthetic data. --no-fallback exits instead.

The VL53L7CX is driven by the pure-Python VL53L5CX ultra-lite port
(driver-compatible per ST). Firmware upload on init takes a while at
the default 100 kHz I2C clock.
"""

import argparse
import json
import logging
import math
import signal
import subprocess
import sys
import threading
import time

import paho.mqtt.client as mqtt

log = logging.getLogger("publish_sensors")

# Defaults. Override from the command line.
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
IMU_TOPIC = "sensors/imu"
TOF_TOPIC = "sensors/tof"
IMU_HZ = 50.0
TOF_RESOLUTION = 8  # 8 or 4. 8x8 tops out at 15 Hz, 4x4 at 60 Hz.
TOF_HZ = 15
I2C_BUS = 1
IMU_ADDRESS = 0x18

# ST target-status codes that mean the distance is usable.
STATUS_VALID = (5, 9)


def ipv4_addresses():
    """Non-loopback IPv4 addresses, so the PC knows where to subscribe."""
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=3)
        addrs = [part for part in out.split() if part and not part.startswith("127.")]
        if addrs:
            return addrs
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("hostname -I failed: %s", exc)
    return []


def odr_for_rate(publish_hz):
    """Pick an LIS3DH output data rate at least as fast as the publish rate."""
    from lis3dh import DATARATE_100_HZ, DATARATE_200_HZ, DATARATE_400_HZ

    if publish_hz <= 50:
        return DATARATE_100_HZ
    if publish_hz <= 100:
        return DATARATE_200_HZ
    return DATARATE_400_HZ


class FakeIMU:
    """Slow sine waves, in m/s^2, so a plot looks alive without the chip."""

    def __init__(self):
        self._t0 = time.perf_counter()

    def read_accel_mps2(self):
        t = time.perf_counter() - self._t0
        return (
            3.0 * math.sin(2 * math.pi * 0.4 * t),
            1.5 * math.sin(2 * math.pi * 0.9 * t + 1.0),
            9.8 + 0.8 * math.sin(2 * math.pi * 0.2 * t),
        )


class FakeToF:
    """A bright blob moving across the zone grid. Distances are millimetres."""

    def __init__(self, zones_per_side):
        self.side = zones_per_side
        self._t0 = time.perf_counter()

    def read(self):
        t = time.perf_counter() - self._t0
        side = self.side
        cx = (side - 1) / 2 + (side / 3) * math.sin(t)
        cy = (side - 1) / 2 + (side / 3) * math.cos(0.7 * t)
        distances = []
        status = []
        for index in range(side * side):
            row, col = divmod(index, side)
            dist = math.hypot(row - cy, col - cx)
            distances.append(int(300 + 700 * dist))
            status.append(5)
        return distances, status, None


class RealToF:
    """Thin wrapper around the VL53L5CX pure-Python driver.

    The VL53L7CX uses the same ST ultra-lite driver. Extra result blocks
    are turned off so each frame is a short I2C read. Range-sigma is off
    on purpose: this port sizes that block with `is None` instead of the
    flag, which under-allocates the buffer when sigma is enabled.
    """

    def __init__(self, bus_id, zones_per_side, frequency_hz):
        from vl53l5cx.api import VL53L5CX_RESOLUTION_4X4, VL53L5CX_RESOLUTION_8X8
        from vl53l5cx.vl53l5cx import VL53L5CX

        if zones_per_side == 4:
            resolution = VL53L5CX_RESOLUTION_4X4
            frequency_hz = max(1, min(int(frequency_hz), 60))
        elif zones_per_side == 8:
            resolution = VL53L5CX_RESOLUTION_8X8
            frequency_hz = max(1, min(int(frequency_hz), 15))
        else:
            raise ValueError("resolution must be 4 or 8")

        self.side = zones_per_side
        self.frequency_hz = frequency_hz
        self.driver = VL53L5CX(
            bus_id=bus_id,
            nb_target_per_zone=1,
            disable_ambient_per_spad=True,
            disable_nb_spads_enabled=True,
            disable_nb_target_detected=True,
            disable_signal_per_spad=True,
            disable_range_sigma_mm=True,
            disable_reflectance_percent=True,
            disable_motion_indicator=True,
        )

        log.info("VL53L7CX: checking device at 0x29")
        if not self.driver.is_alive():
            raise RuntimeError("VL53L7CX did not answer (device id is not VL53L5/L7)")

        log.info("VL53L7CX: uploading firmware. This often takes 15-60 s at 100 kHz I2C.")
        started = time.perf_counter()
        self.driver.init()
        log.info("VL53L7CX: firmware ready in %.1f s", time.perf_counter() - started)

        self.driver.set_resolution(resolution)
        self.driver.set_ranging_frequency_hz(frequency_hz)
        self.driver.start_ranging()
        log.info(
            "VL53L7CX: ranging %dx%d at %d Hz (read size %d bytes)",
            zones_per_side, zones_per_side, frequency_hz, self.driver.data_read_size,
        )

    def check_and_read(self):
        """Return a frame when one is ready, else None.

        The caller must hold the I2C lock across this whole call.
        """
        if not self.driver.check_data_ready():
            return None
        data = self.driver.get_ranging_data()
        count = self.side * self.side
        distances = [int(round(value)) for value in data.distance_mm[:count]]
        status = [int(value) for value in data.target_status[:count]]
        temp = int(getattr(data, "silicon_temp_degc", 0))
        return distances, status, temp

    def stop(self):
        try:
            self.driver.stop_ranging()
        except Exception:
            log.exception("VL53L7CX stop_ranging failed")


def open_imu(bus_id, address, publish_hz):
    from smbus2 import SMBus

    from lis3dh import LIS3DH, RANGE_4_G

    bus = SMBus(bus_id)
    try:
        imu = LIS3DH(bus, address=address, data_rate=odr_for_rate(publish_hz), g_range=RANGE_4_G)
    except Exception:
        bus.close()
        raise
    return bus, imu


def patch_vl53l5cx_driver():
    """Fix two bugs in Abstract-Horizon's pure-Python VL53L5CX port.

    The port is what we use for the VL53L7CX (ST says the drivers match).
    Stock code does two things wrong:

    * Range-sigma buffer size is chosen with `is None`, so the buffer is
      too small whenever that block is enabled.
    * Disabled result blocks are not zeroed in the output list. The ST C
      driver sets those headers to 0. Leaving them set makes the sensor
      emit bytes the parser does not expect.

    The edit is applied to the installed file before it is imported.
    """
    import importlib.util

    spec = importlib.util.find_spec("vl53l5cx.vl53l5cx")
    if spec is None or not spec.origin:
        return
    path = spec.origin
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    original = text

    sigma_bug = (
        "        if disable_range_sigma_mm is None:\n"
        "            self.L5CX_SIGR_SIZE = ((128 * nb_target_per_zone) + 4)\n"
        "        else:\n"
        "            self.L5CX_SIGR_SIZE = 0\n"
    )
    sigma_fix = (
        "        if disable_range_sigma_mm:\n"
        "            self.L5CX_SIGR_SIZE = 0\n"
        "        else:\n"
        "            self.L5CX_SIGR_SIZE = ((128 * nb_target_per_zone) + 4)\n"
    )
    if sigma_bug in text:
        text = text.replace(sigma_bug, sigma_fix, 1)

    header_line = (
        "            if (output[i] == 0) or "
        "((output_bh_enable[i // DIVIDE_FACTOR] & (1 << (i % 32))) == 0):\n"
    )
    header_fix = header_line + "                output[i] = 0\n"
    if header_line in text and header_fix not in text:
        text = text.replace(header_line, header_fix, 1)

    if text != original:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        log.info("Patched VL53L5CX driver at %s", path)


class Publisher:
    def __init__(self, client):
        self.client = client
        self._lock = threading.Lock()

    def send(self, topic, payload):
        body = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            # paho's publish is thread-safe; the lock keeps the log line
            # next to the message it describes when both threads publish.
            info = self.client.publish(topic, body, qos=0)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("publish %s failed: rc=%s", topic, info.rc)


def imu_loop(sensor, publisher, period, stop, fake):
    samples = 0
    errors = 0
    window_start = time.perf_counter()
    next_tick = window_start
    logged_first = False
    while not stop.is_set():
        next_tick += period
        try:
            x, y, z = sensor.read_accel_mps2()
            publisher.send(IMU_TOPIC, {
                "ts": time.time(),
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "fake": fake,
            })
            samples += 1
            errors = 0
            if not logged_first:
                log.info("IMU first sample: x=%.3f y=%.3f z=%.3f m/s^2 fake=%s", x, y, z, fake)
                logged_first = True
        except Exception:
            errors += 1
            if errors == 1 or errors % 50 == 0:
                log.exception("IMU read failed (%d in a row)", errors)
        now = time.perf_counter()
        if now - window_start >= 5.0:
            log.info("IMU %.1f Hz (consecutive errors %d, fake=%s)", samples / (now - window_start), errors, fake)
            samples = 0
            window_start = now
        delay = next_tick - time.perf_counter()
        if delay > 0:
            stop.wait(delay)
        else:
            # Fell behind (usually the ToF thread holding the bus). Skip
            # missed slots instead of bursting to catch up.
            next_tick = time.perf_counter()


def tof_loop(sensor, publisher, resolution_label, stop, fake, i2c_lock, fake_period):
    samples = 0
    consecutive_errors = 0
    window_start = time.perf_counter()
    logged_first = False
    while not stop.is_set():
        got_frame = False
        try:
            if fake:
                distances, status, temp = sensor.read()
                got_frame = True
            else:
                with i2c_lock:
                    frame = sensor.check_and_read()
                if frame is not None:
                    distances, status, temp = frame
                    got_frame = True
            if got_frame:
                publisher.send(TOF_TOPIC, {
                    "ts": time.time(),
                    "resolution": resolution_label,
                    "distances": distances,
                    "status": status,
                    "silicon_temp_c": temp,
                    "fake": fake,
                })
                samples += 1
                consecutive_errors = 0
                if not logged_first:
                    valid = sum(1 for code in status if code in STATUS_VALID)
                    log.info(
                        "ToF first frame: %s, %d zones, %d valid, temp=%s fake=%s",
                        resolution_label, len(distances), valid, temp, fake,
                    )
                    logged_first = True
        except Exception:
            consecutive_errors += 1
            if consecutive_errors == 1 or consecutive_errors % 20 == 0:
                log.exception("ToF read failed (%d in a row)", consecutive_errors)
            stop.wait(0.2)
        now = time.perf_counter()
        if now - window_start >= 5.0:
            log.info(
                "ToF %.1f Hz (consecutive errors %d, fake=%s)",
                samples / (now - window_start), consecutive_errors, fake,
            )
            samples = 0
            window_start = now
        # Real frames arrive when the sensor says so. Poll slowly enough
        # to leave the bus free for the IMU between checks.
        stop.wait(fake_period if fake else 0.005)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Publish SafeSight sensor frames over MQTT.")
    parser.add_argument("--broker", default=BROKER_HOST, help="Mosquitto host (default %(default)s)")
    parser.add_argument("--port", type=int, default=BROKER_PORT, help="Mosquitto port (default %(default)s)")
    parser.add_argument("--imu-hz", type=float, default=IMU_HZ, help="IMU publish rate (default %(default)s)")
    parser.add_argument("--tof-resolution", type=int, choices=(4, 8), default=TOF_RESOLUTION,
                        help="ToF zones per side (default %(default)s)")
    parser.add_argument("--tof-hz", type=int, default=TOF_HZ,
                        help="ToF ranging frequency. Clamped to 15 Hz at 8x8 and 60 Hz at 4x4.")
    parser.add_argument("--fake", action="store_true", help="Publish synthetic data for both sensors.")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Exit if a real sensor cannot be opened. Default is to fall back to fake data.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Stop after this many seconds. 0 means run until Ctrl+C.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def connect_mqtt(host, port, stop):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="safesight-publisher")
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            log.info("MQTT connected to %s:%s", host, port)
        else:
            log.error("MQTT connection refused: %s", reason_code)

    def on_disconnect(client, userdata, flags, reason_code, properties):
        if not stop.is_set():
            log.warning("MQTT disconnected (%s). paho will retry.", reason_code)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    while not stop.is_set():
        try:
            client.connect(host, port, keepalive=30)
            client.loop_start()
            return client
        except Exception as exc:
            log.error("MQTT connect to %s:%s failed: %s", host, port, exc)
            stop.wait(2.0)
    return None


def main(argv):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.imu_hz <= 0:
        log.error("--imu-hz must be positive")
        return 2

    stop = threading.Event()

    def _handle_signal(signum, _frame):
        log.info("signal %s, stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    addresses = ipv4_addresses()
    log.info("Pi IPv4: %s", ", ".join(addresses) if addresses else "(none found)")
    log.info("Publishing %s and %s to %s:%s", IMU_TOPIC, TOF_TOPIC, args.broker, args.port)

    i2c_lock = threading.Lock()
    imu_bus = None
    tof = None
    imu_fake = args.fake
    tof_fake = args.fake

    tof_cap = 15 if args.tof_resolution == 8 else 60
    tof_hz = max(1, min(int(args.tof_hz), tof_cap))

    if args.fake:
        imu = FakeIMU()
        tof_sensor = FakeToF(args.tof_resolution)
        log.warning("FAKE mode: both sensors are synthetic")
    else:
        try:
            with i2c_lock:
                imu_bus, imu = open_imu(I2C_BUS, IMU_ADDRESS, args.imu_hz)
            log.info("LIS3DH ready at 0x%02x, publishing at %.1f Hz", IMU_ADDRESS, args.imu_hz)
        except Exception:
            log.exception("LIS3DH init failed")
            if args.no_fallback:
                return 1
            log.warning("IMU falling back to fake data")
            imu = FakeIMU()
            imu_fake = True
            imu_bus = None

        # Real reads go through the lock. The fake source does not touch I2C,
        # so wrap only the real driver.
        if not imu_fake:
            real_imu = imu

            class _LockedIMU:
                def read_accel_mps2(_self):
                    with i2c_lock:
                        return real_imu.read_accel_mps2()

            imu = _LockedIMU()

        try:
            patch_vl53l5cx_driver()
            # init() uploads ~84 KB of firmware. Hold the bus the whole time.
            with i2c_lock:
                tof = RealToF(I2C_BUS, args.tof_resolution, tof_hz)
            tof_sensor = tof
        except Exception:
            log.exception("VL53L7CX init failed")
            if args.no_fallback:
                if imu_bus is not None:
                    imu_bus.close()
                return 1
            log.warning("ToF falling back to fake data")
            tof_sensor = FakeToF(args.tof_resolution)
            tof_fake = True
            tof = None

    client = connect_mqtt(args.broker, args.port, stop)
    if client is None:
        if tof is not None and not tof_fake:
            with i2c_lock:
                tof.stop()
        if imu_bus is not None:
            imu_bus.close()
        return 1
    publisher = Publisher(client)

    resolution_label = f"{args.tof_resolution}x{args.tof_resolution}"
    threads = [
        threading.Thread(
            target=imu_loop,
            name="imu",
            args=(imu, publisher, 1.0 / args.imu_hz, stop, imu_fake),
            daemon=True,
        ),
        threading.Thread(
            target=tof_loop,
            name="tof",
            args=(tof_sensor, publisher, resolution_label, stop, tof_fake, i2c_lock, 1.0 / tof_hz),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    started = time.perf_counter()
    try:
        while not stop.is_set():
            stop.wait(0.5)
            if args.duration and (time.perf_counter() - started) >= args.duration:
                log.info("duration %.1f s elapsed, stopping", args.duration)
                stop.set()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        if tof is not None and not tof_fake:
            with i2c_lock:
                tof.stop()
        if imu_bus is not None:
            try:
                imu_bus.close()
            except Exception:
                log.exception("closing I2C bus")
        client.loop_stop()
        client.disconnect()
        log.info("stopped. Pi IPv4: %s", ", ".join(ipv4_addresses()) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
