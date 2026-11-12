"""Shared LIS3DH + VL53L7CX acquisition for collect and runtime.

This is the same bring-up as ``publish_sensors.py`` (including the
VL53L5CX driver patch and the fake-data fallback) without the MQTT
loop. Both sensors share I²C bus 1, so a lock wraps every transfer.

IMU samples are queued, not overwritten. The gait window needs every
sample. ToF frames are queued with a short cap: if the pipeline falls
behind, older frames are dropped so the warning stays about the scene
in front of the wearer now, not half a second ago.

Timestamps:

* ``mono`` is ``time.monotonic``, taken just before the read. Use it for
  rates, windows, and TTC. It does not jump when the wall clock is set.
* ``ts`` is ``time.time``. The Pi has no NTP, so this is only a label.
"""

import logging
import threading
import time

from publish_sensors import (
    IMU_ADDRESS,
    I2C_BUS,
    FakeIMU,
    FakeToF,
    RealToF,
    open_imu,
    patch_vl53l5cx_driver,
)

log = logging.getLogger("sensor_session")


class SensorSession:
    def __init__(self, imu_hz=50.0, tof_hz=15, fake=False, no_fallback=False, side=8):
        if imu_hz <= 0:
            raise ValueError("imu_hz must be positive")
        self.imu_hz = float(imu_hz)
        self.tof_hz = int(tof_hz)
        self.fake = bool(fake)
        self.no_fallback = bool(no_fallback)
        self.side = int(side)
        self.stop = threading.Event()
        self.i2c_lock = threading.Lock()
        self.imu_fake = bool(fake)
        self.tof_fake = bool(fake)
        self._imu_bus = None
        self._tof = None
        self._lock = threading.Lock()
        self._imu_q = []
        self._tof_q = []
        self.imu_count = 0
        self.tof_count = 0
        self.imu_errors = 0
        self.tof_errors = 0
        self._threads = []

    def start(self):
        imu_sensor, tof_sensor = self._open()
        self._threads = [
            threading.Thread(
                target=self._imu_loop,
                name="imu",
                args=(imu_sensor,),
                daemon=True,
            ),
            threading.Thread(
                target=self._tof_loop,
                name="tof",
                args=(tof_sensor,),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        log.info(
            "sensors started imu %.1f Hz (%s) tof %d Hz %dx%d (%s)",
            self.imu_hz,
            "fake" if self.imu_fake else "LIS3DH",
            self.tof_hz,
            self.side,
            self.side,
            "fake" if self.tof_fake else "VL53L7CX",
        )

    def _open(self):
        if self.fake:
            log.warning("FAKE mode: both sensors are synthetic")
            return FakeIMU(), FakeToF(self.side)
        imu_sensor = None
        try:
            with self.i2c_lock:
                self._imu_bus, imu_sensor = open_imu(I2C_BUS, IMU_ADDRESS, self.imu_hz)
            log.info("LIS3DH ready at 0x%02x", IMU_ADDRESS)
        except Exception:
            log.exception("LIS3DH init failed")
            if self.no_fallback:
                raise
            log.warning("IMU falling back to fake data")
            imu_sensor = FakeIMU()
            self.imu_fake = True
            self._imu_bus = None
        if not self.imu_fake:
            real_imu = imu_sensor

            class _Locked:
                def read_accel_mps2(_self):
                    with self.i2c_lock:
                        return real_imu.read_accel_mps2()

            imu_sensor = _Locked()
        try:
            patch_vl53l5cx_driver()
            with self.i2c_lock:
                self._tof = RealToF(I2C_BUS, self.side, self.tof_hz)
            return imu_sensor, self._tof
        except Exception:
            log.exception("VL53L7CX init failed")
            if self.no_fallback:
                self.close()
                raise
            log.warning("ToF falling back to fake data")
            self._tof = None
            self.tof_fake = True
            return imu_sensor, FakeToF(self.side)

    def close(self):
        self.stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._tof is not None and not self.tof_fake:
            try:
                with self.i2c_lock:
                    self._tof.stop()
            except Exception:
                log.exception("stopping ToF")
        if self._imu_bus is not None:
            try:
                self._imu_bus.close()
            except Exception:
                log.exception("closing I2C")
            self._imu_bus = None

    def drain_imu(self):
        with self._lock:
            rows = self._imu_q
            self._imu_q = []
        return rows

    def drain_tof(self):
        with self._lock:
            rows = self._tof_q
            self._tof_q = []
        return rows

    def _push_imu(self, sample):
        with self._lock:
            self._imu_q.append(sample)
            if len(self._imu_q) > 500:
                del self._imu_q[: len(self._imu_q) - 500]
            self.imu_count += 1

    def _push_tof(self, sample):
        with self._lock:
            self._tof_q.append(sample)
            # A few frames of slack, then drop the oldest. Latency matters
            # more than a perfect history here; the pipeline keeps its own
            # 0.2 s baseline.
            if len(self._tof_q) > 4:
                del self._tof_q[: len(self._tof_q) - 4]
            self.tof_count += 1

    def _imu_loop(self, sensor):
        period = 1.0 / self.imu_hz
        next_tick = time.perf_counter()
        while not self.stop.is_set():
            next_tick += period
            mono = time.monotonic()
            try:
                x, y, z = sensor.read_accel_mps2()
                self._push_imu((mono, time.time(), float(x), float(y), float(z), self.imu_fake))
                self.imu_errors = 0
            except Exception:
                self.imu_errors += 1
                if self.imu_errors == 1 or self.imu_errors % 50 == 0:
                    log.exception("IMU read failed (%d in a row)", self.imu_errors)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self.stop.wait(delay)
            else:
                next_tick = time.perf_counter()

    def _tof_loop(self, sensor):
        period = 1.0 / max(self.tof_hz, 1)
        while not self.stop.is_set():
            try:
                mono = time.monotonic()
                if self.tof_fake:
                    distances, status, temp = sensor.read()
                    read_ms = (time.monotonic() - mono) * 1000.0
                    self._push_tof((
                        mono, time.time(), list(distances), list(status), temp,
                        True, read_ms,
                    ))
                    self.tof_errors = 0
                    self.stop.wait(period)
                    continue
                with self.i2c_lock:
                    frame = sensor.check_and_read()
                read_ms = (time.monotonic() - mono) * 1000.0
                if frame is None:
                    self.stop.wait(0.005)
                    continue
                distances, status, temp = frame
                self._push_tof((
                    mono, time.time(), distances, status, temp, False, read_ms,
                ))
                self.tof_errors = 0
            except Exception:
                self.tof_errors += 1
                if self.tof_errors == 1 or self.tof_errors % 20 == 0:
                    log.exception("ToF read failed (%d in a row)", self.tof_errors)
                self.stop.wait(0.2)
