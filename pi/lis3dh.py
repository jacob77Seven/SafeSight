"""Small LIS3DH accelerometer driver for the Raspberry Pi I2C bus.

The register sequence and the m/s^2 scaling match Adafruit's
CircuitPython driver (high-resolution mode). This stays on smbus2 so it
can share the bus with the VL53L7CX driver without Adafruit Blinka.
"""

import struct
import time

from smbus2 import SMBus, i2c_msg

WHO_AM_I = 0x0F
WHO_AM_I_VALUE = 0x33
CTRL_REG1 = 0x20
CTRL_REG4 = 0x23
CTRL_REG5 = 0x24
OUT_X_L = 0x28

# Full-scale selection written to CTRL_REG4 bits 5:4.
RANGE_2_G = 0b00
RANGE_4_G = 0b01
RANGE_8_G = 0b10
RANGE_16_G = 0b11

# Output data rate written to CTRL_REG1 bits 7:4.
DATARATE_50_HZ = 0b0100
DATARATE_100_HZ = 0b0101
DATARATE_200_HZ = 0b0110
DATARATE_400_HZ = 0b0111

# Counts per 1 g for a raw int16 reading in high-resolution mode.
# Same dividers as adafruit_lis3dh.
_DIVIDER = {
    RANGE_2_G: 16380,
    RANGE_4_G: 8190,
    RANGE_8_G: 4096,
    RANGE_16_G: 1365,
}

STANDARD_GRAVITY = 9.806


class LIS3DH:
    """Polled LIS3DH. Callers must serialize access with any other I2C user."""

    def __init__(self, bus: SMBus, address: int = 0x18,
                 data_rate: int = DATARATE_100_HZ, g_range: int = RANGE_4_G):
        if g_range not in _DIVIDER:
            raise ValueError(f"unsupported range {g_range}")
        self.bus = bus
        self.address = address
        self._range = g_range

        who = self._read_byte(WHO_AM_I)
        if who != WHO_AM_I_VALUE:
            raise RuntimeError(
                f"LIS3DH not found at 0x{address:02x} "
                f"(WHO_AM_I=0x{who:02x}, expected 0x33)"
            )

        # Reboot memory, then turn on high-resolution measurement.
        # 0x80 in CTRL_REG5 reboots; 0x88 in CTRL_REG4 is BDU | HR.
        self._write_byte(CTRL_REG5, 0x80)
        time.sleep(0.01)
        self._write_byte(CTRL_REG1, 0x07 | ((data_rate & 0x0F) << 4))
        self._write_byte(CTRL_REG4, 0x88 | ((g_range & 0x03) << 4))
        # The first conversion is not ready until one output period has passed.
        # Reading before that returns zeros.
        time.sleep(0.02)

    def _read_byte(self, register: int) -> int:
        write = i2c_msg.write(self.address, [register & 0xFF])
        read = i2c_msg.read(self.address, 1)
        self.bus.i2c_rdwr(write, read)
        return list(read)[0]

    def _write_byte(self, register: int, value: int) -> None:
        msg = i2c_msg.write(self.address, [register & 0xFF, value & 0xFF])
        self.bus.i2c_rdwr(msg)

    def read_accel_mps2(self):
        """Return (x, y, z) acceleration in m/s^2."""
        # Bit 7 of the register address turns on auto-increment.
        write = i2c_msg.write(self.address, [OUT_X_L | 0x80])
        read = i2c_msg.read(self.address, 6)
        self.bus.i2c_rdwr(write, read)
        x, y, z = struct.unpack("<hhh", bytes(read))
        scale = STANDARD_GRAVITY / _DIVIDER[self._range]
        return x * scale, y * scale, z * scale
