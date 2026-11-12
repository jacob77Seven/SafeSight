# SafeSight

Debug setup for a headless Raspberry Pi Zero 2W that reads an accelerometer and a multizone time-of-flight sensor, then streams the samples to this PC.

The Pi has no display. This PC reaches it over USB Ethernet and plots the live data. Nothing here is a production path: the broker allows anonymous clients, and the link is only the USB cable between the two machines.

## What is connected

| Piece | Role |
| --- | --- |
| Raspberry Pi Zero 2W | Reads the sensors and runs the Mosquitto broker |
| Adafruit LIS3DH | 3-axis accelerometer on I²C `0x18` |
| Pololu VL53L7CX | 8×8 time-of-flight sensor on I²C `0x29` |
| GY-PCM5102A | I²S DAC, not part of the MQTT stream |
| This PC | SSH host and the live plot |

Both sensors share I²C bus 1 (GPIO2 SDA, GPIO3 SCL). `i2cdetect -y 1` shows `0x18` and `0x29`.

### Wiring

| Device | Interface | Pi connection |
| --- | --- | --- |
| GY-PCM5102A DAC | I²S | VIN → 5 V, GND → GND, SCK → GND, BCK → GPIO18, LCK → GPIO19, DIN → GPIO21 |
| Adafruit LIS3DH | I²C | VIN → 3.3 V, GND → GND, SDA → GPIO2, SCL → GPIO3, CS → 3Vo (I²C mode) |
| Pololu VL53L7CX | I²C | VIN → 3.3 V, GND → GND, SDA → GPIO2, SCL → GPIO3 |

The DAC overlay is `dtoverlay=hifiberry-dac`.

## Network and SSH

The Pi's USB port enumerates on this PC as an Ethernet adapter. There is no DHCP and no route to the internet on that interface. Both sides use a link-local address (`169.254.x.x`).

The Pi's address on `usb0` right now is **169.254.239.255/16**. It can change. On the Pi, `hostname -I` prints the address the visualizer should use. The publisher prints it when it starts and when it stops.

SSH is already set up in `~/.ssh/config` on this PC:

```
Host pi
    HostName raspberrypi.local
    User jPi0
    IdentityFile ~/.ssh/connect-to-pi
```

Connect and open the project directory:

```bash
ssh pi
cd ~/Documents/SafeSight
```

If `raspberrypi.local` does not resolve, use the link-local address:

```bash
ssh -i ~/.ssh/connect-to-pi jPi0@169.254.239.255
```

The login user is `jPi0`. The `pi` name is only the SSH host alias. The Pi project directory is `~/Documents/SafeSight`. This repo is the copy on the PC. `pi/` is what runs on the Pi, and `pc/` is the visualizer.

The USB link cannot resolve `raspbian.raspberrypi.org` or PyPI. Package installs on the Pi were done by copying debs and wheels from this PC. There is also no NTP, so the clock does not survive a reboot. `ts` in the MQTT payloads is the Pi's clock.

## MQTT debug stream

Mosquitto runs on the Pi as `mosquitto.service` and listens on `0.0.0.0:1883` with anonymous access. The config on the Pi is `/etc/mosquitto/conf.d/safesight.conf` (same text as `pi/mosquitto-safesight.conf`). Leave that config on this USB link only.

The publisher sends JSON, QoS 0:

| Stream | Topic | Payload |
| --- | --- | --- |
| LIS3DH | `sensors/imu` | `ts` (Unix seconds), `x`, `y`, `z` in m/s², `fake` |
| VL53L7CX | `sensors/tof` | `ts`, `resolution` (`8x8` or `4x4`), `distances` (mm), `status`, `silicon_temp_c`, `fake` |

Zone 0 is the bottom-left zone in ST's numbering. `distances` is row-major, row 0 along the bottom. Status 5 and 9 are valid ranges. `fake: true` means that stream is synthetic because the sensor or its library did not start.

### Start the publisher on the Pi

```bash
ssh pi
cd ~/Documents/SafeSight
./venv/bin/python publish_sensors.py
```

Flags: `--imu-hz 50`, `--tof-resolution 8`, `--tof-hz 15`, `--fake`, `--duration 20`, `--no-fallback`. A foreground run is the easiest way to watch the log. To leave it running:

```bash
nohup ./venv/bin/python publish_sensors.py > publish.log 2>&1 &
```

Ctrl+C stops a foreground run. The IMU target is 50 Hz. Sharing the bus with the ToF usually lands near 40 Hz. The ToF tops out at 15 Hz in 8×8 and 60 Hz in 4×4. Firmware upload on ToF init takes about 15 seconds at the Pi's 100 kHz I²C clock.

### Run the visualizer on this PC

A virtualenv in `pc/.venv` already has the dependencies. From this repo:

```bash
cd pc
.venv\Scripts\python visualize_sensors.py --broker 169.254.239.255
```

To recreate that environment:

```bash
cd pc
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python visualize_sensors.py --broker 169.254.239.255
```

`--broker` overrides `DEFAULT_BROKER` at the top of `pc/visualize_sensors.py`. The window shows a scrolling x/y/z plot, a distance heatmap, the same distances as a text grid, and a status line with the MQTT connection, the payload timestamp, and the measured rate. Parentheses in the grid are zones whose status is not 5 or 9.

### Libraries

`pi/requirements.txt` is `paho-mqtt` and `smbus2`. `pi/lis3dh.py` reads the accelerometer directly. The scale matches Adafruit's LIS3DH high-resolution mode. Adafruit Blinka is not installed. The USB link cannot reach PyPI, and Blinka is a large stack for the same register reads.

The VL53L7CX uses a pure-Python port of ST's VL53L5CX ultra-lite driver, which does not need a compiler:

https://github.com/Abstract-Horizon/vl53l5cx_python

ST documents the VL53L7CX as driver-compatible with the VL53L5CX. The port is not on PyPI. On a machine that has internet:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install "vl53l5cx @ git+https://github.com/Abstract-Horizon/vl53l5cx_python.git"
```

On this Pi the venv was installed from wheels and a checkout copied from the PC. The first publisher run patches two bugs in that port (range-sigma buffer size, and disabled result headers left set). If a sensor still fails, the log says so and that stream switches to fake data unless you pass `--no-fallback`.
