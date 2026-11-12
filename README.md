# SafeSight

Wearable obstacle warning on a headless Raspberry Pi Zero 2W. An LIS3DH and a VL53L7CX share I²C. A PCM5102 plays a short spatialized tone when something is closing on the walking line. The same samples can be streamed to this PC over MQTT for debugging.

`pi/` runs on the Pi. `pc/` is the live plot and the gait-training notebook. Sensor geometry, gait, audio, and the debug topics are in [pc/PIPELINE.md](pc/PIPELINE.md).

## Hardware

I²C bus 1 is GPIO2 SDA and GPIO3 SCL. `i2cdetect -y 1` shows `0x18` and `0x29`. The DAC overlay is `dtoverlay=hifiberry-dac`.

| Device | Interface | Connection |
| --- | --- | --- |
| GY-PCM5102A | I²S | VIN → 5 V, GND → GND, SCK → GND, BCK → GPIO18, LCK → GPIO19, DIN → GPIO21 |
| Adafruit LIS3DH | I²C `0x18` | VIN → 3.3 V, GND → GND, SDA → GPIO2, SCL → GPIO3, CS → 3Vo |
| Pololu VL53L7CX | I²C `0x29` | VIN → 3.3 V, GND → GND, SDA → GPIO2, SCL → GPIO3 |

## Debug stream

The Pi can be connected over Ethernet (or ethernet over USB). Then, Mosquitto on the Pi listens on `0.0.0.0:1883` and allows anonymous clients. The config is `pi/mosquitto-safesight.conf`. Payloads are JSON, QoS 0.

| Topic | Payload |
| --- | --- |
| `sensors/imu` | `ts`, `x`, `y`, `z` (m/s²), `fake` |
| `sensors/tof` | `ts`, `resolution` (`8x8` or `4x4`), `distances` (mm), `status`, `silicon_temp_c`, `fake` |

`distances` is row-major. Zone 0 is the bottom-left cell. Status 5 and 9 are valid ranges. `fake: true` means that stream is synthetic.

On the Pi, from `~/Documents/SafeSight`:

```bash
./venv/bin/python publish_sensors.py
```

Leave it running with `nohup ./venv/bin/python publish_sensors.py > publish.log 2>&1 &`. Ctrl+C stops a foreground run. Defaults are 50 Hz IMU and 15 Hz ToF at 8×8. The two sensors share the bus, so measured rates are lower. `--fake` skips the hardware. `--no-fallback` exits if a sensor fails to open.

On this PC:

```bash
cd pc
.venv\Scripts\python visualize_sensors.py --broker raspberrypi.local
```

Create that environment with `python -m venv .venv` and `.venv\Scripts\pip install -r requirements.txt`.

Stop the publisher before the scripts below. It holds I²C.

## Obstacle warning

Collect on the Pi. Ctrl+C ends the session and closes the JSONL. `--duration 120` stops after 120 seconds.

```bash
./venv/bin/python collect_data.py --mqtt
echo walking >> data/session_<time>.ctl
echo still >> data/session_<time>.ctl
```

The log prints the real control-file path. `obstacle`, `step-down`, and `arm-wave` are stored too. Training uses `walking` and `still`. Window on `mono`. The Pi has no NTP, so `ts` can be wrong after a reboot.

Copy the file the log named, then train:

```powershell
scp pi:Documents/SafeSight/data/session_YYYYMMDD_HHMMSS.jsonl data\
```

Open `pc/train_gait.ipynb`, set `PATHS` to the JSONL files, and run all cells. The Pi loads `pi/models/gait_logreg.json`. The feature vector is `pi/models/INPUTS.md`. An empty `PATHS` overwrites that JSON with a demo fit.

Copy the code and the model back, then run:

```powershell
ssh pi "mkdir -p Documents/SafeSight/models"
scp pi\*.py pi:Documents/SafeSight/
scp pi\models\* pi:Documents/SafeSight/models/
```

```bash
./venv/bin/python runtime.py --mqtt-raw --mqtt-debug --mqtt-audio
```

`pi/selftest.py` checks the geometry on synthetic frames.

## Dependencies

The Pi venv needs `paho-mqtt`, `smbus2` (`pi/requirements.txt`), and the pure-Python [VL53L5CX driver](https://github.com/Abstract-Horizon/vl53l5cx_python). ST's VL53L7CX uses that driver. Install on a machine with internet, then copy the result to the Pi if the Pi does not have internet.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install "vl53l5cx @ git+https://github.com/Abstract-Horizon/vl53l5cx_python.git"
```

`pc/requirements.txt` is the visualizer and the notebook.
