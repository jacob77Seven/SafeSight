# SafeSight edge pipeline

The wearable path is a geometric obstacle detector on the VL53L7CX, a walking-versus-still label from the LIS3DH, and a short stereo tone on the PCM5102. Each stage lives in its own module so it can be turned off from `runtime.py` without editing the others.

| Module | Runs on | Role |
| --- | --- | --- |
| `pi/collect_data.py` | Pi | Log both sensors and typed labels to JSONL |
| `pc/train_gait.ipynb` | PC | Fit a logistic regression and write JSON weights |
| `pi/runtime.py` | Pi | Sensors, geometry, gait, tone, MQTT debug |
| `pi/pipeline.py` | Pi | Median, clusters, cone, floor, limb radius, TTC |
| `pi/gait.py` | both | The same 1.5 s features on the PC and the Pi |
| `pi/audio_warn.py` | Pi | Pan and pitch for one beep |
| `pi/sensor_session.py` | Pi | LIS3DH and VL53L7CX threads, fake-data fallback |
| `pi/mqtt_debug.py` | Pi | Existing sensor topics plus `debug/pipeline` and `debug/audio` |
| `pi/session_io.py` | both | JSONL reader and writer |

The Pi venv has `paho-mqtt`, `smbus2`, and the pure-Python VL53L5CX driver. It does not have numpy, scikit-learn, or sounddevice. Inference is a dot product and a sigmoid. Audio goes to `aplay` on the hifiberry device unless `sounddevice` is already importable. Do not add packages on the Pi: the USB link has no route to PyPI.

`publish_sensors.py` is unchanged. Collect and runtime import it for the IMU open, the ToF wrapper, the driver patch, and the fake sensors. Stop the publisher before a real collect or runtime. Both want I²C bus 1, and the publisher already holds it. `--fake` on either script does not open the bus, so it can run beside the publisher.

## Collect

On the Pi, from `~/Documents/SafeSight`:

```bash
./venv/bin/python collect_data.py --mqtt
```

| Flag | Meaning |
| --- | --- |
| `--duration` | Stop after this many seconds. `0` runs until Ctrl+C. |
| `--out` | JSONL path. Default `data/session_<YYYYMMDD_HHMMSS>.jsonl`. |
| `--imu-hz` | LIS3DH target. Default 50. Sharing the bus with the ToF usually lands near 40. |
| `--tof-hz` | VL53L7CX target. 8×8 is clamped at 15 Hz, same as the publisher. |
| `--mqtt` | Also publish `sensors/imu` and `sensors/tof`. Off by default so a battery session does not need the broker. |
| `--rotate-s` | Open a new part file after this many seconds. `0` keeps one file. |
| `--fake` | Synthetic sensors, for a dry run of the file format. |
| `--no-fallback` | Exit if a real sensor fails to open, instead of switching that stream to fake data. |

The log prints a control path, defaulting to the JSONL name with a `.ctl` suffix. The file is emptied when the session starts. Append a label while the session runs, from the same terminal if it is a TTY, or from a second shell:

```bash
echo walking >> data/session_<time>.ctl
echo still >> data/session_<time>.ctl
echo obstacle >> data/session_<time>.ctl
echo step-down >> data/session_<time>.ctl
echo arm-wave >> data/session_<time>.ctl
```

Any other text is stored as-is. Labels flush to disk immediately. Sensor rows flush about once a second so a power cut loses at most that much, and the card is not rewritten at 50 Hz.

Each line is one JSON object:

| `rec` | Fields |
| --- | --- |
| `meta` | rates, resolution, the label names |
| `imu` | `mono`, `ts`, `x`, `y`, `z` (m/s²), `fake` |
| `tof` | `mono`, `ts`, `resolution`, `distances` (mm, 64 ints, row-major), `status`, `silicon_temp_c`, `fake` |
| `event` | `mono`, `ts`, `label` |

`mono` is `time.monotonic`, taken just before the read. Use it for windowing. `ts` is `time.time`. The Pi has no NTP and no fake-hwclock, so `ts` can be months off after a reboot and must not be subtracted across boots. `mono` also resets at boot, so train one file at a time or keep files from different boots separate. The notebook does that.

Zone 0 is the bottom-left cell. Status 5 and 9 are valid ranges. `fake: true` means that stream is synthetic.

Progress prints every 5 seconds: elapsed time, IMU Hz, ToF Hz, the control path, and bytes written. Ctrl+C or SIGTERM closes the file.

## Train

On the PC, with `pc/.venv`:

```bash
cd pc
.venv\Scripts\pip install -r requirements.txt
```

Open `train_gait.ipynb` from `pc/` (VS Code or Jupyter) and run all cells. Set `PATHS` to one or more JSONL files. Leave it empty and the notebook writes a synthesized still-then-walking session under `data/` so the cells run before any real collection. That demo fit is not a model of the wearer.

The notebook builds 1.5 s windows with a 0.5 s hop from the accelerometer only. Features are computed by `gait.features_from_samples`, the same function the Pi uses. A window is labeled `walking` or `still` from the latest gait event at or before the window start. A gait event inside the window drops it. `obstacle`, `step-down`, and `arm-wave` do not change the gait label. If fewer than eight windows are labeled, the next cell prints a table and waits for `MANUAL_LABELS` (`{window_index: "walking"}` or `"still"`). When `ipywidgets` is installed, that cell also shows a dropdown per unlabeled window.

The classifier is `sklearn.linear_model.LogisticRegression` on features standardized with `StandardScaler`. The split is by time inside each class. Test windows that start before the last training window ends are dropped, so the reported accuracy is a later slice of each bout. The notebook prints accuracy, a confusion matrix, a classification report, the coefficients, and the per-class feature means. Two plots follow: a still window and a walking window as magnitude versus time, and the decision in the `mag_std` / `band_energy` plane with the other six features held at the training mean.

Export:

- `pi/models/gait_logreg.json` is what the Pi loads. See `pi/models/INPUTS.md` for the vector.
- `pc/models/gait_logreg.joblib` is the sklearn object for a later PC experiment. The Pi never reads it.

The notebook checks that `GaitModel.score` matches `predict_proba` on the training rows before it writes the file.

## Run

Copy the pipeline modules and the JSON to the Pi (commands are in the README), then:

```bash
cd ~/Documents/SafeSight
./venv/bin/python runtime.py --mqtt-raw --mqtt-debug --mqtt-audio
```

| Flag | Meaning |
| --- | --- |
| `--duration` | Stop after this many seconds. `0` runs until Ctrl+C. |
| `--model` | JSON path. Default `pi/models/gait_logreg.json` next to the script. |
| `--log` | Warning and rate JSONL. Default `data/runtime_<time>.jsonl`. |
| `--imu-hz`, `--tof-hz` | Same defaults as the collector (50 and 15). |
| `--mqtt-raw` | Publish `sensors/imu` and `sensors/tof` in the existing shape. |
| `--mqtt-debug` | Publish `debug/pipeline`. |
| `--mqtt-audio` | Publish `debug/audio` when a tone is requested. |
| `--no-audio` | Log the warning and skip the DAC. |
| `--audio-device` | `sounddevice` name or `aplay -D` value. Empty uses the default DAC. |
| `--fake`, `--no-fallback` | Same fallback policy as the publisher. |
| `--no-gait` | One TTC threshold, the walking one, on every frame. |
| `--fixed-ttc 1.5` | That many seconds, gait ignored for the threshold. |
| `--no-cone` | Warn outside the central columns too. |
| `--no-floor` | Do not treat the bottom rows as ground. |
| `--no-limb-filter` | Do not ignore the near field. |
| `--floor-rows`, `--limb-radius-mm` | Override those two thresholds. |

The loop sleeps out a 10 ms tick. Sensors are on their own threads. IMU samples queue (the gait window needs them). ToF keeps a few frames and drops the oldest so a slow moment does not warn about a scene that has already moved past.

Every 5 seconds the log prints IMU Hz, ToF Hz, the last pipeline time, end-to-end time, and the gait label. `e2e_ms` is from the start of that ToF read to the end of `ObstaclePipeline.update`. The geometric stage on an 8×8 grid is well under a millisecond. The frame itself arrives about every 67 ms at 15 Hz, which is the bound on how soon a new obstacle can be seen. The tone starts just after the pipeline returns. A beep that is already playing is dropped rather than blocking the loop.

If a sensor fails to open, that stream falls back to the publisher's fake data and the log says so, unless `--no-fallback` is set. A failed read is counted and retried. It does not kill the other stream.

## MQTT debug

Broker, port, and the two sensor payloads are the same as `publish_sensors.py`. QoS 0, JSON.

`debug/pipeline` is one object per ToF frame:

- `gait`, `gait_p`, `sit_to_stand`
- `clusters`: `row`, `col`, `azimuth_deg`, `elevation_deg`, `depth_mm`, `closing_mm_s`, `ttc_s`, `in_cone`, `is_floor`, `step_down`, `ignored_limb`, `warn`
- `warning`: the cluster that fired, or null. `kind` is `ttc` or `step_down`.
- `pipeline_ms`, `e2e_ms`, `read_ms`, `valid_zones`, `tof_fake`, `temp_c`

`debug/audio` is the warning event: kind, azimuth, depth, TTC, frequency, pan, gait, and whether playback started.

The PC visualizer subscribes to the sensor topics. It does not draw `debug/pipeline`. `mosquitto_sub -t debug/pipeline` on the Pi is enough to watch clusters while tuning a threshold.

## Geometry

The VL53L7CX is treated as a 60°×60° image (about 90° on the diagonal). Index `row * 8 + col`, row 0 along the bottom, column 0 on the wearer's left. Column 0 is azimuth −30°, column 7 is +30°. Row 0 is elevation −30°.

Distances are millimetres. Speeds are millimetres per second. TTC is depth / closing speed, in seconds. Invalid status codes are dropped before any of that. So are depths outside (0, 4000] mm.

1. **Median.** 3×3 median of valid neighbours, and only when at least three of them exist. A single sparkling zone is replaced by that median. A zone the median would move by more than 150 mm is put back when a 4-connected neighbour has the same depth (within 250 mm). A plain median erases a two-zone chair leg because the other cells in the window are the far background. Noise on a real wall is about 1–2 cm, under the 150 mm restore, so the wall still gets smoothed.

2. **Motion mask.** While walking, a zone is foreground when it is at least 60 mm closer than it was about 0.20 s ago. One frame of noise is 1–2 cm, and 1 m/s of walking is only about 7 cm per frame at 15 Hz, so a single-frame threshold either false-triggers or misses the wall. 60 mm over 0.2 s is 0.3 m/s. A baseline older than 0.50 s is ignored: a stalled I²C read is not motion, and using it would invent a huge closing speed.

   While still, the reference is a slow background (blend 0.02 per frame) instead of frame differencing. The first still frame copies the current grid, so a chair that is already there is the background. Later frames are compared before the blend, then the blend moves forward. An object 100 mm closer than that background is foreground. Frame differencing would miss a person who walks in and then stops, and it would beep on sensor noise while the wearer is seated.

3. **Clusters.** 4-connected components. Two zones merge only when their depths agree within 250 mm, so a near arm and a far wall that touch in the image stay two objects. Components smaller than 2 zones are dropped. Step-down cells are clustered separately from closing cells so a curb and a wall that touch stay two events.

4. **Cone.** ±15° of azimuth is the central columns of the 8-wide, 60° image (zone centres in columns 2 through 5). A cluster outside that cone is beside the walking line. `--no-cone` keeps the cluster in the debug topic and allows it to warn.

5. **Floor.** The bottom two rows are the lowest part of a forward-looking field, about 15°. They are ground. They warn only when the range jumps up by at least 350 mm (the ground fell away: a curb or the top of a stair). `--no-floor` or `--floor-rows 0` disables that. Aim the sensor down and the floor assumption is wrong; set the rows to 0.

6. **Limbs.** A cluster closer than 1200 mm whose closing speed is under 800 mm/s is treated as the wearer's arm or leg. A fast radial approach still warns. A missed obstacle at arm's length is the worse error. Step-down is not passed through this filter. `--no-limb-filter` disables it.

7. **TTC.** Closing speed is `(old depth − new depth) / dt`, positive when the cluster is approaching. Speeds under 50 mm/s do not produce a TTC. Non-closing clusters do not warn. The warning threshold is 2.2 s while walking and 0.7 s while still. `--no-gait` always uses 2.2 s. `--fixed-ttc` replaces both. The fired warning prefers a step-down, then the smallest TTC, then the nearest depth.

Gait scaling is why the still threshold exists. A person standing in a kitchen should not hear a beep for every cabinet inside a 2 s horizon. The same cabinet, approached at a walk, should beep early enough to matter.

## Gait

The window is 1.5 s, long enough for one or two steps and short enough to notice a stop. Features are the magnitude mean, standard deviation, and peak-to-peak, the per-axis standard deviations, the zero-crossing rate of the demeaned magnitude, and Goertzel energy in the step band (1.0 to 3.0 Hz). Magnitude is used so a rotation of the Pi in the pouch does not require a new model.

The tracker reports still until the window is full, so the first second after boot does not beep from an empty buffer. Sit-to-stand is not a gait cycle: if `||a| − 9.806|` reaches 4 m/s² inside a 0.35 s lookback, the label is forced to walking for 2 s. That is the moment the wearer is about to move, and the cautious TTC threshold should already be on.

The shipped JSON is whatever the notebook last wrote. A checkout that has only been run on the demo session is a synthetic fit. Train on collected `walking` and `still` labels before trusting it. Delete the file and the runtime falls back to `gait.builtin_prior()`, which separates a quiet trace from a ~2 Hz walking trace and is not a fit to a person.

## Audio

`render_tone` builds 120 ms of S16_LE stereo at 22050 Hz with 10 ms ramps. Equal-power pan: full left at azimuth −30°, full right at +30°, both channels at the centre. Pitch and level carry urgency.

| Situation | Frequency | Amplitude |
| --- | --- | --- |
| Step-down | 280 Hz | 0.45 |
| TTC under 0.6 s | 1400 Hz | 0.55 |
| TTC under 1.2 s | 950 Hz | 0.40 |
| Longer TTC | 620 Hz | 0.25 |

A second tone is dropped while one is playing. The cooldown is 0.18 s under 0.8 s TTC, 0.40 s otherwise, and 0.45 s for a step-down. `aplay` is fed from a side thread so a full pipe cannot stall the pipeline. With `--no-audio`, or with no backend at all, the warning is still logged and, if requested, published.

## Clock, dropouts, and latency

There is no NTP. Every interval in the pipeline and the gait window uses `time.monotonic` or `time.perf_counter`. `time.time` is only the `ts` field.

A ToF zone with a status other than 5 or 9 is missing. A gap in the IMU stream shortens the gait window; the label stays still until 1.5 s is covered again. A baseline gap over 0.50 s is not used as motion. The still-background is reseeded on the transition into still, so a walk into a new room does not beep for the furniture that was already there when the wearer stopped.

`pi/selftest.py` builds synthetic grids and IMU traces and checks the median, the cone, the floor, the limb radius, both TTC thresholds, sit-to-stand, the tone pan, and the JSONL round trip. `--quick` skips a two-second `--fake` collect and runtime. Neither path opens I²C.

```bash
cd ~/Documents/SafeSight
./venv/bin/python selftest.py
```
