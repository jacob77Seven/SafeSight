#!/usr/bin/env python3
"""SafeSight on-body runtime: geometry, gait, and a spatialized beep.

Read both sensors, update the walking/still label from the last 1.5 s of
acceleration, run ``ObstaclePipeline`` on each ToF frame, and play a short
stereo tone on the PCM5102 when something on the walking line is closing
in. Stages can be removed individually so a later test can compare:

    --no-gait          one TTC threshold (the walking one) for every frame
    --fixed-ttc 1.5    that many seconds, gait ignored for the threshold
    --no-cone          warn on clusters outside the central columns too
    --no-floor         do not treat the bottom rows as ground
    --no-limb-filter   do not ignore the near field

MQTT is optional and split the way the debug visualizer expects:

    --mqtt-raw      sensors/imu and sensors/tof (existing payloads)
    --mqtt-debug    debug/pipeline  (clusters, gait, TTC, warning)
    --mqtt-audio    debug/audio     (the tone that was requested)

The model file is JSON from ``pc/train_gait.ipynb``. If it is missing,
a built-in prior is used and the log says so. Copy a trained file to
``pi/models/gait_logreg.json`` before trusting the gait label.

Stop ``publish_sensors.py`` before a real run. It holds I²C. ``--fake``
runs the loop on synthetic frames and does not touch the bus.

Latency: the geometric stage on an 8×8 grid is well under a millisecond.
The ToF frame itself arrives about every 66 ms at 15 Hz, and the I²C
read is inside that. ``e2e_ms`` in the log is the time from the start of
the read to the end of the pipeline. The tone starts just after that.
"""

import argparse
import logging
import os
import signal
import sys
import time

from audio_warn import TonePlayer, cooldown_s, render_tone
from gait import GaitModel, GaitTracker, builtin_prior
from mqtt_debug import AUDIO_TOPIC, DEBUG_TOPIC, MqttDebug
from pipeline import ObstaclePipeline
from sensor_session import SensorSession
from session_io import JsonlWriter

log = logging.getLogger("runtime")

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gait_logreg.json")


def load_model(path):
    if path and os.path.isfile(path):
        model = GaitModel.load(path)
        log.info("gait model %s", model.source)
        return model
    log.warning("no gait model at %s. Using the built-in prior.", path)
    return builtin_prior()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the SafeSight obstacle warning loop.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. 0 runs until Ctrl+C.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Logistic-regression JSON from the training notebook.")
    parser.add_argument("--log", default="", help="JSONL of warnings and rates. Default runtime_<time>.jsonl under data/.")
    parser.add_argument("--imu-hz", type=float, default=50.0)
    parser.add_argument("--tof-hz", type=int, default=15)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--mqtt-raw", action="store_true", help="Publish sensors/imu and sensors/tof.")
    parser.add_argument("--mqtt-debug", action="store_true", help="Publish debug/pipeline.")
    parser.add_argument("--mqtt-audio", action="store_true", help="Publish debug/audio when a tone is requested.")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--audio-device", default="", help="sounddevice name or aplay -D value. Empty uses the default DAC.")
    parser.add_argument("--no-gait", action="store_true", help="Do not scale TTC by walking vs still.")
    parser.add_argument("--fixed-ttc", type=float, default=None, help="Fixed TTC warning threshold in seconds.")
    parser.add_argument("--no-cone", action="store_true")
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--no-limb-filter", action="store_true")
    parser.add_argument("--floor-rows", type=int, default=2)
    parser.add_argument("--limb-radius-mm", type=float, default=1200.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _warning_key(warning):
    return (
        warning["kind"],
        round(warning["azimuth_deg"] / 5.0),
        None if warning["ttc_s"] is None else round(warning["ttc_s"], 1),
    )


def main(argv):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    stop_flag = []

    def _handle(signum, _frame):
        log.info("signal %s, stopping", signum)
        stop_flag.append(True)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    model = load_model(args.model)
    tracker = GaitTracker(model)
    pipe = ObstaclePipeline(
        side=8,
        use_cone=not args.no_cone,
        use_floor=not args.no_floor,
        use_limb_filter=not args.no_limb_filter,
        use_gait=not args.no_gait,
        fixed_ttc_s=args.fixed_ttc,
        floor_rows=args.floor_rows,
        limb_radius_mm=args.limb_radius_mm,
    )
    session = SensorSession(
        imu_hz=args.imu_hz,
        tof_hz=args.tof_hz,
        fake=args.fake,
        no_fallback=args.no_fallback,
        side=8,
    )
    player = None if args.no_audio else TonePlayer(device=args.audio_device or None)
    log_path = args.log
    if not log_path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("data", "runtime_%s.jsonl" % stamp)
    writer = JsonlWriter(log_path)
    mqtt = None
    want_mqtt = args.mqtt_raw or args.mqtt_debug or args.mqtt_audio
    last_tone = 0.0
    last_key = None
    pipeline_ms = 0.0
    e2e_ms = 0.0
    frames = 0
    try:
        session.start()
        if want_mqtt:
            mqtt = MqttDebug(args.broker, args.port, "safesight-runtime")
            mqtt.start()
        writer.write({
            "rec": "meta",
            "ts": time.time(),
            "mono": time.monotonic(),
            "model": model.source,
            "use_gait": not args.no_gait,
            "use_cone": not args.no_cone,
            "use_floor": not args.no_floor,
            "use_limb_filter": not args.no_limb_filter,
            "fixed_ttc_s": args.fixed_ttc,
        }, force_flush=True)
        log.info(
            "runtime cone=%s floor=%s limb=%s gait=%s fixed_ttc=%s",
            not args.no_cone,
            not args.no_floor,
            not args.no_limb_filter,
            not args.no_gait,
            args.fixed_ttc,
        )
        started = time.monotonic()
        last_report = started
        report_imu = 0
        report_tof = 0
        while not stop_flag:
            loop_start = time.perf_counter()
            for sample in session.drain_imu():
                mono, _ts, x, y, z, _fake = sample
                tracker.push(mono, x, y, z)
                report_imu += 1
                if mqtt is not None and args.mqtt_raw:
                    mqtt.publish_imu(sample)
            gait_label, gait_p = tracker.update(time.monotonic())
            for frame in session.drain_tof():
                mono, ts, distances, status, temp, fake, read_ms = frame
                report_tof += 1
                if mqtt is not None and args.mqtt_raw:
                    mqtt.publish_tof(frame, "8x8")
                t0 = time.perf_counter()
                result = pipe.update(distances, status, mono, gait_label)
                pipeline_ms = (time.perf_counter() - t0) * 1000.0
                e2e_ms = (time.monotonic() - mono) * 1000.0
                frames += 1
                result["ts"] = ts
                result["mono"] = round(mono, 4)
                result["gait_p"] = None if gait_p is None else round(gait_p, 3)
                result["sit_to_stand"] = bool(tracker.sit_to_stand)
                result["pipeline_ms"] = round(pipeline_ms, 3)
                result["e2e_ms"] = round(e2e_ms, 1)
                result["read_ms"] = round(read_ms, 1)
                result["tof_fake"] = bool(fake)
                result["temp_c"] = temp
                if mqtt is not None and args.mqtt_debug:
                    mqtt.publish(DEBUG_TOPIC, result)
                warning = result["warning"]
                if warning is None:
                    continue
                now = time.monotonic()
                kind = warning["kind"]
                gap = cooldown_s(warning["ttc_s"], kind == "step_down")
                key = _warning_key(warning)
                if key == last_key and (now - last_tone) < gap:
                    continue
                pcm, meta = render_tone(
                    warning["azimuth_deg"],
                    warning["ttc_s"],
                    kind == "step_down",
                )
                played = False
                if player is not None:
                    played = player.play(pcm, meta)
                # Start the cooldown even when the DAC was busy or absent.
                # Otherwise a warning that cannot be played is retried on
                # every ToF frame and the log fills up at 15 Hz.
                last_tone = now
                last_key = key
                event = {
                    "rec": "warn",
                    "ts": ts,
                    "mono": round(now, 4),
                    "played": bool(played),
                    "gait": gait_label,
                    "kind": kind,
                    "azimuth_deg": warning["azimuth_deg"],
                    "depth_mm": warning["depth_mm"],
                    "ttc_s": warning["ttc_s"],
                    "freq_hz": meta["freq_hz"],
                    "pan": meta["pan"],
                    "e2e_ms": round(e2e_ms, 1),
                }
                writer.write(event, force_flush=True)
                log.info(
                    "WARN %s gait=%s depth=%.0f mm ttc=%s az=%.0f e2e=%.0f ms played=%s",
                    kind,
                    gait_label,
                    warning["depth_mm"],
                    warning["ttc_s"],
                    warning["azimuth_deg"],
                    e2e_ms,
                    played,
                )
                if mqtt is not None and args.mqtt_audio:
                    mqtt.publish(AUDIO_TOPIC, event)
            now = time.monotonic()
            if now - last_report >= 5.0:
                span = now - last_report
                imu_hz = report_imu / span
                tof_hz = report_tof / span
                log.info(
                    "rate imu %.1f Hz tof %.1f Hz pipeline %.2f ms e2e %.0f ms gait=%s p=%s",
                    imu_hz,
                    tof_hz,
                    pipeline_ms,
                    e2e_ms,
                    gait_label,
                    "none" if gait_p is None else "%.2f" % gait_p,
                )
                writer.write({
                    "rec": "rate",
                    "ts": time.time(),
                    "mono": round(now, 4),
                    "imu_hz": round(imu_hz, 2),
                    "tof_hz": round(tof_hz, 2),
                    "pipeline_ms": round(pipeline_ms, 3),
                    "e2e_ms": round(e2e_ms, 1),
                    "gait": gait_label,
                    "gait_p": None if gait_p is None else round(gait_p, 3),
                    "frames": frames,
                })
                report_imu = 0
                report_tof = 0
                last_report = now
            if args.duration and (now - started) >= args.duration:
                log.info("duration %.1f s elapsed", args.duration)
                break
            # Sleep out the rest of a 10 ms tick. The sensors are on their
            # own threads, so this only paces the pipeline.
            spent = time.perf_counter() - loop_start
            time.sleep(max(0.0, 0.01 - spent))
    finally:
        session.close()
        if player is not None:
            player.close()
        if mqtt is not None:
            mqtt.close()
        writer.close()
        log.info("closed %s", writer.path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
