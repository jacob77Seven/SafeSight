#!/usr/bin/env python3
"""Geometry, gait, and file-format checks. No I2C and no MQTT.

Run on the Pi or the PC from this directory:

    ./venv/bin/python selftest.py

``--quick`` skips the one-second fake collect/runtime smoke. The smoke
uses ``--fake`` and does not open the sensors, so it is safe while
``publish_sensors.py`` is holding the bus.
"""

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time

import audio_warn
import collect_data
import gait
import pipeline
import runtime
import session_io

SIDE = 8


def fail(message):
    raise SystemExit("FAIL " + message)


def check(condition, message):
    if not condition:
        fail(message)


def frame(fill, blocks, bad=None):
    """``blocks`` is (row, col, height, width, depth_mm)."""
    distances = [int(fill)] * (SIDE * SIDE)
    status = [5] * (SIDE * SIDE)
    for row0, col0, height, width, depth in blocks:
        for row in range(row0, row0 + height):
            for col in range(col0, col0 + width):
                distances[row * SIDE + col] = int(depth)
    if bad:
        for row, col, code in bad:
            status[row * SIDE + col] = code
    return distances, status


def run_frames(pipe, gait_label, frames):
    """``frames`` is (mono, distances, status). Returns the last result."""
    result = None
    for mono, distances, status in frames:
        result = pipe.update(distances, status, mono, gait_label)
    return result


def approach(gait_label, block, depth0, depth1, fill=2500, **pipe_kwargs):
    """Three frames, 0.1 s apart. The 0.2 s baseline sees the full delta."""
    pipe = pipeline.ObstaclePipeline(**pipe_kwargs)
    row, col, height, width = block
    result = None
    for mono in (0.0, 0.1, 0.2):
        depth = depth0 + (depth1 - depth0) * (mono / 0.2)
        distances, status = frame(fill, [(row, col, height, width, depth)])
        result = pipe.update(distances, status, mono, gait_label)
    return result


def test_median_and_masking():
    pipe = pipeline.ObstaclePipeline()
    raw = [2000.0] * (SIDE * SIDE)
    raw[4 * SIDE + 4] = 400.0  # one sparkling zone in the middle
    smoothed = pipe._median_grid(raw)
    check(smoothed[4 * SIDE + 4] == 2000.0, "sparkle was not replaced: %s" % smoothed[4 * SIDE + 4])

    pair = [2000.0] * (SIDE * SIDE)
    for index in (4 * SIDE + 4, 4 * SIDE + 5, 5 * SIDE + 4, 5 * SIDE + 5):
        pair[index] = 800.0
    kept = pipe._median_grid(pair)
    for index in (4 * SIDE + 4, 4 * SIDE + 5, 5 * SIDE + 4, 5 * SIDE + 5):
        check(kept[index] == 800.0, "2x2 object erased at %d (%s)" % (index, kept[index]))

    distances, status = frame(1800, [(3, 3, 2, 2, 700)], bad=[(3, 3, 0), (3, 4, 4), (4, 3, 2)])
    # One zone of the pair stays valid (status 5) but it has no agreeing
    # neighbour, so the median drops it. Status 9 elsewhere is valid.
    distances[6 * SIDE + 6] = 900
    status[6 * SIDE + 6] = 9
    grid = pipe._valid_grid(distances, status)
    check(grid[3 * SIDE + 3] is None, "status 0 was kept")
    check(grid[3 * SIDE + 4] is None, "status 4 was kept")
    check(grid[6 * SIDE + 6] == 900.0, "status 9 was dropped")
    distances[0] = 0
    distances[1] = 9000
    status[0] = 5
    status[1] = 5
    grid = pipe._valid_grid(distances, status)
    check(grid[0] is None and grid[1] is None, "out-of-range distances were kept")


def test_warnings_and_ablations():
    # Far, fast, centered. TTC = 1700 / (300 / 0.2) = 1.133 s.
    # Walking threshold is 2.2 s, still is 0.7 s, limb radius is 1.2 m.
    center = (3, 3, 3, 3)
    walking = approach("walking", center, 2000, 1700)
    check(walking["warning"] is not None, "walking approach did not warn: %s" % walking)
    check(walking["warning"]["kind"] == "ttc", walking["warning"])
    check(abs(walking["warning"]["azimuth_deg"]) < 15.0, walking["warning"])
    check(walking["warning"]["ttc_s"] < 2.2, walking["warning"])

    still = approach("still", center, 2000, 1700)
    check(still["warning"] is None, "still gait warned on a 1.1 s TTC: %s" % still["warning"])

    no_gait = approach("still", center, 2000, 1700, use_gait=False)
    check(no_gait["warning"] is not None, "--no-gait should use the walking threshold")

    tight = approach("walking", center, 2000, 1700, fixed_ttc_s=0.4)
    check(tight["warning"] is None, "fixed 0.4 s TTC should not fire on a 1.1 s contact")
    loose = approach("still", center, 2000, 1700, fixed_ttc_s=2.0)
    check(loose["warning"] is not None, "fixed 2.0 s TTC should fire")

    # Inside the limb radius, closing slower than 0.8 m/s: ignored.
    # 1000 -> 900 mm over 0.2 s is 500 mm/s, TTC 1.8 s.
    slow_limb = approach("walking", center, 1000, 900, fill=2500)
    check(slow_limb["warning"] is None, "slow near object warned: %s" % slow_limb)
    check(any(item["ignored_limb"] for item in slow_limb["clusters"]), slow_limb["clusters"])
    unfiltered = approach("walking", center, 1000, 900, fill=2500, use_limb_filter=False)
    check(unfiltered["warning"] is not None, "limb filter did not change the result")

    # 1000 -> 800 over 0.2 s is 1000 mm/s, above the limb-speed gate.
    fast_limb = approach("walking", center, 1000, 800, fill=2500)
    check(fast_limb["warning"] is not None, "fast near approach was ignored: %s" % fast_limb)

    # Left-hand columns. Centroid azimuth is past the 15 degree cone.
    side = approach("walking", (3, 0, 3, 3), 2000, 1700)
    check(side["warning"] is None, "off-cone cluster warned: %s" % side)
    check(side["clusters"] and not side["clusters"][0]["in_cone"], side["clusters"])
    side_open = approach("walking", (3, 0, 3, 3), 2000, 1700, use_cone=False)
    check(side_open["warning"] is not None, "--no-cone should warn beside the path")

    # Floor falls away by 500 mm. That is a step-down, not a closing wall.
    pipe = pipeline.ObstaclePipeline()
    results = []
    for mono, depth in ((0.0, 1000), (0.1, 1250), (0.2, 1500)):
        distances, status = frame(2200, [(0, 0, 2, SIDE, depth)])
        results.append(pipe.update(distances, status, mono, "walking"))
    step = results[-1]
    check(step["warning"] is not None and step["warning"]["kind"] == "step_down", step)
    pipe_nf = pipeline.ObstaclePipeline(use_floor=False)
    for mono, depth in ((0.0, 1000), (0.1, 1250), (0.2, 1500)):
        distances, status = frame(2200, [(0, 0, 2, SIDE, depth)])
        quiet = pipe_nf.update(distances, status, mono, "walking")
    check(quiet["warning"] is None, "--no-floor step-down still warned: %s" % quiet["warning"])

    # A chair that is already in view when the wearer goes still is the
    # background. It must not beep just because it is close.
    seated = pipeline.ObstaclePipeline()
    distances, status = frame(2500, [(3, 3, 3, 3, 900)])
    for mono in (0.0, 0.2, 0.4):
        stayed = seated.update(distances, status, mono, "still")
        check(stayed["warning"] is None, "static near object warned while still")

    # Compare against the background before blending it forward. A 100 mm
    # step toward the wearer is exactly the still-threshold. Blending
    # first would shrink that gap under the threshold.
    crept = pipeline.ObstaclePipeline()
    distances, status = frame(2000, [])
    crept.update(distances, status, 0.0, "still")
    distances, status = frame(2000, [(3, 3, 3, 3, 1900)])
    appeared = crept.update(distances, status, 0.2, "still")
    check(appeared["clusters"], "100 mm still-scene change produced no cluster: %s" % appeared)
    check(appeared["warning"] is None, "a 100 mm creep should not reach the still TTC")

    # Invalid zones cannot invent a closing obstacle.
    pipe = pipeline.ObstaclePipeline()
    for mono, depth in ((0.0, 2000), (0.1, 1850), (0.2, 1700)):
        distances, status = frame(2500, [(3, 3, 3, 3, depth)], bad=[
            (row, col, 0) for row in range(3, 6) for col in range(3, 6)
        ])
        blind = pipe.update(distances, status, mono, "walking")
    check(blind["warning"] is None and not blind["clusters"], blind)

    try:
        pipeline.ObstaclePipeline(not_a_setting=1)
        fail("unknown setting was accepted")
    except TypeError:
        pass

    encoded = json.dumps(walking)
    check("warning" in encoded and "clusters" in encoded, "pipeline result is not JSON")


def _trace(kind, seconds=2.0, hz=50.0):
    rows = []
    count = int(seconds * hz)
    for i in range(count):
        t = i / hz
        if kind == "still":
            rows.append((t, 0.02, -0.01, 9.806))
        else:
            w = 2.0 * math.pi * 2.0 * t
            rows.append((t, 1.6 * math.sin(w), 0.5 * math.sin(w + 0.8), 9.806 + 2.6 * math.sin(w)))
    return rows


def test_gait():
    check(gait.features_from_samples(_trace("still", 0.3)) is None, "short window produced features")
    still_f = gait.features_from_samples(_trace("still"))
    walk_f = gait.features_from_samples(_trace("walking"))
    check(still_f is not None and len(still_f) == len(gait.FEATURE_NAMES), still_f)
    check(walk_f[1] > still_f[1], "walking mag_std did not exceed still: %s vs %s" % (walk_f, still_f))
    check(walk_f[-1] > still_f[-1], "walking band energy did not exceed still")

    model = gait.builtin_prior()
    still_label, still_p = model.predict_label(still_f)
    walk_label, walk_p = model.predict_label(walk_f)
    check(still_label == "still" and still_p < 0.35, (still_label, still_p, still_f))
    check(walk_label == "walking" and walk_p > 0.65, (walk_label, walk_p, walk_f))

    tracker = gait.GaitTracker(model)
    for mono, x, y, z in _trace("walking", 0.4):
        tracker.push(mono, x, y, z)
    label, prob = tracker.update(0.4)
    check(label == "still" and prob is None, "partial window was classified: %s %s" % (label, prob))

    tracker = gait.GaitTracker(model)
    rows = _trace("still", 1.6)
    for mono, x, y, z in rows:
        tracker.push(mono, x, y, z)
    label, prob = tracker.update(rows[-1][0])
    check(label == "still" and prob is not None and prob < 0.5, (label, prob))

    # Sit-to-stand on a buffer that is not a full gait window yet.
    fresh = gait.GaitTracker(model)
    for i in range(8):
        fresh.push(i * 0.02, 0.0, 0.0, 16.0)
    label, _prob = fresh.update(0.16)
    check(label == "walking" and fresh.sit_to_stand, "sit-to-stand did not force walking")

    folder = tempfile.mkdtemp(prefix="safesight-gait-")
    path = os.path.join(folder, "model.json")
    model.save(path)
    loaded = gait.GaitModel.load(path)
    check(abs(loaded.score(walk_f) - model.score(walk_f)) < 1e-9, "reloaded model drifted")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["features"] = list(reversed(payload["features"]))
    bad = os.path.join(folder, "bad.json")
    with open(bad, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    try:
        gait.GaitModel.load(bad)
        fail("mismatched feature list was accepted")
    except ValueError:
        pass


def test_audio_and_log():
    pcm, meta = audio_warn.render_tone(-30.0, 1.5, False)
    check(meta["freq_hz"] == 620.0, meta)
    check(len(pcm) == int(audio_warn.SAMPLE_RATE * 0.12) * 4, len(pcm))
    left, right = _energy(pcm)
    check(left > right * 5, "left pan was not left-heavy %s %s" % (left, right))
    pcm, _meta = audio_warn.render_tone(30.0, 0.4, False)
    left, right = _energy(pcm)
    check(right > left * 5, "right pan was not right-heavy")
    pcm, meta = audio_warn.render_tone(0.0, None, True)
    check(meta["freq_hz"] == 280.0, meta)
    left, right = _energy(pcm)
    check(abs(left - right) / max(left, 1) < 0.05, "centre pan was uneven")
    check(audio_warn.cooldown_s(0.5, False) < audio_warn.cooldown_s(2.0, False), "urgent cooldown")
    check(audio_warn.cooldown_s(None, True) == 0.45, "step cooldown")

    folder = tempfile.mkdtemp(prefix="safesight-log-")
    path = os.path.join(folder, "session.jsonl")
    writer = session_io.JsonlWriter(path)
    writer.write({"rec": "meta", "mono": 0.0}, force_flush=True)
    writer.write({"rec": "imu", "mono": 0.02, "x": 0.0, "y": 0.0, "z": 9.8, "fake": True})
    writer.close()
    rows = list(session_io.load_records(path))
    check(len(rows) == 2 and rows[1]["rec"] == "imu", rows)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
    try:
        list(session_io.load_records(path))
        fail("bad JSONL was accepted")
    except ValueError:
        pass


def _energy(pcm):
    left = 0
    right = 0
    for i in range(0, len(pcm), 4):
        sample_l, sample_r = struct.unpack_from("<hh", pcm, i)
        left += sample_l * sample_l
        right += sample_r * sample_r
    return left, right


def test_fake_session():
    folder = tempfile.mkdtemp(prefix="safesight-fake-")
    out_path = os.path.join(folder, "session.jsonl")
    control = os.path.join(folder, "session.ctl")

    def poke():
        time.sleep(0.6)
        with open(control, "a", encoding="utf-8") as handle:
            handle.write("walking\n")

    threading.Thread(target=poke, daemon=True).start()
    code = collect_data.main([
        "--fake", "--duration", "2", "--out", out_path, "--control", control,
        "--imu-hz", "40", "--tof-hz", "10",
    ])
    check(code == 0, "collect_data --fake returned %s" % code)
    rows = list(session_io.load_records(out_path))
    kinds = set(row["rec"] for row in rows)
    check("meta" in kinds and "imu" in kinds and "tof" in kinds, kinds)
    labels = [row["label"] for row in rows if row["rec"] == "event"]
    check(labels == ["walking"], labels)
    imu_n = sum(1 for row in rows if row["rec"] == "imu")
    tof_n = sum(1 for row in rows if row["rec"] == "tof")
    check(imu_n >= 30, "only %d imu rows" % imu_n)
    check(tof_n >= 8, "only %d tof rows" % tof_n)
    # The existing topic schema, checked without a broker.
    imu = next(row for row in rows if row["rec"] == "imu")
    tof = next(row for row in rows if row["rec"] == "tof")
    for key in ("ts", "x", "y", "z", "fake"):
        check(key in imu, "imu record missing %s" % key)
    for key in ("ts", "resolution", "distances", "status", "silicon_temp_c", "fake"):
        check(key in tof, "tof record missing %s" % key)
    check(tof["resolution"] == "8x8" and len(tof["distances"]) == 64, tof["resolution"])

    log_path = os.path.join(folder, "runtime.jsonl")
    code = runtime.main([
        "--fake", "--no-audio", "--duration", "1.5",
        "--log", log_path,
        "--model", os.path.join(folder, "missing.json"),
        "--imu-hz", "40", "--tof-hz", "10",
    ])
    check(code == 0, "runtime --fake returned %s" % code)
    runtime_rows = list(session_io.load_records(log_path))
    check(any(row["rec"] == "meta" for row in runtime_rows), runtime_rows[:1])


def main(argv):
    parser = argparse.ArgumentParser(description="SafeSight geometry self-test.")
    parser.add_argument("--quick", action="store_true", help="Skip the fake collect/runtime smoke.")
    args = parser.parse_args(argv)
    test_median_and_masking()
    test_warnings_and_ablations()
    test_gait()
    test_audio_and_log()
    if not args.quick:
        test_fake_session()
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
