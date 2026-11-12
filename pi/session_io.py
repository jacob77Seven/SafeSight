"""JSONL session logs shared by collection and the training notebook.

One JSON object per line. Record types:

* ``meta``   session header (rates, resolution)
* ``imu``    ``mono``, ``ts``, ``x``, ``y``, ``z``, ``fake``
* ``tof``    ``mono``, ``ts``, ``resolution``, ``distances``, ``status``,
  ``silicon_temp_c``, ``fake``
* ``event``  ``mono``, ``ts``, ``label``  (free text from the operator)
* ``rate``   periodic counters, ignored by training
* ``warn``   runtime warning, ignored by training

``mono`` is ``time.monotonic`` on the Pi that wrote the file. Use it for
windowing. ``ts`` is the Pi wall clock and can be wrong by months when
NTP is down. Do not subtract ``ts`` values from two different boots:
monotonic resets at boot, and these files do not share a clock across boots.
"""

import json
import os
import time


def default_out_path(prefix="session"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("data", "%s_%s.jsonl" % (prefix, stamp))


class JsonlWriter:
    """Line-buffered enough for a crash, flushed on a timer for the SD card.

    Flushing every sample at 50 Hz wears the card and costs current on a
    battery. Flushing once a second loses at most that much on a hard
    power cut. Event labels flush immediately so a marker typed just
    before unplugging is kept.
    """

    def __init__(self, path, rotate_s=0.0):
        self.base_path = path
        self.rotate_s = float(rotate_s or 0.0)
        self.part = 0
        self.opened_at = time.monotonic()
        self.path = path
        self._fh = None
        self._last_flush = time.monotonic()
        self.bytes = 0
        self._open(path)

    def _open(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self.path = path
        self.opened_at = time.monotonic()

    def write(self, record, force_flush=False):
        if self.rotate_s > 0 and (time.monotonic() - self.opened_at) >= self.rotate_s:
            self._rotate()
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._fh.write(line)
        self.bytes += len(line)
        now = time.monotonic()
        if force_flush or (now - self._last_flush) >= 1.0:
            self._fh.flush()
            self._last_flush = now

    def _rotate(self):
        self._fh.flush()
        self._fh.close()
        self.part += 1
        root, ext = os.path.splitext(self.base_path)
        self._open("%s_part%02d%s" % (root, self.part, ext or ".jsonl"))

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def load_records(paths):
    """Yield decoded records from one path or a list of paths."""
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError("%s:%d is not JSON (%s)" % (path, line_no, exc)) from exc
