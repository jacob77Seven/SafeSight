"""Walking-vs-still gait features and a logistic-regression loader.

The Pi does not run scikit-learn. Training on the PC (see
``pc/train_gait.ipynb``) fits a logistic regression and writes the
weights, the feature mean, and the feature scale as JSON. This module
is the whole inference implementation: a dot product and a sigmoid.

Windows are 1.5 s of LIS3DH samples. Timing inside a window uses
``time.monotonic`` values (the ``mono`` field), not the wall clock.
The Pi has no NTP, so ``time.time`` can jump; intervals must not.

Feature order is fixed. Do not reorder ``FEATURE_NAMES`` without
retraining. ``pi/models/INPUTS.md`` describes the same vector.
"""

import json
import math

# One and a half seconds matches the in-class walking-vs-sitting window:
# long enough for one or two steps, short enough to react to a stop.
WINDOW_S = 1.5
# Uniform rate used only for the frequency features. The sensor itself
# still runs at whatever rate the collector asked for.
RESAMPLE_HZ = 50
RESAMPLE_N = int(WINDOW_S * RESAMPLE_HZ)  # 75

# Bins inside a typical step cadence. Walking is energy here; standing
# still is not, even if the mean acceleration is still about 1 g.
BAND_HZ = (1.0, 1.5, 2.0, 2.5, 3.0)

FEATURE_NAMES = (
    "mag_mean",
    "mag_std",
    "mag_ptp",
    "x_std",
    "y_std",
    "z_std",
    "zcr_hz",
    "band_energy",
)

# Sit-to-stand is a brief vertical impulse, not a walk. | |a| - g | above
# this, while the classifier still says "still", forces the cautious
# (walking) threshold for a couple of seconds so the first step is not
# treated as standing.
GRAVITY = 9.806
SIT_TO_STAND_MPS2 = 4.0
SIT_TO_STAND_HOLD_S = 2.0
MIN_SAMPLES = 20


def _mean(values):
    return sum(values) / float(len(values))


def _std(values, mean):
    if len(values) < 2:
        return 0.0
    acc = 0.0
    for value in values:
        d = value - mean
        acc += d * d
    return math.sqrt(acc / float(len(values)))


def _resample(times, values, n):
    """Linear resample onto ``n`` uniform points from the first to the last stamp.

    Goertzel assumes even spacing. IMU reads jitter when the ToF thread
    holds the I²C bus, so the frequency features see a cleaned 50 Hz trace.
    """
    if len(times) < 2 or len(times) != len(values):
        return None
    t0 = times[0]
    t1 = times[-1]
    if t1 <= t0:
        return None
    span = t1 - t0
    out = []
    j = 0
    last = len(times) - 1
    for i in range(n):
        t = t0 + span * (i / float(n - 1))
        while j < last and times[j + 1] < t:
            j += 1
        if j >= last:
            out.append(values[-1])
            continue
        dt = times[j + 1] - times[j]
        if dt <= 0:
            out.append(values[j])
            continue
        w = (t - times[j]) / dt
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0
        out.append(values[j] * (1.0 - w) + values[j + 1] * w)
    return out


def _goertzel_power(samples, sample_rate, freq):
    """Power of one frequency bin. Pure Python, no FFT dependency."""
    n = len(samples)
    if n == 0 or sample_rate <= 0:
        return 0.0
    coeff = 2.0 * math.cos(2.0 * math.pi * freq / float(sample_rate))
    s1 = 0.0
    s2 = 0.0
    for value in samples:
        s0 = value + coeff * s1 - s2
        s2 = s1
        s1 = s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    if power < 0.0:
        power = 0.0
    return power / float(n)


def features_from_samples(samples):
    """Build one feature vector from ``(mono, x, y, z)`` samples.

    Returns a list in ``FEATURE_NAMES`` order, or None when the window
    is too short or has no time span. Magnitude is the Euclidean norm,
    so the feature does not depend on how the Pi is rotated in the pouch.
    """
    if samples is None or len(samples) < MIN_SAMPLES:
        return None
    times = [row[0] for row in samples]
    if times[-1] - times[0] < WINDOW_S * 0.6:
        # A 1.5 s label on a 0.2 s scrap is not a gait cycle.
        return None
    xs = [row[1] for row in samples]
    ys = [row[2] for row in samples]
    zs = [row[3] for row in samples]
    mags = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(xs, ys, zs)]
    mag_mean = _mean(mags)
    mag_std = _std(mags, mag_mean)
    mag_ptp = max(mags) - min(mags)
    x_std = _std(xs, _mean(xs))
    y_std = _std(ys, _mean(ys))
    z_std = _std(zs, _mean(zs))

    demeaned = [m - mag_mean for m in mags]
    crossings = 0
    for a, b in zip(demeaned, demeaned[1:]):
        if (a <= 0.0 < b) or (b <= 0.0 < a):
            crossings += 1
    duration = max(times[-1] - times[0], 1e-3)
    zcr_hz = crossings / duration

    uniform = _resample(times, demeaned, RESAMPLE_N)
    if uniform is None:
        band = 0.0
    else:
        band = 0.0
        for freq in BAND_HZ:
            band += _goertzel_power(uniform, RESAMPLE_HZ, freq)
    return [mag_mean, mag_std, mag_ptp, x_std, y_std, z_std, zcr_hz, band]


def _sigmoid(value):
    # Clamp so a wild feature cannot overflow math.exp on the Pi.
    if value > 40.0:
        return 1.0
    if value < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


class GaitModel:
    """Standardized logistic regression. Positive class is walking."""

    def __init__(self, mean, std, weights, bias, threshold=0.5, source="memory"):
        if not (len(mean) == len(std) == len(weights) == len(FEATURE_NAMES)):
            raise ValueError(
                "model width %d does not match %d features"
                % (len(weights), len(FEATURE_NAMES))
            )
        self.mean = [float(v) for v in mean]
        self.std = [float(v) if float(v) > 1e-6 else 1.0 for v in std]
        self.weights = [float(v) for v in weights]
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.source = source

    def score(self, features):
        """Return P(walking) in 0..1."""
        z = self.bias
        for feature, mean, scale, weight in zip(features, self.mean, self.std, self.weights):
            z += weight * ((float(feature) - mean) / scale)
        return _sigmoid(z)

    def predict_label(self, features):
        probability = self.score(features)
        label = "walking" if probability >= self.threshold else "still"
        return label, probability

    def to_dict(self):
        return {
            "type": "logreg",
            "positive_class": "walking",
            "window_s": WINDOW_S,
            "resample_hz": RESAMPLE_HZ,
            "features": list(FEATURE_NAMES),
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_dict(cls, payload, source="dict"):
        return cls(
            mean=payload["mean"],
            std=payload["std"],
            weights=payload["weights"],
            bias=payload["bias"],
            threshold=payload.get("threshold", 0.5),
            source=source,
        )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        names = payload.get("features")
        if list(names) != list(FEATURE_NAMES):
            raise ValueError("feature list in %s does not match gait.FEATURE_NAMES" % path)
        return cls.from_dict(payload, source=path)


def builtin_prior():
    """A cautious prior used until a trained file is copied onto the Pi.

    The midpoint of each standardized feature sits between a quiet
    standing trace and a ~2 Hz walking trace (see the numbers printed
    by the training notebook). Weights on ``mag_std`` and ``band_energy``
    are the actual decision. The others are small so a single noisy axis
    cannot flip the label. Replace this file by training.
    """
    return GaitModel(
        mean=[9.8, 0.9, 2.5, 0.6, 0.4, 0.9, 3.0, 1.5],
        std=[0.6, 0.8, 2.0, 0.6, 0.4, 0.9, 2.5, 2.0],
        weights=[0.1, 2.4, 0.6, 0.2, 0.2, 0.3, 0.4, 1.6],
        bias=-0.2,
        threshold=0.5,
        source="builtin-prior",
    )


class GaitTracker:
    """Rolling 1.5 s window plus the sit-to-stand override.

    Until the window is full the label is ``still``. A wearable that
    beeps for the first second after boot is worse than waiting one
    gait cycle. Sit-to-stand still punches through that, because the
    impulse is the moment the wearer is about to move.
    """

    def __init__(self, model, window_s=WINDOW_S):
        self.model = model
        self.window_s = window_s
        self.samples = []  # (mono, x, y, z)
        self.forced_until = 0.0
        self.last_probability = None
        self.last_features = None
        self.sit_to_stand = False

    def push(self, mono, x, y, z):
        self.samples.append((float(mono), float(x), float(y), float(z)))
        cutoff = float(mono) - self.window_s
        if len(self.samples) > 400:
            # Bound memory if mono ever stalls. 400 samples is ~8 s at 50 Hz.
            self.samples = [row for row in self.samples if row[0] >= cutoff]
        else:
            drop = 0
            for row in self.samples:
                if row[0] >= cutoff:
                    break
                drop += 1
            if drop:
                del self.samples[:drop]
        self.sit_to_stand = self._detect_sit_to_stand(float(mono))
        if self.sit_to_stand:
            self.forced_until = float(mono) + SIT_TO_STAND_HOLD_S

    def _detect_sit_to_stand(self, now):
        # Look at a third of a second. A sit-to-stand is one impulse, not
        # a sustained oscillation (that would already look like walking).
        recent = [row for row in self.samples if row[0] >= now - 0.35]
        if len(recent) < 5:
            return False
        peak = 0.0
        for _mono, x, y, z in recent:
            mag = math.sqrt(x * x + y * y + z * z)
            peak = max(peak, abs(mag - GRAVITY))
        return peak >= SIT_TO_STAND_MPS2

    def update(self, now):
        """Return ``(label, probability or None)``.

        The classifier waits until the buffer covers the full 1.5 s.
        Sit-to-stand does not: the impulse itself is the cue to leave
        still mode, including during the first second after boot.
        """
        now = float(now)
        if self._detect_sit_to_stand(now):
            self.forced_until = max(self.forced_until, now + SIT_TO_STAND_HOLD_S)
        if now < self.forced_until:
            self.sit_to_stand = True
            # Keep the last probability for the debug topic. The label is
            # forced, and the TTC stage treats it as walking.
            return "walking", self.last_probability
        self.sit_to_stand = False
        if not self.samples or (self.samples[-1][0] - self.samples[0][0]) < (self.window_s - 0.05):
            self.last_features = None
            self.last_probability = None
            return "still", None
        features = features_from_samples(self.samples)
        self.last_features = features
        if features is None:
            self.last_probability = None
            return "still", None
        label, probability = self.model.predict_label(features)
        self.last_probability = probability
        return label, probability
