# Gait model input

`runtime.py` scores one vector each time the 1.5 s LIS3DH buffer is full. The vector is 8 floats, in this order. `pc/train_gait.ipynb` writes `gait_logreg.json` beside this file. `gait.py` loads it. Do not reorder the names without training again.

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | `mag_mean` | Mean of the acceleration magnitude, m/s². Magnitude is `sqrt(x²+y²+z²)`, so the pouch orientation does not matter. |
| 1 | `mag_std` | Standard deviation of that magnitude. |
| 2 | `mag_ptp` | Peak-to-peak of the magnitude. |
| 3 | `x_std` | Standard deviation of x. |
| 4 | `y_std` | Standard deviation of y. |
| 5 | `z_std` | Standard deviation of z. |
| 6 | `zcr_hz` | Zero crossings per second of the demeaned magnitude. |
| 7 | `band_energy` | Sum of Goertzel power at 1.0, 1.5, 2.0, 2.5, and 3.0 Hz. |

Intervals inside the window use `mono` (`time.monotonic` on the Pi). The frequency features resample that window to 50 Hz, which is 75 samples. Standard deviation divides by N, the same convention as `StandardScaler` in scikit-learn.

The JSON object is a logistic regression:

- `mean` and `std` standardize each feature.
- `weights` and `bias` are the decision. The positive class is `walking`.
- `sigmoid(weights · z + bias) >= threshold` (0.5) is walking. Below that is still.

Until the buffer holds 1.5 s, the runtime reports still. A sit-to-stand impulse (`||a| − 9.806| ≥ 4` m/s²) forces walking for 2 s immediately, including during that first window.

If `gait_logreg.json` is missing, `runtime.py` uses the prior compiled into `gait.builtin_prior()` and says so in the log. Replace that prior by running the notebook on collected walks.
