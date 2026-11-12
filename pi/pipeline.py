"""Geometric obstacle pipeline for an 8×8 VL53L7CX worn looking forward.

The sensor is a 60°×60° square field (about 90° on the diagonal). Zone 0
is the bottom-left cell. The list is row-major, so index = row * side + col
and row 0 is the bottom of the image.

Stages, in order, so each one can be switched off from ``runtime.py``:

1. Keep only zones whose status is 5 or 9 (ST: valid range). Other codes
   are "no target", "wrap", or a sigma failure. Using them as distances
   paints phantom obstacles.
2. 3×3 median over the valid neighbours. A single sparkling zone is common
   on this sensor at short range; the median rejects it without blurring
   a real edge the way a mean would. A cell the median would move by more
   than 150 mm is put back when a 4-connected neighbour agrees with it.
   That keeps a two-zone chair leg, which a plain 3×3 median would erase
   into the far background.
3. Motion mask.
   - Walking: compare with a baseline about 0.2 s earlier. One frame of
     noise is ~1–2 cm, so a single-frame threshold either false-triggers
     or misses a normal walking approach (1 m/s is only ~7 cm per frame
     at 15 Hz). 60 mm over 0.2 s is 0.3 m/s, above noise and below a
     slow walk into a wall.
   - Still: compare with a slow background. The scene should not move.
     An object that becomes closer than that background is the event.
     The background adapts over a few seconds so a stationary chair does
     not beep forever after the wearer sits down in front of it.
4. Connected components. Two foreground zones merge only when they share
   an edge and their depths agree within ``depth_link_mm``. That keeps a
   near arm and a far wall from becoming one blob just because they touch
   in the image.
5. Path cone. ±15° of azimuth is the central four columns of an 8-wide,
   60° image. Warnings outside that cone are someone passing beside the
   wearer, not something on the walking line.
6. Floor. The bottom two rows are the lowest ~15° of the field. On a
   forward-looking wearable those cells are ground. They do not warn
   unless the range jumps up by ``step_down_mm`` (the ground fell away:
   a curb or a stair). Aim the sensor down and this assumption is wrong;
   set ``floor_rows`` to 0.
7. Limb radius. Anything inside ~1.2 m that is not closing faster than
   ~0.8 m/s is treated as the wearer's own arm or leg. A fast radial
   approach still warns: that is either a real obstacle at arm's length
   or a hard arm swing, and a missed obstacle is the worse error.
8. Time to contact is depth / closing speed, and only when the cluster
   is actually getting closer. The gait label picks the threshold:
   walking warns early (about 2.2 s), standing still only when contact
   is under about 0.7 s. Sit-to-stand is handled in ``gait.py`` and
   arrives here already relabelled as walking.

Distances are millimetres. Speeds are millimetres per second. Times are
seconds. All of those cancel in the TTC ratio.
"""

STATUS_VALID = (5, 9)

# Default thresholds. The comments above are the reason for each number.
# Flags on the runtime override the booleans; the millimetre values are
# constructor arguments so a later ablation can change one of them.
DEFAULTS = dict(
    hfov_deg=60.0,
    vfov_deg=60.0,
    median_radius=1,
    # Sensor noise is about 1–2 cm. 150 mm is far above that and below the
    # jump from a real nearby surface to the background behind it.
    median_restore_mm=150.0,
    close_mm=60.0,
    baseline_s=0.20,
    still_background_mm=100.0,
    background_alpha=0.02,
    depth_link_mm=250.0,
    min_cluster_zones=2,
    cone_half_deg=15.0,
    floor_rows=2,
    step_down_mm=350.0,
    limb_radius_mm=1200.0,
    limb_fast_mm_s=800.0,
    min_closing_mm_s=50.0,
    ttc_walking_s=2.2,
    ttc_still_s=0.7,
    max_baseline_gap_s=0.50,
)


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _angle_deg(index, count, fov_deg):
    """Map a zone index to degrees. Index 0 is the negative edge of the FoV."""
    if count <= 1:
        return 0.0
    half = fov_deg / 2.0
    center = (count - 1) / 2.0
    return (index - center) / center * half


class ObstaclePipeline:
    def __init__(
        self,
        side=8,
        use_cone=True,
        use_floor=True,
        use_limb_filter=True,
        use_gait=True,
        fixed_ttc_s=None,
        **overrides
    ):
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise TypeError("unknown pipeline setting %s" % sorted(unknown))
        self.cfg = dict(DEFAULTS)
        self.cfg.update(overrides)
        self.side = int(side)
        self.use_cone = bool(use_cone)
        self.use_floor = bool(use_floor)
        self.use_limb_filter = bool(use_limb_filter)
        self.use_gait = bool(use_gait)
        self.fixed_ttc_s = None if fixed_ttc_s is None else float(fixed_ttc_s)
        # History entries are (mono, depth grid with None for invalid).
        self.history = []
        self.background = None  # floats or None, updated only while still
        self._was_still = False

    def _cell(self, grid, row, col):
        return grid[row * self.side + col]

    def _valid_grid(self, distances, status):
        count = self.side * self.side
        grid = [None] * count
        if not distances:
            return grid
        for index in range(min(count, len(distances))):
            code = status[index] if status and index < len(status) else 0
            if code not in STATUS_VALID:
                continue
            try:
                depth = float(distances[index])
            except (TypeError, ValueError):
                continue
            if depth <= 0.0 or depth > 4000.0:
                # The VL53L7CX useful range is about 2 cm to 3.5 m.
                # 0 mm is "no target" that leaked through a status bug.
                continue
            grid[index] = depth
        return grid

    def _supported(self, grid, row, col, raw):
        """True when a 4-connected neighbour is the same surface as ``raw``."""
        side = self.side
        link = float(self.cfg["depth_link_mm"])
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = row + dr
            cc = col + dc
            if rr < 0 or cc < 0 or rr >= side or cc >= side:
                continue
            other = grid[rr * side + cc]
            if other is not None and abs(other - raw) <= link:
                return True
        return False

    def _median_grid(self, grid):
        """Median-smooth the grid, then restore small real objects.

        A lone zone is replaced by the neighbourhood median (sparkle).
        Two adjacent zones at the same depth are kept even when the 3×3
        window is mostly background, because that pair is the smallest
        cluster ``min_cluster_zones`` is willing to warn on.
        """
        radius = int(self.cfg["median_radius"])
        side = self.side
        restore = float(self.cfg["median_restore_mm"])
        count = side * side
        medians = [None] * count
        for row in range(side):
            for col in range(side):
                index = row * side + col
                if grid[index] is None:
                    continue
                window = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr = row + dr
                        cc = col + dc
                        if rr < 0 or cc < 0 or rr >= side or cc >= side:
                            continue
                        value = grid[rr * side + cc]
                        if value is not None:
                            window.append(value)
                # Require a few real neighbours. A lone valid zone has no
                # median and is dropped below unless a neighbour agrees.
                if len(window) >= 3:
                    medians[index] = _median(window)
        out = [None] * count
        for row in range(side):
            for col in range(side):
                index = row * side + col
                raw = grid[index]
                if raw is None:
                    continue
                supported = self._supported(grid, row, col, raw)
                med = medians[index]
                if med is None:
                    out[index] = raw if supported else None
                elif abs(raw - med) <= restore:
                    out[index] = med
                elif supported:
                    out[index] = raw
                else:
                    out[index] = med
        return out

    def _baseline(self, now):
        """Return (grid, dt) from about ``baseline_s`` ago, or None."""
        if not self.history:
            return None
        target = now - float(self.cfg["baseline_s"])
        chosen = None
        for mono, grid in self.history:
            if mono <= target:
                chosen = (mono, grid)
            else:
                break
        if chosen is None:
            # Not enough history yet. One frame back is better than nothing
            # once two frames exist, but only if they are not the same stamp.
            if len(self.history) < 2:
                return None
            mono, grid = self.history[-2]
            chosen = (mono, grid)
        mono, grid = chosen
        dt = now - mono
        gap = float(self.cfg["max_baseline_gap_s"])
        if dt <= 1e-3 or dt > gap:
            # A stall (USB hiccup, long I²C retry) is not motion. Using it
            # as a baseline would invent a huge closing speed.
            return None
        return grid, dt

    def _prepare_background(self, grid, gait):
        """Seed the still-scene model before this frame is compared.

        The first still frame copies the grid. The chair already in view
        is then the background, not an obstacle that just appeared.
        Later still frames are compared against the seed from last time
        and only then blended forward. Blending first would shrink a
        brand-new approach by ``background_alpha`` and could hide a
        change that sits right on ``still_background_mm``.
        """
        still = gait == "still"
        if self.background is None or (still and not self._was_still):
            self.background = list(grid)

    def _blend_background(self, grid, gait):
        still = gait == "still"
        if still and self._was_still and self.background is not None:
            alpha = float(self.cfg["background_alpha"])
            for index, value in enumerate(grid):
                old = self.background[index]
                if value is None:
                    continue
                if old is None:
                    self.background[index] = value
                else:
                    self.background[index] = (1.0 - alpha) * old + alpha * value
        self._was_still = still

    def _foreground(self, grid, gait, now):
        """Return (mask, closing_mm_s per zone, step_mask, dt or None)."""
        count = self.side * self.side
        mask = [False] * count
        step = [False] * count
        closing = [0.0] * count
        baseline = self._baseline(now)
        dt = None
        if baseline is not None:
            previous, dt = baseline
            for index, value in enumerate(grid):
                old = previous[index]
                if value is None or old is None:
                    continue
                delta = old - value  # positive: this zone got closer
                closing[index] = delta / dt
                row = index // self.side
                if (
                    self.use_floor
                    and row < int(self.cfg["floor_rows"])
                    and (value - old) >= float(self.cfg["step_down_mm"])
                ):
                    step[index] = True
                if gait != "still" and delta >= float(self.cfg["close_mm"]):
                    mask[index] = True
        if gait == "still" and self.background is not None:
            # While still, the slow background is the reference. Frame
            # differencing would miss a person who walks in and then stops
            # inside the cone, and it would also beep on sensor noise.
            need = float(self.cfg["still_background_mm"])
            for index, value in enumerate(grid):
                old = self.background[index]
                if value is None or old is None:
                    continue
                if old - value >= need:
                    mask[index] = True
                    if dt is None or closing[index] <= 0.0:
                        # No fresh speed estimate. Treat the background gap
                        # as a slow approach so TTC stays finite and large
                        # instead of dividing by zero.
                        closing[index] = max(closing[index], need / 1.0)
        return mask, closing, step, dt

    def _components(self, grid, mask):
        side = self.side
        seen = [False] * (side * side)
        link = float(self.cfg["depth_link_mm"])
        clusters = []
        for start in range(side * side):
            if seen[start] or not mask[start]:
                continue
            stack = [start]
            seen[start] = True
            members = []
            while stack:
                index = stack.pop()
                members.append(index)
                row, col = divmod(index, side)
                depth = grid[index]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = row + dr
                    cc = col + dc
                    if rr < 0 or cc < 0 or rr >= side or cc >= side:
                        continue
                    nxt = rr * side + cc
                    if seen[nxt] or not mask[nxt]:
                        continue
                    other = grid[nxt]
                    if depth is None or other is None or abs(depth - other) > link:
                        continue
                    seen[nxt] = True
                    stack.append(nxt)
            clusters.append(members)
        return clusters

    def _describe(self, members, grid, closing, step_mask, gait):
        side = self.side
        rows = []
        cols = []
        depths = []
        speeds = []
        step_votes = 0
        for index in members:
            row, col = divmod(index, side)
            rows.append(row)
            cols.append(col)
            if grid[index] is not None:
                depths.append(grid[index])
            speeds.append(closing[index])
            if step_mask[index]:
                step_votes += 1
        if not depths:
            return None
        row_c = sum(rows) / float(len(rows))
        col_c = sum(cols) / float(len(cols))
        depth = sum(depths) / float(len(depths))
        speed = sum(speeds) / float(len(speeds))
        azimuth = _angle_deg(col_c, side, self.cfg["hfov_deg"])
        elevation = _angle_deg(row_c, side, self.cfg["vfov_deg"])
        in_cone = abs(azimuth) <= float(self.cfg["cone_half_deg"])
        is_floor = row_c < float(self.cfg["floor_rows"])
        step_down = step_votes * 2 >= len(members) and step_votes > 0
        ttc = None
        if speed >= float(self.cfg["min_closing_mm_s"]):
            ttc = depth / speed
        ignored = False
        if (
            self.use_limb_filter
            and not step_down
            and depth < float(self.cfg["limb_radius_mm"])
            and speed < float(self.cfg["limb_fast_mm_s"])
        ):
            ignored = True
        if self.fixed_ttc_s is not None:
            ttc_limit = self.fixed_ttc_s
        elif not self.use_gait or gait != "still":
            ttc_limit = float(self.cfg["ttc_walking_s"])
        else:
            ttc_limit = float(self.cfg["ttc_still_s"])
        on_path = in_cone or not self.use_cone
        # Floor cells are the ground plane, not an obstacle, unless the
        # ground just disappeared (step_down) or the floor stage is off.
        floor_blocked = self.use_floor and is_floor and not step_down
        warn = False
        if on_path and not ignored and not floor_blocked:
            if step_down:
                warn = True
            elif ttc is not None and ttc <= ttc_limit:
                warn = True
        return {
            "zones": len(members),
            "row": round(row_c, 2),
            "col": round(col_c, 2),
            "azimuth_deg": round(azimuth, 2),
            "elevation_deg": round(elevation, 2),
            "depth_mm": round(depth, 1),
            "closing_mm_s": round(speed, 1),
            "ttc_s": None if ttc is None else round(ttc, 3),
            "in_cone": bool(in_cone),
            "is_floor": bool(is_floor),
            "step_down": bool(step_down),
            "ignored_limb": bool(ignored),
            "warn": bool(warn),
        }

    def update(self, distances, status, mono, gait):
        """Consume one ToF frame. ``gait`` is ``walking`` or ``still``."""
        now = float(mono)
        raw = self._valid_grid(distances, status)
        grid = self._median_grid(raw)
        self._prepare_background(grid, gait)
        mask, closing, step_mask, dt = self._foreground(grid, gait, now)
        self._blend_background(grid, gait)
        # Step-down zones are a separate family of components so a closing
        # obstacle that happens to touch the floor row is not relabelled
        # as a curb.
        clusters = []
        min_zones = int(self.cfg["min_cluster_zones"])
        # Step-down cells are not also "closing" cells. A curb and a wall
        # that touch in the image stay two events.
        closing_mask = [flag and not step_mask[i] for i, flag in enumerate(mask)]
        for members in self._components(grid, closing_mask):
            if len(members) < min_zones:
                continue
            described = self._describe(members, grid, closing, step_mask, gait)
            if described is not None:
                clusters.append(described)
        if any(step_mask):
            for members in self._components(grid, step_mask):
                if len(members) < min_zones:
                    continue
                described = self._describe(members, grid, closing, step_mask, gait)
                if described is None:
                    continue
                # _describe already refuses the limb filter for step-down.
                # Force the flag in case the vote was split, and warn when
                # the drop is inside the walking cone.
                described["step_down"] = True
                described["ignored_limb"] = False
                on_path = described["in_cone"] or not self.use_cone
                described["warn"] = bool(on_path)
                clusters.append(described)
        self.history.append((now, grid))
        # Keep a little more than the baseline so a late frame still finds
        # a sample 0.2 s back. Drop anything older than the stall gap.
        keep = float(self.cfg["max_baseline_gap_s"])
        self.history = [item for item in self.history if now - item[0] <= keep]
        warning = None
        for cluster in clusters:
            if not cluster["warn"]:
                continue
            if warning is None:
                warning = cluster
                continue
            # Prefer a step-down, then the smallest TTC, then the nearest.
            if cluster["step_down"] and not warning["step_down"]:
                warning = cluster
            elif cluster["step_down"] == warning["step_down"]:
                a = cluster["ttc_s"] if cluster["ttc_s"] is not None else 1e9
                b = warning["ttc_s"] if warning["ttc_s"] is not None else 1e9
                if a < b or (a == b and cluster["depth_mm"] < warning["depth_mm"]):
                    warning = cluster
        valid = sum(1 for value in grid if value is not None)
        return {
            "gait": gait,
            "dt_s": None if dt is None else round(dt, 4),
            "valid_zones": valid,
            "clusters": clusters,
            "warning": None if warning is None else {
                "kind": "step_down" if warning["step_down"] else "ttc",
                "azimuth_deg": warning["azimuth_deg"],
                "elevation_deg": warning["elevation_deg"],
                "depth_mm": warning["depth_mm"],
                "closing_mm_s": warning["closing_mm_s"],
                "ttc_s": warning["ttc_s"],
            },
        }
