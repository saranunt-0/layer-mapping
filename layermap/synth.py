"""Procedural painterly landscapes with exact paint-order layer labels.

Each scene is painted back to front, the way a landscape painter blocks in a
picture: sky, clouds, far hills, water, mid-ground, trees and buildings,
foreground, flowers. Every brush stroke also writes its layer number into an
ID buffer, so the label at a pixel is the index of the last layer painted
there (0 = painted first = furthest back).

Each layer also carries simple scene geometry, so `render(..., return_depth=True)`
returns a continuous disparity map (0 = sky at infinity, 1 = bottom of a flat
ground plane): plains and roads recede smoothly, hills rise steeper than the
plane behind their ridge, and trees, houses and figures stand at the depth of
their base. Paint order and physical depth therefore agree at most boundaries
but not everywhere (a tree in the middle distance is painted over a meadow
that extends in front of it), just as in real paintings.

Everything is generated from scratch, so the data carries no third-party
license or copyright.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Bump when the generator's output changes, so cached datasets are not reused by mistake.
GENERATOR_VERSION = 2
# Minimum RGB distance (0..sqrt(3)) between a layer's colour and the layer painted before it.
MIN_LAYER_CONTRAST = 0.12


@dataclass
class SynthConfig:
    size: int = 576
    # Fraction of images rendered as flat, crisp fills (vector-illustration look).
    flat_prob: float = 0.15
    # Painted stroke area as a multiple of each layer's area.
    stroke_coverage: tuple[float, float] = (0.7, 1.8)
    # Stroke length and width as a fraction of the image size.
    stroke_len: tuple[float, float] = (0.008, 0.045)
    stroke_width: tuple[float, float] = (0.004, 0.014)
    # Probability that a stroke takes a shifted hue ("broken color").
    broken_color_prob: tuple[float, float] = (0.03, 0.2)
    max_strokes_per_layer: int = 25000
    jpeg_quality: tuple[int, int] = (55, 95)


@dataclass
class Layer:
    mask: np.ndarray  # bool (H, W)
    kind: str
    top_hsv: np.ndarray  # colour at the top of the layer's extent
    bottom_hsv: np.ndarray  # colour at the bottom of the layer's extent
    angle: float  # main stroke direction, degrees
    angle_jitter: float
    depth: np.ndarray  # float32 (H, W) disparity of this layer's surface, defined everywhere


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


def _noise_1d(rng: np.random.Generator, n: int, n_ctrl: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] sampled at n points."""
    ctrl = rng.uniform(-1.0, 1.0, n_ctrl)
    xs = np.linspace(0, n_ctrl - 1, n)
    i0 = np.clip(np.floor(xs).astype(int), 0, n_ctrl - 2)
    t = xs - i0
    t = t * t * (3 - 2 * t)
    return ctrl[i0] * (1 - t) + ctrl[i0 + 1] * t


def _ridge(rng, width, y_mean, amp, octaves=5, pointy=False, slope=0.0):
    """Height line y(x) of a hill (rounded) or mountain range (pointy peaks)."""
    y = slope * (np.arange(width) - width / 2)
    for k in range(octaves):
        n = _noise_1d(rng, width, 2 ** (k + 1) + 1)
        if pointy:
            # |noise| has sharp minima, which become sharp peaks (y grows downward).
            n = np.abs(n) * 2 - 1
        y += n * amp / (2**k)
    return y_mean + y


def _below(line: np.ndarray, h: int) -> np.ndarray:
    return np.arange(h)[:, None] >= line[None, :]


def _tree_mask(rng, h, w, x, base_y, scale, style):
    m = np.zeros((h, w), np.uint8)
    trunk_w = max(1, int(scale * 0.08))
    if style == "poplar":
        crown_h, crown_w = scale * 2.2, scale * 0.35
    elif style == "pine":
        crown_h, crown_w = scale * 1.5, scale * 0.6
    else:
        crown_h, crown_w = scale * 1.0, scale * 0.9
    crown_bottom = base_y - scale * 0.25
    cv2.rectangle(m, (int(x - trunk_w), int(crown_bottom - 2)), (int(x + trunk_w), int(base_y)), 1, -1)
    cy = crown_bottom - crown_h / 2
    if style == "pine":
        pts = np.array([[x, crown_bottom - crown_h], [x - crown_w / 2, crown_bottom], [x + crown_w / 2, crown_bottom]])
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    else:
        for _ in range(rng.integers(3, 8)):
            ox = rng.normal(0, crown_w * 0.2)
            oy = rng.normal(0, crown_h * 0.2)
            ax = max(2, int(crown_w * rng.uniform(0.25, 0.5)))
            ay = max(2, int(crown_h * rng.uniform(0.25, 0.5)))
            cv2.ellipse(m, (int(x + ox), int(cy + oy)), (ax, ay), rng.uniform(-20, 20), 0, 360, 1, -1)
    return m.astype(bool)


def _house_mask(h, w, x, base_y, scale, rng):
    m = np.zeros((h, w), np.uint8)
    bw, bh = scale * rng.uniform(0.8, 1.6), scale * rng.uniform(0.5, 0.9)
    x0, x1 = x - bw / 2, x + bw / 2
    top = base_y - bh
    cv2.rectangle(m, (int(x0), int(top)), (int(x1), int(base_y)), 1, -1)
    roof = np.array([[x0 - scale * 0.1, top], [x1 + scale * 0.1, top], [x, top - scale * rng.uniform(0.3, 0.6)]])
    cv2.fillPoly(m, [roof.astype(np.int32)], 1)
    return m.astype(bool)


def _figure_mask(h, w, x, base_y, scale):
    m = np.zeros((h, w), np.uint8)
    body = np.array([[x - scale * 0.15, base_y - scale * 0.75], [x + scale * 0.15, base_y - scale * 0.75],
                     [x + scale * 0.25, base_y], [x - scale * 0.25, base_y]])
    cv2.fillPoly(m, [body.astype(np.int32)], 1)
    cv2.circle(m, (int(x), int(base_y - scale * 0.85)), max(1, int(scale * 0.12)), 1, -1)
    return m.astype(bool)


def _boat_mask(h, w, x, base_y, scale):
    m = np.zeros((h, w), np.uint8)
    hull = np.array([[x - scale * 0.6, base_y - scale * 0.15], [x + scale * 0.6, base_y - scale * 0.15],
                     [x + scale * 0.4, base_y], [x - scale * 0.4, base_y]])
    sail = np.array([[x, base_y - scale * 1.2], [x, base_y - scale * 0.2], [x + scale * 0.5, base_y - scale * 0.2]])
    cv2.fillPoly(m, [hull.astype(np.int32), sail.astype(np.int32)], 1)
    return m.astype(bool)


def _cloud_mask(rng, h, w, horizon):
    m = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(1, 5)):
        cx, cy = rng.uniform(0, w), rng.uniform(0.05 * h, max(0.1 * h, horizon * 0.7))
        cw, ch = rng.uniform(0.08, 0.3) * w, rng.uniform(0.03, 0.08) * h
        for _ in range(rng.integers(3, 9)):
            cv2.ellipse(m, (int(cx + rng.normal(0, cw * 0.4)), int(cy + rng.normal(0, ch * 0.3))),
                        (max(2, int(cw * rng.uniform(0.2, 0.5))), max(2, int(ch * rng.uniform(0.4, 1.0)))),
                        0, 0, 360, 1, -1)
    return m.astype(bool)


def _dab_mask(rng, h, w, y0, density):
    m = np.zeros((h, w), np.uint8)
    n = int(density * w * (h - y0) / 400)
    ys = rng.uniform(y0, h, n)
    xs = rng.uniform(0, w, n)
    r = np.maximum(1, (ys - y0) / max(1, h - y0) * 0.012 * w + 1).astype(int)
    for x, y, rr in zip(xs.astype(int), ys.astype(int), r):
        cv2.circle(m, (x, y), int(rr), 1, -1)
    return m.astype(bool)


def _road_mask(rng, h, w, y_v):
    """A road running from the bottom edge to a vanishing point at row y_v, with a gentle bend."""
    m = np.zeros((h, w), np.uint8)
    ys = np.linspace(y_v, h + 2, 48)
    t = (ys - y_v) / max(1.0, h - y_v)
    vx = rng.uniform(0.3, 0.7) * w
    bottom_x = vx + rng.uniform(-0.3, 0.3) * w
    centre = vx + (bottom_x - vx) * t + rng.uniform(-0.12, 0.12) * w * np.sin(np.pi * t)
    half = rng.uniform(0.12, 0.3) * w * t + 1
    pts = np.concatenate([np.stack([centre - half, ys], 1), np.stack([centre + half, ys], 1)[::-1]])
    cv2.fillPoly(m, [pts.round().astype(np.int32)], 1)
    return m.astype(bool)


# --------------------------------------------------------------------------- #
# Geometry (disparity: larger = closer, 0 = infinitely far)
# --------------------------------------------------------------------------- #


def _ground_curve(h: int, horizon: float) -> np.ndarray:
    """Disparity of a flat ground plane for each image row: 0.02 at the horizon, 1 at the bottom row."""
    y = np.arange(h, dtype=np.float32)
    below = np.clip((y - horizon) / max(1.0, h - horizon), 0, 1)
    above = np.clip((horizon - y) / max(1.0, horizon), 0, 1)
    return (0.02 + 0.98 * below - 0.01 * above).astype(np.float32)


def _terrain_depth(ground: np.ndarray, ridge: np.ndarray, slope: float) -> np.ndarray:
    """Terrain whose top edge is `ridge`: it starts at the ground-plane disparity of the ridge and grows
    `slope` times as fast as the plane below it (1 = flat plain, < 1 = hill rising toward the ridge)."""
    g_ridge = np.interp(np.clip(ridge, 0, len(ground) - 1), np.arange(len(ground)), ground)[None, :]
    return (g_ridge + slope * (ground[:, None] - g_ridge)).astype(np.float32)


def _object_depth(parts: list[tuple[np.ndarray, float]], h: int, w: int) -> np.ndarray:
    """Constant disparity per object, extended to nearby pixels so stroke spill inherits it."""
    d = np.full((h, w), np.nan, np.float32)
    for mask, value in parts:
        d[mask] = value
    holes = np.isnan(d)
    if holes.all():
        return np.zeros((h, w), np.float32)
    if holes.any():
        from scipy import ndimage

        idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        d = d[tuple(idx)]
    return d


def _depth_at(depth: np.ndarray, x: float, y: float) -> float:
    h, w = depth.shape
    return float(depth[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))])


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #


def _hsv(h, s, v):
    return np.array([h % 360.0, np.clip(s, 0, 1), np.clip(v, 0, 1)], np.float32)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """(..., 3) HSV with H in degrees -> float RGB in [0, 1]."""
    shp = hsv.shape
    flat = np.ascontiguousarray(hsv.reshape(-1, 1, 3).astype(np.float32))
    flat[..., 0] %= 360.0
    return cv2.cvtColor(flat, cv2.COLOR_HSV2RGB).reshape(shp)


def _haze(hsv, sky_hsv, amount):
    """Atmospheric perspective: pull a colour toward the horizon sky colour."""
    rgb = _hsv_to_rgb(hsv)
    sky = _hsv_to_rgb(sky_hsv)
    out = rgb * (1 - amount) + sky * amount
    return cv2.cvtColor(out.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(3)


# --------------------------------------------------------------------------- #
# Scene composition (back to front)
# --------------------------------------------------------------------------- #


def _compose(rng: np.random.Generator, h: int, w: int) -> list[Layer]:
    layers: list[Layer] = []
    mood = rng.choice(["day", "sunset", "overcast"], p=[0.55, 0.25, 0.2])
    if mood == "day":
        zenith = _hsv(rng.uniform(195, 230), rng.uniform(0.35, 0.7), rng.uniform(0.7, 0.95))
        horizon_c = _hsv(zenith[0] + rng.uniform(-15, 15), zenith[1] * 0.4, min(1.0, zenith[2] + 0.1))
    elif mood == "sunset":
        zenith = _hsv(rng.uniform(220, 280), rng.uniform(0.2, 0.45), rng.uniform(0.55, 0.85))
        horizon_c = _hsv(rng.uniform(15, 50), rng.uniform(0.5, 0.9), rng.uniform(0.85, 1.0))
    else:
        zenith = _hsv(rng.uniform(190, 230), rng.uniform(0.05, 0.2), rng.uniform(0.55, 0.85))
        horizon_c = _hsv(zenith[0], zenith[1] * 0.5, min(1.0, zenith[2] + 0.1))
    warm = mood == "sunset"

    def add(mask, kind, top, bottom, angle, jitter, depth):
        if not mask.any():
            return
        if layers:
            # A layer that is nearly the colour of the one before it is invisible, so its label would be
            # noise. Painters separate planes by value; do the same.
            prev = _hsv_to_rgb(layers[-1].bottom_hsv)
            if np.linalg.norm(_hsv_to_rgb(top) - prev) < MIN_LAYER_CONTRAST:
                shift = -0.18 if top[2] > 0.35 else 0.18
                top = _hsv(top[0], top[1], top[2] + shift)
                bottom = _hsv(bottom[0], bottom[1], bottom[2] + shift)
        depth = np.broadcast_to(np.asarray(depth, np.float32), (h, w))
        layers.append(Layer(mask, kind, top, bottom, angle, jitter, depth))

    horizon = rng.uniform(0.3, 0.65) * h
    ground = _ground_curve(h, horizon)
    plane = np.repeat(ground[:, None], w, axis=1)
    # "Open plain" scenes: no mid-ground hills, flat foreground reaching (almost) to the horizon.
    open_plain = rng.random() < 0.25
    add(np.ones((h, w), bool), "sky", zenith, horizon_c, 0.0, 15.0, 0.0)

    if mood != "overcast" or rng.random() < 0.5:
        for _ in range(rng.integers(0, 3)):
            c = _hsv(rng.uniform(20, 60) if warm else rng.uniform(200, 260), rng.uniform(0.02, 0.25),
                     rng.uniform(0.85, 1.0))
            add(_cloud_mask(rng, h, w, horizon), "cloud", c, _hsv(c[0], c[1] + 0.1, c[2] - 0.15), 0.0, 30.0,
                0.001)

    # Distant ranges: each one a separate layer, hazier the further back.
    n_far = rng.integers(0, 3)
    for i in range(n_far):
        dist = 1.0 - i / max(1, n_far)
        y = _ridge(rng, w, horizon - rng.uniform(0.0, 0.15) * h * dist, rng.uniform(0.03, 0.12) * h,
                   pointy=rng.random() < 0.6)
        base = _hsv(rng.uniform(200, 260) if rng.random() < 0.6 else rng.uniform(80, 150),
                    rng.uniform(0.2, 0.5), rng.uniform(0.35, 0.7))
        c = _haze(base, horizon_c, 0.35 + 0.4 * dist)
        add(_below(y, h), "mountain", c, _hsv(c[0], c[1], c[2] - 0.08), rng.uniform(-30, 30), 25.0,
            0.002 + 0.006 * (i + 1) / (n_far + 1))

    has_water = rng.random() < 0.45
    if has_water:
        y = _ridge(rng, w, horizon + rng.uniform(0.0, 0.05) * h, 0.004 * h, octaves=2)
        c = _hsv(horizon_c[0] + rng.uniform(-10, 10), horizon_c[1] + 0.1, horizon_c[2] - 0.1)
        add(_below(y, h), "water", c, _hsv(zenith[0], zenith[1] + 0.1, zenith[2] - 0.25), 0.0, 5.0, plane)
        if rng.random() < 0.5:
            parts = []
            for _ in range(rng.integers(1, 4)):
                base_y = horizon + rng.uniform(0.04, 0.15) * h
                bm = _boat_mask(h, w, rng.uniform(0, w), base_y, rng.uniform(0.03, 0.08) * h)
                parts.append((bm, ground[int(min(base_y, h - 1))]))
            m = np.any([p[0] for p in parts], axis=0)
            c = _hsv(rng.uniform(0, 360), rng.uniform(0.2, 0.7), rng.uniform(0.3, 0.95))
            add(m, "boat", c, _hsv(c[0], c[1], c[2] - 0.2), 90.0, 40.0, _object_depth(parts, h, w))

    # Mid ground: fields/hills, optionally with a row of trees or houses on top.
    ground_top = horizon + (rng.uniform(0.12, 0.25) * h if has_water else rng.uniform(-0.02, 0.08) * h)
    n_mid = 0 if open_plain else rng.integers(1, 3)
    for i in range(n_mid):
        t = (i + 1) / (n_mid + 1)
        y_mean = ground_top + t * (h - ground_top) * 0.4
        y = _ridge(rng, w, y_mean, rng.uniform(0.01, 0.05) * h, slope=rng.normal(0, 0.08))
        field_depth = _terrain_depth(ground, y, 1.0 if rng.random() < 0.4 else rng.uniform(0.4, 0.85))
        g = _hsv(rng.uniform(55, 140), rng.uniform(0.3, 0.7), rng.uniform(0.4, 0.8))
        g = _haze(g, horizon_c, 0.25 * (1 - t))
        add(_below(y, h), "field", g, _hsv(g[0] + rng.uniform(-15, 15), g[1], g[2] - 0.1),
            rng.choice([0.0, 90.0]), 25.0, field_depth)
        if rng.random() < 0.6:
            scale = (y_mean / h) * rng.uniform(0.08, 0.2) * h
            parts = []
            kind = rng.choice(["tree", "house"], p=[0.75, 0.25])
            style = rng.choice(["round", "poplar", "pine"])
            for x in rng.uniform(0, w, rng.integers(1, 7)):
                base_y = float(np.interp(x, np.arange(w), y)) + scale * 0.1
                om = _tree_mask(rng, h, w, x, base_y, scale, style) if kind == "tree" else \
                    _house_mask(h, w, x, base_y, scale, rng)
                parts.append((om, _depth_at(field_depth, x, base_y)))
            m = np.any([p[0] for p in parts], axis=0)
            od = _object_depth(parts, h, w)
            if kind == "tree":
                c = _hsv(rng.uniform(70, 160), rng.uniform(0.35, 0.8), rng.uniform(0.2, 0.55))
                add(m, "tree", _hsv(c[0] - 15, c[1], c[2] + 0.2), c, rng.uniform(0, 180), 90.0, od)
            else:
                c = _hsv(rng.uniform(20, 50), rng.uniform(0.1, 0.5), rng.uniform(0.7, 0.95))
                add(m, "house", _hsv(rng.uniform(0, 20), 0.6, 0.6), c, 90.0, 10.0, od)

    # Foreground ground, then objects standing on it, then flowers.
    if open_plain:
        y_fg = _ridge(rng, w, horizon + rng.uniform(0.0, 0.05) * h, 0.005 * h, octaves=3)
        fg_depth = _terrain_depth(ground, y_fg, 1.0)
    else:
        y_fg = _ridge(rng, w, ground_top + (h - ground_top) * rng.uniform(0.5, 0.75), rng.uniform(0.02, 0.06) * h,
                      slope=rng.normal(0, 0.12))
        fg_depth = _terrain_depth(ground, y_fg, 1.0 if rng.random() < 0.4 else rng.uniform(0.4, 0.85))
    fg_mask = _below(y_fg, h)
    g = _hsv(rng.uniform(40, 130), rng.uniform(0.4, 0.8), rng.uniform(0.35, 0.75))
    add(fg_mask, "ground", g, _hsv(g[0] + rng.uniform(-20, 20), g[1], g[2] - 0.15), 90.0, 30.0, fg_depth)
    # A road receding over the foreground to its crest (or to the horizon on an open plain).
    road = np.zeros((h, w), bool)
    if rng.random() < (0.7 if open_plain else 0.35):
        road = _road_mask(rng, h, w, float(np.min(y_fg)) + 0.01 * h) & fg_mask
        c = _hsv(rng.uniform(25, 45), rng.uniform(0.15, 0.45), rng.uniform(0.6, 0.9))
        add(road, "road", _haze(c, horizon_c, 0.3), c, 90.0, 20.0, fg_depth + 0.001)
    # Tall foreground objects overlap everything behind them; they are the main source of
    # "upper pixel in front" pairs that the vertical prior gets wrong.
    if rng.random() < 0.65:
        parts = []
        kind = rng.choice(["tree", "figure"], p=[0.6, 0.4])
        for x in rng.uniform(0, w, rng.integers(1, 3 if kind == "tree" else 5)):
            base_y = float(np.interp(x, np.arange(w), y_fg)) + rng.uniform(0.02, 0.1) * h
            if kind == "tree":
                om = _tree_mask(rng, h, w, x, base_y, rng.uniform(0.25, 0.5) * h,
                                rng.choice(["round", "poplar", "pine"]))
            else:
                om = _figure_mask(h, w, x, base_y, rng.uniform(0.1, 0.25) * h)
            parts.append((om, _depth_at(fg_depth, x, base_y)))
        m = np.any([p[0] for p in parts], axis=0)
        c = _hsv(rng.uniform(0, 360) if kind == "figure" else rng.uniform(70, 160), rng.uniform(0.3, 0.8),
                 rng.uniform(0.2, 0.6))
        add(m, kind, _hsv(c[0], c[1], c[2] + 0.15), c, 90.0 if kind == "figure" else rng.uniform(0, 180), 60.0,
            _object_depth(parts, h, w))
    if rng.random() < 0.5:
        c = _hsv(rng.choice([0, 45, 280, 330, 55]) + rng.uniform(-10, 10), rng.uniform(0.6, 1.0),
                 rng.uniform(0.75, 1.0))
        y0 = int(np.clip(np.median(y_fg), 0, h - 2))
        add(_dab_mask(rng, h, w, y0, rng.uniform(0.5, 3.0)) & ~road, "flowers", c, c, 0.0, 180.0, fg_depth + 0.002)

    return layers


# --------------------------------------------------------------------------- #
# Painting
# --------------------------------------------------------------------------- #


def _layer_colours(layer: Layer, ys: np.ndarray, y_top: float, y_bot: float) -> np.ndarray:
    t = np.clip((ys - y_top) / max(1.0, y_bot - y_top), 0, 1)[:, None]
    # Interpolate hue along the shortest arc.
    dh = (layer.bottom_hsv[0] - layer.top_hsv[0] + 180) % 360 - 180
    hsv = layer.top_hsv[None, :] + t * np.array([dh, *(layer.bottom_hsv[1:] - layer.top_hsv[1:])])[None, :]
    return hsv.astype(np.float32)


def _paint_layer(rng, img, ids, layer: Layer, k: int, cfg: SynthConfig, flat: bool, broken_p: float):
    h, w = ids.shape
    ys, xs = np.nonzero(layer.mask)
    y_top, y_bot = float(ys.min()), float(ys.max())

    # Underpainting: flat fill with the layer's vertical gradient plus low-frequency mottling.
    row_hsv = _layer_colours(layer, np.arange(h, dtype=np.float32), y_top, y_bot)
    row_rgb = _hsv_to_rgb(row_hsv)
    fill = row_rgb[ys]
    if not flat:
        mottle = cv2.resize(rng.normal(0, 0.04, (8, 8, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        fill = fill + mottle[ys, xs]
    img[ys, xs] = (np.clip(fill, 0, 1) * 255).astype(np.uint8)
    ids[ys, xs] = k
    if flat:
        return

    s = cfg.size
    length = rng.uniform(*cfg.stroke_len) * s
    width = rng.uniform(*cfg.stroke_width) * s
    n = int(rng.uniform(*cfg.stroke_coverage) * len(ys) / max(1.0, length * width))
    n = min(n, cfg.max_strokes_per_layer)
    if n == 0:
        return
    pick = rng.integers(0, len(ys), n)
    py, px = ys[pick].astype(np.float32), xs[pick].astype(np.float32)

    hsv = _layer_colours(layer, py, y_top, y_bot)
    hsv[:, 0] += rng.normal(0, 6, n)
    hsv[:, 1] += rng.normal(0, 0.08, n)
    hsv[:, 2] += rng.normal(0, 0.08, n)
    broken = rng.random(n) < broken_p
    hsv[broken, 0] += rng.choice([-1, 1], broken.sum()) * rng.uniform(15, 45, broken.sum())
    hsv[:, 1:] = np.clip(hsv[:, 1:], 0, 1)
    rgb = (_hsv_to_rgb(hsv) * 255).astype(np.uint8)

    ang = np.deg2rad(layer.angle + rng.normal(0, layer.angle_jitter, n))
    ln = length * rng.uniform(0.4, 1.3, n) / 2
    dx, dy = np.cos(ang) * ln, np.sin(ang) * ln
    p1 = np.stack([px - dx, py - dy], 1).round().astype(np.int32)
    p2 = np.stack([px + dx, py + dy], 1).round().astype(np.int32)
    th = np.maximum(1, (width * rng.uniform(0.6, 1.4, n)).round()).astype(int)

    for a, b, c, t in zip(p1.tolist(), p2.tolist(), rgb.tolist(), th.tolist()):
        cv2.line(img, a, b, c, t, cv2.LINE_AA)
        cv2.line(ids, a, b, k, t, cv2.LINE_8)


def _canvas_finish(rng, img8: np.ndarray, cfg: SynthConfig, flat: bool) -> np.ndarray:
    h, w = img8.shape[:2]
    out = img8.astype(np.float32) / 255
    if not flat:
        # Canvas weave + grain.
        f = rng.uniform(0.25, 0.6)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        weave = np.sin(xx * f) * np.sin(yy * f) * rng.uniform(0.0, 0.03)
        grain = cv2.GaussianBlur(rng.normal(0, rng.uniform(0.0, 0.04), (h, w)).astype(np.float32), (0, 0), 0.8)
        out = out * (1 + weave + grain)[..., None]
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.2))
    out = np.clip(out * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
    out8 = (out * 255).astype(np.uint8)
    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    ok, buf = cv2.imencode(".jpg", out8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy() if ok else out8


def render(seed: int, cfg: SynthConfig | None = None, return_depth: bool = False):
    """Render one scene.

    Returns:
        rgb: uint8 (H, W, 3).
        labels: uint8 (H, W) dense layer index, 0 = back (painted first).
        depth (only with return_depth): float32 (H, W) disparity of the visible surface,
            0 = infinitely far, 1 = bottom row of a flat ground plane.
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    h = w = cfg.size
    layers = _compose(rng, h, w)
    flat = rng.random() < cfg.flat_prob
    broken_p = rng.uniform(*cfg.broken_color_prob)

    img = np.zeros((h, w, 3), np.uint8)
    ids = np.zeros((h, w), np.uint8)
    for k, layer in enumerate(layers):
        _paint_layer(rng, img, ids, layer, k, cfg, flat, broken_p)

    rgb = _canvas_finish(rng, img, cfg, flat)
    _, labels = np.unique(ids, return_inverse=True)
    labels = labels.reshape(h, w).astype(np.uint8)
    if not return_depth:
        return rgb, labels
    depth = np.zeros((h, w), np.float32)
    for k, layer in enumerate(layers):
        sel = ids == k
        depth[sel] = layer.depth[sel]
    return rgb, labels, depth


def scene_kinds(seed: int, cfg: SynthConfig | None = None) -> list[str]:
    """Kinds of the layers `render(seed, cfg)` paints, back to front (e.g. to pick scenes with a road)."""
    cfg = cfg or SynthConfig()
    return [layer.kind for layer in _compose(np.random.default_rng(seed), cfg.size, cfg.size)]
