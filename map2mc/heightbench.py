"""Synthetic benchmark for the height estimation (no height map available).

Builds a deterministic island with a *known* height field, paints it like a
top-down game map (class colours, hill shading from the north-west, dark tree
dots, bright house roofs, a river and a town) and measures how well
terrain.compute_heights() recovers the true heights from the picture alone.

    python -m map2mc.heightbench          # prints the scores

Scores (computed on land only):
  corr      Pearson correlation between estimate and truth (1 = perfect shape)
  rmse      error in blocks after the best linear fit of the estimate to the truth
  streaks   share of relief energy in thin diagonal stripes (artefacts, lower = better)
  pits      share of tree-dot pixels sitting in a hole > 1.5 blocks deep (lower = better)
  river     how much an island-crossing river lowers the land 20 px away from it,
            relative to the land height (lower = better)
"""
from __future__ import annotations

import numpy as np

from .analysis import Structure
from .config import CID, Settings

COL = {
    "water_deep": (35, 95, 114), "water_shallow": (68, 164, 159), "sand": (243, 216, 176),
    "grass": (121, 144, 59), "forest": (86, 110, 49), "rock": (120, 112, 95),
    "urban": (166, 162, 161), "road": (90, 90, 95), "path": (149, 133, 85),
}


def _hills(H, W, rng, n, rmin, rmax, amp):
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    out = np.zeros((H, W), np.float32)
    for _ in range(n):
        cx, cy = rng.uniform(0.2 * W, 0.8 * W), rng.uniform(0.2 * H, 0.8 * H)
        r = rng.uniform(rmin, rmax)
        out += amp * rng.uniform(0.4, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    return out


def make_island(size: int = 320, seed: int = 7, shading: float = 1.0):
    """Return (img uint8 HxWx3, Structure, true_height float32 blocks, info dict)."""
    import cv2
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    c = size / 2
    r = np.hypot(xx - c, yy - c) / (0.40 * size)
    ang = np.arctan2(yy - c, xx - c)
    wobble = 0.08 * np.sin(3 * ang + 0.5) + 0.05 * np.sin(5 * ang + 1.3)
    shape = 1.0 - (r / (1 + wobble)) ** 2                 # >0 on the island
    land = shape > 0
    # true height: dome + hills + one mountain massif, 0 at the coast
    truth = 30 * np.clip(shape, 0, 1) ** 0.7
    truth += _hills(H, W, rng, 9, size * 0.03, size * 0.08, 14) * land
    mx, my = 0.62 * size, 0.40 * size
    mount = 45 * np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * (0.09 * size) ** 2))
    truth += mount
    # river: a valley across the island (south-west to east)
    t = np.linspace(0, 1, 400)
    rx = 0.10 * size + 0.80 * size * t
    ry = 0.70 * size - 0.25 * size * t + 0.04 * size * np.sin(6 * t)
    pts = np.stack([rx, ry], 1).astype(np.int32)
    river_line = np.zeros((H, W), np.uint8)
    cv2.polylines(river_line, [pts], False, 1, 1)
    d_river = cv2.distanceTransform((1 - river_line).astype(np.uint8), cv2.DIST_L2, 5)
    river = (d_river <= 2.5) & land
    truth = truth - 6 * np.exp(-(d_river / 10.0) ** 2) * land
    truth = np.where(land, np.maximum(truth, 1.0), -8.0).astype(np.float32)
    truth = np.where(river, 0.0, truth).astype(np.float32)

    # classes
    cls = np.full((H, W), CID["water_deep"], np.uint8)
    cls[(shape > -0.25)] = CID["water_shallow"]
    cls[land] = CID["grass"]
    cls[land & (shape < 0.10)] = CID["sand"]
    forest = (_hills(H, W, np.random.default_rng(seed + 1), 6, size * 0.04, size * 0.07, 1) > 0.5) & land
    cls[forest & (cls == CID["grass"])] = CID["forest"]
    cls[land & (mount > 22)] = CID["rock"]
    # town plateau in the south-east with roads
    tx, ty, tr = int(0.66 * size), int(0.70 * size), int(0.07 * size)
    town = (np.hypot(xx - tx, yy - ty) < tr) & land & ~river
    cls[town] = CID["urban"]
    roads = np.zeros((H, W), np.uint8)
    cv2.line(roads, (tx - tr, ty), (tx + tr, ty), 1, 2)
    cv2.line(roads, (tx, ty - tr), (tx, ty + tr), 1, 2)
    cls[(roads > 0) & town] = CID["road"]
    # a dirt path through the countryside
    path = np.zeros((H, W), np.uint8)
    cv2.line(path, (int(0.25 * size), int(0.35 * size)), (tx - tr, ty), 1, 2)
    cls[(path > 0) & land & ~town & ~river] = CID["path"]
    cls[river] = CID["water_shallow"]
    lake_id = np.zeros((H, W), np.int32)
    lake_id[river] = 1
    # the town is flat in reality, too
    truth = np.where(town, float(np.median(truth[town])), truth).astype(np.float32)

    # picture: albedo * hill shading (light from the north-west) + map icons
    img = np.zeros((H, W, 3), np.float32)
    for k, v in COL.items():
        img[cls == CID[k]] = v
    gy, gx = np.gradient(cv2.GaussianBlur(truth, (0, 0), 1.0))
    lx, ly, lz = -1.0, -1.0, 1.6
    n = np.sqrt(gx * gx + gy * gy + 1.0)
    shade = (-gx * lx - gy * ly + lz) / (n * np.sqrt(lx * lx + ly * ly + lz * lz))
    shade = 1.0 + shading * (shade - float(np.median(shade[land]))) * 0.7
    img = img * np.where(land, shade, 1.0)[..., None]
    # tree dots (dark crowns) in forests and here and there on grass
    trees = []
    for _ in range(int(size * size * 0.004)):
        u, v = int(rng.integers(0, W)), int(rng.integers(0, H))
        k = cls[v, u]
        if k == CID["forest"] or (k == CID["grass"] and rng.random() < 0.3):
            rad = int(rng.integers(2, 4))
            cv2.circle(img, (u, v), rad, (40, 62, 28), -1)
            trees.append((u, v))
    # house roofs in town
    for _ in range(25):
        u, v = int(rng.integers(tx - tr, tx + tr)), int(rng.integers(ty - tr, ty + tr))
        if town[v, u] and cls[v, u] == CID["urban"]:
            cv2.rectangle(img, (u - 3, v - 2), (u + 3, v + 2), (180, 70, 70), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ocean = (~land) & (lake_id == 0)
    st = Structure(classes=cls, bridge=np.zeros((H, W), bool), ocean=ocean, lake_id=lake_id, land=~(ocean | river))
    info = {"trees": np.array(trees, np.int32).reshape(-1, 2), "river": river, "d_river": d_river, "town": town}
    return img, st, truth, info


def score(est: np.ndarray, truth: np.ndarray, st: Structure, info: dict) -> dict:
    import cv2
    land = st.land
    e, t = est[land].astype(np.float64), truth[land].astype(np.float64)
    corr = float(np.corrcoef(e, t)[0, 1])
    A = np.stack([e, np.ones_like(e)], 1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ coef - t) ** 2)))
    # diagonal streaks: high-frequency energy along the NE-SW direction of the residual
    res = np.where(land, est - cv2.GaussianBlur(est, (0, 0), 6), 0).astype(np.float32)
    F = np.abs(np.fft.fft2(res)) ** 2
    H, W = res.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    k = np.hypot(kx, ky) + 1e-9
    along = np.abs(kx - ky) / np.sqrt(2) / k            # 1 = wave vector along NW-SE (stripes NE-SW)
    band = (k > 0.03) & (k < 0.25)
    streaks = float(F[band & (along > 0.95)].sum() / max(F[band].sum(), 1e-9))
    # pits under tree dots
    pits = 0.0
    tr = info["trees"]
    if len(tr):
        local = cv2.GaussianBlur(est, (0, 0), 5)
        deep = (local - est)[tr[:, 1], tr[:, 0]] > 1.5
        on_land = land[tr[:, 1], tr[:, 0]]
        pits = float(deep[on_land].mean()) if on_land.any() else 0.0
    # river: land 15-25 px from the river vs. land 40-60 px away, relative to truth
    d = info["d_river"]
    near = land & (d > 15) & (d < 25)
    far = land & (d > 40) & (d < 60)
    sc = coef[0]
    drop_est = (np.median(est[far]) - np.median(est[near])) * sc
    drop_true = np.median(truth[far]) - np.median(truth[near])
    river = float(max(0.0, drop_est - drop_true) / max(np.median(truth[land]), 1.0))
    return {"corr": corr, "rmse": rmse, "streaks": streaks, "pits": pits, "river": river}


def run(seeds=(7, 11, 23), size=320, compute=None):
    from . import terrain
    compute = compute or terrain.compute_heights
    cfg = Settings()
    cfg.blocks_per_pixel = 3.0
    rows = []
    for s in seeds:
        img, st, truth, info = make_island(size, s)
        h, _ = compute(img, st, cfg)
        rows.append(score(h, truth, st, info))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:8s} {v:.3f}")
