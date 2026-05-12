#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Path collection
# -------------------------

def collect_real_raw(real_dir: Path) -> List[Path]:
    # ONLY metal*.raw, float32 512x512
    paths = sorted(real_dir.rglob("metal*.raw"))
    good = []
    for p in paths:
        if p.stat().st_size == 512 * 512 * 4:
            good.append(p)
    return good


def collect_aapm_png(aapm_dir: Path) -> List[Path]:
    return sorted(aapm_dir.glob("body_img*.png"))


def collect_ours_png_by_imgid(ours_dir: Path) -> List[Path]:
    """
    OURS (new): body_img<id>_gen.png
    We parse <id> and sort by it to keep deterministic ordering and easy debugging.
    """
    pat = re.compile(r"body_img(\d+)_gen\.png$")
    items: List[Tuple[int, Path]] = []
    for p in ours_dir.glob("body_img*_gen.png"):
        m = pat.search(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return [p for _, p in items]


# -------------------------
# Sampling
# -------------------------

def sample_paths(paths: List[Path], n: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    return [paths[i] for i in idx[:n]]


# -------------------------
# REAL RAW -> 8-bit (WL/WW)
# -------------------------

def raw_float32_512(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != 512 * 512:
        raise ValueError(f"Unexpected raw size: {path} -> {arr.size} floats")
    return arr.reshape(512, 512)


def hu_to_png8(img_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    wmin = wl - ww / 2.0
    wmax = wl + ww / 2.0
    img_w = np.clip(img_hu, wmin, wmax)
    img_n = (img_w - wmin) / max(1e-6, ww)  # 0..1
    return (img_n * 255.0).astype(np.uint8)


def load_png_gray_u8(path: Path, size=(512, 512)) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read PNG: {path}")
    if img.shape != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


# -------------------------
# VGG embedder
# -------------------------

class VGGEmbedder(nn.Module):
    def __init__(self, device: torch.device, cut_idx: int = 28):
        super().__init__()
        # keep weights loading as it works in your env
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        self.backbone = vgg
        self.cut_idx = cut_idx
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[1] == 1:
            x01 = x01.repeat(1, 3, 1, 1)
        x = (x01 - self.mean) / self.std
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.cut_idx:
                break
        return x.mean(dim=(2, 3))  # GAP -> BxC


# -------------------------
# IO helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def npy_save(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, arr)


def json_save(path: Path, obj: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------
# Embedding extraction + caching
# -------------------------

def extract_embeddings(
    dataset_name: str,
    kind: str,  # "real_raw" | "png"
    paths: List[Path],
    out_npy: Path,
    wl: float,
    ww: float,
    batch_size: int,
    cut_idx: int,
    force: bool,
) -> Path:
    if out_npy.exists() and not force:
        print(f"[SKIP] embeddings exist: {out_npy}")
        return out_npy

    print(f"[RUN ] embeddings: {dataset_name} | N={len(paths)} | kind={kind}")
    embedder = VGGEmbedder(DEVICE, cut_idx=cut_idx)

    embs = []
    bs = batch_size

    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]

        if kind == "real_raw":
            imgs_u8 = []
            for p in chunk:
                hu = raw_float32_512(p)
                imgs_u8.append(hu_to_png8(hu, wl=wl, ww=ww))
            arr_u8 = np.stack(imgs_u8, axis=0)  # BxHxW
        elif kind == "png":
            arr_u8 = np.stack([load_png_gray_u8(p) for p in chunk], axis=0)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        x01 = torch.from_numpy(arr_u8).float().unsqueeze(1).to(DEVICE) / 255.0
        e = embedder(x01).detach().cpu().numpy().astype(np.float32)
        embs.append(e)

    embs = np.concatenate(embs, axis=0)
    npy_save(out_npy, embs)
    print(f"[OK  ] saved: {out_npy} | shape={embs.shape}")
    return out_npy


# -------------------------
# Distances on embeddings
# -------------------------

def _cov(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    return (xc.T @ xc) / max(1, (x.shape[0] - 1))


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Matrix square root for symmetric PSD matrix using eigendecomposition.
    Assumes mat is (numerically) symmetric.
    """
    mat = mat.astype(np.float64)
    mat = 0.5 * (mat + mat.T)  # enforce symmetry
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)) @ v.T


def fid(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """
    Numerically-stable FID on embeddings.

    FID = ||mx-my||^2 + Tr(Cx + Cy - 2 * sqrtm(Cx^{1/2} Cy Cx^{1/2}))
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    cx = _cov(x)
    cy = _cov(y)

    cx = 0.5 * (cx + cx.T)
    cy = 0.5 * (cy + cy.T)
    cx += np.eye(cx.shape[0], dtype=np.float64) * eps
    cy += np.eye(cy.shape[0], dtype=np.float64) * eps

    diff = mx - my

    sqrt_cx = _sqrtm_psd(cx, eps=1e-12)
    prod = sqrt_cx @ cy @ sqrt_cx
    prod = 0.5 * (prod + prod.T)
    covmean = _sqrtm_psd(prod, eps=1e-12)

    val = diff @ diff + np.trace(cx + cy - 2.0 * covmean)
    val = float(np.real(val))
    if val < 0 and val > -1e-6:
        val = 0.0
    return val


def kid(x: np.ndarray, y: np.ndarray, n_subsets: int = 100, subset_size: int = 100, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    n = min(len(x), len(y))
    subset_size = min(subset_size, n)
    if subset_size < 2:
        return float("nan")

    def poly_k(a, b):
        d = a.shape[1]
        return ((a @ b.T) / d + 1.0) ** 3

    mmds = []
    for _ in range(n_subsets):
        ix = rng.choice(len(x), subset_size, replace=False)
        iy = rng.choice(len(y), subset_size, replace=False)
        xs = x[ix]
        ys = y[iy]
        kxx = poly_k(xs, xs)
        kyy = poly_k(ys, ys)
        kxy = poly_k(xs, ys)
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        mmd = (kxx.sum() / (subset_size * (subset_size - 1))
               + kyy.sum() / (subset_size * (subset_size - 1))
               - 2.0 * kxy.mean())
        mmds.append(mmd)
    return float(np.mean(mmds))


# ---- SWD (Sliced Wasserstein) ----

def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    return float(np.mean(np.abs(a - b)))


def swd(x: np.ndarray, y: np.ndarray, n_proj: int = 128, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    d = x.shape[1]
    vals = []
    for _ in range(n_proj):
        v = rng.normal(size=(d,))
        v /= (np.linalg.norm(v) + 1e-12)
        px = x @ v
        py = y @ v
        vals.append(_wasserstein_1d(px, py))
    return float(np.mean(vals))


# ---- Energy distance (2 versions) ----

def _mean_pairwise_l2(a: np.ndarray, b: np.ndarray, block: int = 512) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    n = a.shape[0]
    m = b.shape[0]
    aa = np.sum(a * a, axis=1, keepdims=True)  # n x 1
    bb = np.sum(b * b, axis=1, keepdims=True).T  # 1 x m

    total = 0.0
    count = 0
    for i in range(0, n, block):
        a_blk = a[i:i+block]
        aa_blk = aa[i:i+block]
        d2 = aa_blk + bb - 2.0 * (a_blk @ b.T)
        d = np.sqrt(np.maximum(d2, 0.0))
        total += float(d.sum())
        count += d.size
    return total / max(1, count)


def energy_distance_full(x: np.ndarray, y: np.ndarray, block: int = 512) -> float:
    exy = _mean_pairwise_l2(x, y, block=block)
    exx = _mean_pairwise_l2(x, x, block=block)
    eyy = _mean_pairwise_l2(y, y, block=block)
    return float(2.0 * exy - exx - eyy)


def energy_distance_sub(x: np.ndarray, y: np.ndarray, max_n: int = 1000, seed: int = 42, block: int = 512) -> float:
    rng = np.random.RandomState(seed)
    n = min(len(x), len(y), max_n)
    ix = rng.choice(len(x), n, replace=False)
    iy = rng.choice(len(y), n, replace=False)
    return energy_distance_full(x[ix], y[iy], block=block)


# ---- MMD-RBF ----

def _median_heuristic_sigma(x: np.ndarray, y: np.ndarray, max_pairs: int = 20000, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    nx = len(x)
    ny = len(y)
    k = min(max_pairs, nx * ny)
    ix = rng.randint(0, nx, size=k)
    iy = rng.randint(0, ny, size=k)
    dif = x[ix] - y[iy]
    d2 = np.sum(dif * dif, axis=1)
    med = np.median(d2)
    sigma2 = max(med, 1e-12)
    return float(np.sqrt(sigma2))


def _rbf_kernel_sum(a: np.ndarray, b: np.ndarray, sigma: float, block: int = 512, exclude_diag: bool = False) -> Tuple[float, int]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    n = a.shape[0]
    m = b.shape[0]
    sigma2 = max(1e-12, sigma * sigma)
    aa = np.sum(a * a, axis=1, keepdims=True)  # n x 1
    bb = np.sum(b * b, axis=1, keepdims=True).T  # 1 x m

    total = 0.0
    count = 0
    for i in range(0, n, block):
        a_blk = a[i:i+block]
        aa_blk = aa[i:i+block]
        d2 = aa_blk + bb - 2.0 * (a_blk @ b.T)
        kblk = np.exp(-d2 / (2.0 * sigma2))
        if exclude_diag and (a is b):
            for t in range(kblk.shape[0]):
                j = i + t
                if 0 <= j < m:
                    kblk[t, j] = 0.0
        total += float(kblk.sum())
        count += kblk.size
    if exclude_diag and (a is b):
        count = n * m - min(n, m)
    return total, count


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float, block: int = 512) -> float:
    sx, cx = _rbf_kernel_sum(x, x, sigma=sigma, block=block, exclude_diag=True)
    sy, cy = _rbf_kernel_sum(y, y, sigma=sigma, block=block, exclude_diag=True)
    sxy, cxy = _rbf_kernel_sum(x, y, sigma=sigma, block=block, exclude_diag=False)
    kxx = sx / max(1, cx)
    kyy = sy / max(1, cy)
    kxy = sxy / max(1, cxy)
    return float(kxx + kyy - 2.0 * kxy)


def mmd_rbf_sub(x: np.ndarray, y: np.ndarray, max_n: int = 1000, seed: int = 42, block: int = 512) -> float:
    rng = np.random.RandomState(seed)
    n = min(len(x), len(y), max_n)
    ix = rng.choice(len(x), n, replace=False)
    iy = rng.choice(len(y), n, replace=False)
    xs = x[ix]
    ys = y[iy]
    sigma = _median_heuristic_sigma(xs, ys, seed=seed)
    return mmd_rbf(xs, ys, sigma=sigma, block=block)


# -------------------------
# Bootstrap CI
# -------------------------

def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = min(len(x), len(y))
    vals = []
    for _ in range(n_boot):
        ix = rng.choice(len(x), n, replace=True)
        iy = rng.choice(len(y), n, replace=True)
        vals.append(fn(x[ix], y[iy]))
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p2.5": float(np.percentile(vals, 2.5)),
        "p97.5": float(np.percentile(vals, 97.5)),
    }


# -------------------------
# Metrics computation + caching
# -------------------------

def ensure_metrics_pair(
    pair_name: str,
    emb_a: Path,
    emb_b: Path,
    out_json: Path,
    n_boot: int,
    seed: int,
    force: bool,
    swd_proj: int,
    ed_block: int,
    ed_sub_n: int,
    mmd_sub_n: int,
) -> Path:
    if out_json.exists() and not force:
        print(f"[SKIP] metrics exist: {out_json}")
        return out_json

    x = np.load(emb_a)
    y = np.load(emb_b)

    fid_val = fid(x, y)
    kid_val = kid(x, y, seed=seed)
    swd_val = swd(x, y, n_proj=swd_proj, seed=seed)
    ed_full_val = energy_distance_full(x, y, block=ed_block)
    ed_sub_val = energy_distance_sub(x, y, max_n=ed_sub_n, seed=seed, block=ed_block)
    mmd_rbf_val = mmd_rbf_sub(x, y, max_n=mmd_sub_n, seed=seed, block=ed_block)

    fid_ci = bootstrap_ci(x, y, fid, n_boot=n_boot, seed=seed)
    kid_ci = bootstrap_ci(x, y, lambda a, b: kid(a, b, seed=seed), n_boot=n_boot, seed=seed)
    swd_ci = bootstrap_ci(x, y, lambda a, b: swd(a, b, n_proj=swd_proj, seed=seed), n_boot=n_boot, seed=seed)

    ed_sub_ci = bootstrap_ci(
        x, y,
        lambda a, b: energy_distance_sub(a, b, max_n=ed_sub_n, seed=seed, block=ed_block),
        n_boot=n_boot, seed=seed
    )

    mmd_ci = bootstrap_ci(
        x, y,
        lambda a, b: mmd_rbf_sub(a, b, max_n=mmd_sub_n, seed=seed, block=ed_block),
        n_boot=n_boot, seed=seed
    )

    res = {
        "pair": pair_name,
        "n_x": int(len(x)),
        "n_y": int(len(y)),
        "seed": int(seed),
        "n_boot": int(n_boot),

        "FID": float(fid_val),
        "KID": float(kid_val),
        "SWD": float(swd_val),
        "ED_full": float(ed_full_val),
        "ED_sub": float(ed_sub_val),
        "MMD_RBF": float(mmd_rbf_val),

        "FID_bootstrap": fid_ci,
        "KID_bootstrap": kid_ci,
        "SWD_bootstrap": swd_ci,
        "ED_sub_bootstrap": ed_sub_ci,
        "MMD_RBF_bootstrap": mmd_ci,

        "params": {
            "vgg_embedding": True,
            "swd_proj": int(swd_proj),
            "ed_block": int(ed_block),
            "ed_sub_n": int(ed_sub_n),
            "mmd_sub_n": int(mmd_sub_n),
        }
    }
    json_save(out_json, res)
    print(f"[OK  ] saved metrics: {out_json}")
    return out_json


# -------------------------
# Table output (console + CSV)
# -------------------------

def write_csv_table(rows: List[Dict], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    cols = [
        "Compared_to_REAL", "N",
        "FID", "FID_CI_2.5", "FID_CI_97.5",
        "KID", "KID_CI_2.5", "KID_CI_97.5",
        "SWD", "SWD_CI_2.5", "SWD_CI_97.5",
        "ED_full",
        "ED_sub", "ED_sub_CI_2.5", "ED_sub_CI_97.5",
        "MMD_RBF", "MMD_RBF_CI_2.5", "MMD_RBF_CI_97.5",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join([
            str(r["Compared_to_REAL"]),
            str(r["N"]),
            f"{r['FID']:.6g}", f"{r['FID_CI_2.5']:.6g}", f"{r['FID_CI_97.5']:.6g}",
            f"{r['KID']:.6g}", f"{r['KID_CI_2.5']:.6g}", f"{r['KID_CI_97.5']:.6g}",
            f"{r['SWD']:.6g}", f"{r['SWD_CI_2.5']:.6g}", f"{r['SWD_CI_97.5']:.6g}",
            f"{r['ED_full']:.6g}",
            f"{r['ED_sub']:.6g}", f"{r['ED_sub_CI_2.5']:.6g}", f"{r['ED_sub_CI_97.5']:.6g}",
            f"{r['MMD_RBF']:.6g}", f"{r['MMD_RBF_CI_2.5']:.6g}", f"{r['MMD_RBF_CI_97.5']:.6g}",
        ]))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_ci(val, lo, hi) -> str:
    return f"{val:.4g} [{lo:.4g}, {hi:.4g}]"


def print_table(rows: List[Dict]) -> None:
    headers = ["Compared_to_REAL", "N", "FID", "KID", "SWD", "ED_full", "ED_sub (CI95%)", "MMD_RBF (CI95%)"]
    w0 = max(len(headers[0]), max(len(r["Compared_to_REAL"]) for r in rows))
    w1 = max(len(headers[1]), max(len(str(r["N"])) for r in rows))

    print("\n" + "-" * 120)
    print(f"{headers[0]:<{w0}} | {headers[1]:>{w1}} | {headers[2]:<16} | {headers[3]:<16} | {headers[4]:<16} | {headers[5]:<16} | {headers[6]:<22} | {headers[7]:<22}")
    print("-" * 120)
    for r in rows:
        print(
            f"{r['Compared_to_REAL']:<{w0}} | {r['N']:>{w1}} | "
            f"{r['FID']:<16.4g} | {r['KID']:<16.4g} | {r['SWD']:<16.4g} | {r['ED_full']:<16.4g} | "
            f"{_fmt_ci(r['ED_sub'], r['ED_sub_CI_2.5'], r['ED_sub_CI_97.5']):<22} | "
            f"{_fmt_ci(r['MMD_RBF'], r['MMD_RBF_CI_2.5'], r['MMD_RBF_CI_97.5']):<22}"
        )
    print("-" * 120 + "\n")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--real_raw", default="/net/storage/pr3/plgrid/plggsmartudlteam/data/real_raw/")
    ap.add_argument("--aapm_png", default="/net/storage/pr3/plgrid/plggsmartudlteam/output/RPI/artifact/")
    # NEW default for OURS:
    ap.add_argument("--ours_png", default="/net/storage/pr3/plgrid/plggsmartudlteam/output/experiments_output/body_generation/")

    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=500)

    ap.add_argument("--wl", type=float, default=40.0)
    ap.add_argument("--ww", type=float, default=400.0)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--vgg_cut_idx", type=int, default=28)

    ap.add_argument("--swd_proj", type=int, default=128)
    ap.add_argument("--ed_block", type=int, default=512)
    ap.add_argument("--ed_sub_n", type=int, default=1000)
    ap.add_argument("--mmd_sub_n", type=int, default=1000)

    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    real_all = collect_real_raw(Path(args.real_raw))
    aapm_all = collect_aapm_png(Path(args.aapm_png))
    ours_all = collect_ours_png_by_imgid(Path(args.ours_png))

    if len(real_all) == 0:
        raise SystemExit(f"[ERROR] No REAL RAW found in: {args.real_raw}")
    if len(aapm_all) == 0:
        raise SystemExit(f"[ERROR] No AAPM PNG found in: {args.aapm_png}")
    if len(ours_all) == 0:
        raise SystemExit(f"[ERROR] No OURS PNG found in: {args.ours_png} (expected body_img*_gen.png)")

    n = min(args.n_samples, len(real_all), len(aapm_all), len(ours_all))
    print(f"[INFO] counts: REAL={len(real_all)} | AAPM={len(aapm_all)} | OURS={len(ours_all)}")
    print(f"[INFO] using N={n} | seed={args.seed} | wl={args.wl} ww={args.ww} | cut_idx={args.vgg_cut_idx}")

    real_s = sample_paths(real_all, n, args.seed)
    aapm_s = sample_paths(aapm_all, n, args.seed)
    ours_s = sample_paths(ours_all, n, args.seed)

    emb_dir = out_dir / "embeddings"
    tag = f"n{n}_seed{args.seed}_wl{int(args.wl)}_ww{int(args.ww)}_cut{args.vgg_cut_idx}"

    emb_real = emb_dir / f"REAL_{tag}.npy"
    emb_aapm = emb_dir / f"AAPM_{tag}.npy"
    emb_ours = emb_dir / f"OURS_{tag}.npy"

    emb_real = extract_embeddings("REAL", "real_raw", real_s, emb_real, args.wl, args.ww, args.batch_size, args.vgg_cut_idx, args.force)
    emb_aapm = extract_embeddings("AAPM", "png", aapm_s, emb_aapm, args.wl, args.ww, args.batch_size, args.vgg_cut_idx, args.force)
    emb_ours = extract_embeddings("OURS", "png", ours_s, emb_ours, args.wl, args.ww, args.batch_size, args.vgg_cut_idx, args.force)

    met_dir = out_dir / "metrics"
    m_aapm_real = ensure_metrics_pair(
        "AAPM_vs_REAL", emb_aapm, emb_real, met_dir / f"AAPM_vs_REAL_{tag}.json",
        n_boot=args.n_boot, seed=args.seed, force=args.force,
        swd_proj=args.swd_proj, ed_block=args.ed_block, ed_sub_n=args.ed_sub_n, mmd_sub_n=args.mmd_sub_n
    )
    m_ours_real = ensure_metrics_pair(
        "OURS_vs_REAL", emb_ours, emb_real, met_dir / f"OURS_vs_REAL_{tag}.json",
        n_boot=args.n_boot, seed=args.seed, force=args.force,
        swd_proj=args.swd_proj, ed_block=args.ed_block, ed_sub_n=args.ed_sub_n, mmd_sub_n=args.mmd_sub_n
    )

    ma = load_json(m_aapm_real)
    mo = load_json(m_ours_real)

    rows = []
    for label, m in [("AAPM", ma), ("OURS", mo)]:
        rows.append({
            "Compared_to_REAL": label,
            "N": n,

            "FID": m["FID"],
            "FID_CI_2.5": m["FID_bootstrap"]["p2.5"],
            "FID_CI_97.5": m["FID_bootstrap"]["p97.5"],

            "KID": m["KID"],
            "KID_CI_2.5": m["KID_bootstrap"]["p2.5"],
            "KID_CI_97.5": m["KID_bootstrap"]["p97.5"],

            "SWD": m["SWD"],
            "SWD_CI_2.5": m["SWD_bootstrap"]["p2.5"],
            "SWD_CI_97.5": m["SWD_bootstrap"]["p97.5"],

            "ED_full": m["ED_full"],

            "ED_sub": m["ED_sub"],
            "ED_sub_CI_2.5": m["ED_sub_bootstrap"]["p2.5"],
            "ED_sub_CI_97.5": m["ED_sub_bootstrap"]["p97.5"],

            "MMD_RBF": m["MMD_RBF"],
            "MMD_RBF_CI_2.5": m["MMD_RBF_bootstrap"]["p2.5"],
            "MMD_RBF_CI_97.5": m["MMD_RBF_bootstrap"]["p97.5"],
        })

    delta = {
        "N": n,
        "seed": args.seed,
        "wl": args.wl,
        "ww": args.ww,
        "vgg_cut_idx": args.vgg_cut_idx,
        "params": {"swd_proj": args.swd_proj, "ed_sub_n": args.ed_sub_n, "mmd_sub_n": args.mmd_sub_n},
        "AAPM_vs_REAL": {k: ma[k] for k in ["FID", "KID", "SWD", "ED_full", "ED_sub", "MMD_RBF"]},
        "OURS_vs_REAL": {k: mo[k] for k in ["FID", "KID", "SWD", "ED_full", "ED_sub", "MMD_RBF"]},
        "delta_(AAPM-OURS)_vs_REAL": {
            "FID": float(ma["FID"] - mo["FID"]),
            "KID": float(ma["KID"] - mo["KID"]),
            "SWD": float(ma["SWD"] - mo["SWD"]),
            "ED_full": float(ma["ED_full"] - mo["ED_full"]),
            "ED_sub": float(ma["ED_sub"] - mo["ED_sub"]),
            "MMD_RBF": float(ma["MMD_RBF"] - mo["MMD_RBF"]),
        }
    }
    json_save(out_dir / "summary" / f"DELTA_vs_REAL_{tag}.json", delta)

    out_csv = out_dir / "summary" / f"TABLE_vs_REAL_{tag}.csv"
    write_csv_table(rows, out_csv)
    print(f"[OK  ] wrote table: {out_csv}")

    print_table(rows)
    print(f"[OK  ] delta summary: {out_dir / 'summary' / f'DELTA_vs_REAL_{tag}.json'}")
    print("[DONE]")


if __name__ == "__main__":
    main()