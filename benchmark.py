"""
Sparse optical-flow / feature-matching benchmark.
Measures: keypoint count, match count, inlier ratio, mean reprojection error, latency.

Usage:
    python benchmark.py --img1 frame0.png --img2 frame1.png [--methods all] [--out results.csv]

Methods (--methods CSV or 'all'):
    orb, sift, akaze, lkof, superpoint, lightglue, xfeat, aliked, loftr

Each method must implement the Matcher protocol: detect_and_match(img1, img2) -> MatchResult.
"""
import argparse, time, csv, sys, importlib
from dataclasses import dataclass, fields
from typing import Optional
import cv2
import numpy as np

# ── result container ──────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    method:       str
    n_kp1:        int   = 0          # keypoints in frame 1
    n_kp2:        int   = 0          # keypoints in frame 2
    n_matches:    int   = 0          # raw matches
    n_inliers:    int   = 0          # RANSAC inliers
    inlier_ratio: float = 0.0        # inliers / matches
    reproj_err:   float = float('nan')  # mean px reprojection error (if homography found)
    latency_ms:   float = 0.0        # end-to-end wall time
    error:        Optional[str] = None


def _homography_metrics(kp1, kp2, matches, shape):
    """Return (n_inliers, mean_reproj_err) via RANSAC homography."""
    if len(matches) < 4:
        return 0, float('nan')
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return 0, float('nan')
    inliers = int(mask.sum())
    proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
    err = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return inliers, err


# ── classical methods ─────────────────────────────────────────────────────────

def _run_classical(name, detector, img1_g, img2_g):
    kp1, des1 = detector.detectAndCompute(img1_g, None)
    kp2, des2 = detector.detectAndCompute(img2_g, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0, 0, [], kp1, kp2
    norm = cv2.NORM_HAMMING if name in ("orb", "akaze") else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    return len(kp1), len(kp2), matches, kp1, kp2


def match_orb(img1, img2):
    det = cv2.ORB_create(nfeatures=2000)
    img1_g, img2_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    n1, n2, m, kp1, kp2 = _run_classical("orb", det, img1_g, img2_g)
    ni, re = _homography_metrics(kp1, kp2, m, img1.shape)
    return n1, n2, len(m), ni, re


def match_sift(img1, img2):
    det = cv2.SIFT_create(nfeatures=2000)
    img1_g, img2_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    n1, n2, m, kp1, kp2 = _run_classical("sift", det, img1_g, img2_g)
    ni, re = _homography_metrics(kp1, kp2, m, img1.shape)
    return n1, n2, len(m), ni, re


def match_akaze(img1, img2):
    det = cv2.AKAZE_create()
    img1_g, img2_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    n1, n2, m, kp1, kp2 = _run_classical("akaze", det, img1_g, img2_g)
    ni, re = _homography_metrics(kp1, kp2, m, img1.shape)
    return n1, n2, len(m), ni, re


def match_lkof(img1, img2):
    """Lucas-Kanade sparse optical flow on ShiTomasi corners."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1 = cv2.goodFeaturesToTrack(g1, maxCorners=500, qualityLevel=0.01, minDistance=7)
    if kp1 is None:
        return 0, 0, 0, 0, float('nan')
    kp2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, kp1, None)
    good1 = kp1[status.ravel() == 1]
    good2 = kp2[status.ravel() == 1]
    n_match = len(good1)
    if n_match < 4:
        return len(kp1), len(kp1), n_match, 0, float('nan')
    # ponytail: build fake cv2.DMatch list so _homography_metrics works cleanly
    src = good1.reshape(-1, 1, 2).astype(np.float32)
    dst = good2.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return len(kp1), len(kp1), n_match, 0, float('nan')
    ni = int(mask.sum())
    proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
    re = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return len(kp1), len(kp1), n_match, ni, re


# ── deep methods ──────────────────────────────────────────────────────────────
# Each returns (n_kp1, n_kp2, n_matches, n_inliers, reproj_err)
# Import failures are caught at call time so the script stays runnable without GPU deps.

def match_superpoint_lightglue(img1, img2):
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    def _prep(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        f1 = extractor.extract(_prep(img1))
        f2 = extractor.extract(_prep(img2))
        out = matcher({"image0": f1, "image1": f2})
        f1, f2, out = [rbd(x) for x in [f1, f2, out]]

    kp1 = f1["keypoints"].cpu().numpy()
    kp2 = f2["keypoints"].cpu().numpy()
    matches = out["matches"].cpu().numpy()
    
    if len(matches) < 4:
        return len(kp1), len(kp2), len(matches), 0, float('nan')
        
    src = kp1[matches[..., 0]].reshape(-1, 1, 2).astype(np.float32)
    dst = kp2[matches[..., 1]].reshape(-1, 1, 2).astype(np.float32)
    n_m = len(src)
    
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ni = int(mask.sum()) if mask is not None else 0
    re = float('nan')
    if H is not None and mask is not None and ni > 0:
        proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
        re   = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return len(kp1), len(kp2), n_m, ni, re


def match_xfeat(img1, img2):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ponytail: load via torch hub, no pip package needed
    xf = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1024).eval().to(device)

    def _prep(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        out1 = xf.detectAndCompute(_prep(img1), top_k=1024)[0]
        out2 = xf.detectAndCompute(_prep(img2), top_k=1024)[0]
        idxs0, idxs1 = xf.match(out1["descriptors"], out2["descriptors"])

    kp1 = out1["keypoints"][idxs0].cpu().numpy()
    kp2 = out2["keypoints"][idxs1].cpu().numpy()
    n_m = len(kp1)
    if n_m < 4:
        return len(out1["keypoints"]), len(out2["keypoints"]), n_m, 0, float('nan')
    src = kp1.reshape(-1, 1, 2).astype(np.float32)
    dst = kp2.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ni = int(mask.sum()) if mask is not None else 0
    re = float('nan')
    if H is not None and mask is not None and ni > 0:
        proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
        re   = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return len(out1["keypoints"]), len(out2["keypoints"]), n_m, ni, re


def match_aliked(img1, img2):
    import torch
    from lightglue import LightGlue, ALIKED
    from lightglue.utils import rbd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ALIKED(max_num_keypoints=1024, model_name="aliked-n16rot").eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    def _prep(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        f1 = extractor.extract(_prep(img1))
        f2 = extractor.extract(_prep(img2))
        out = matcher({"image0": f1, "image1": f2})
        f1, f2, out = [rbd(x) for x in [f1, f2, out]]

    kp1 = f1["keypoints"].cpu().numpy()
    kp2 = f2["keypoints"].cpu().numpy()
    matches = out["matches"].cpu().numpy()
    
    if len(matches) < 4:
        return len(kp1), len(kp2), len(matches), 0, float('nan')
        
    src = kp1[matches[..., 0]].reshape(-1, 1, 2).astype(np.float32)
    dst = kp2[matches[..., 1]].reshape(-1, 1, 2).astype(np.float32)
    n_m = len(src)
    
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ni = int(mask.sum()) if mask is not None else 0
    re = float('nan')
    if H is not None and mask is not None and ni > 0:
        proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
        re   = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return len(kp1), len(kp2), n_m, ni, re


def match_loftr(img1, img2):
    import torch
    import kornia.feature as KF
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

    def _prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return torch.from_numpy(g)[None, None].to(device)

    with torch.inference_mode():
        inp = {"image0": _prep(img1), "image1": _prep(img2)}
        corr = loftr(inp)

    src = corr["keypoints0"].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
    dst = corr["keypoints1"].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
    conf = corr["confidence"].cpu().numpy()
    n_m = len(src)
    if n_m < 4:
        return n_m, n_m, n_m, 0, float('nan')
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ni = int(mask.sum()) if mask is not None else 0
    re = float('nan')
    if H is not None and mask is not None and ni > 0:
        proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
        re   = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    # ponytail: LoFTR has no separate kp1/kp2 count (detector-free); report n_m for both
    return n_m, n_m, n_m, ni, re


# ── registry ──────────────────────────────────────────────────────────────────

METHODS = {
    "orb":        match_orb,
    "sift":       match_sift,
    "akaze":      match_akaze,
    "lkof":       match_lkof,
    "superpoint": match_superpoint_lightglue,   # SP+LG together
    "lightglue":  match_superpoint_lightglue,   # alias
    "xfeat":      match_xfeat,
    "aliked":     match_aliked,
    "loftr":      match_loftr,
}

DEDUP = {                  # prevent running SP+LG twice when both aliases given
    match_superpoint_lightglue: "superpoint",
}


def run_one(name, fn, img1, img2, warmup=1) -> MatchResult:
    try:
        # warmup (matters for GPU JIT)
        for _ in range(warmup):
            fn(img1, img2)
        t0 = time.perf_counter()
        n1, n2, nm, ni, re = fn(img1, img2)
        lat = (time.perf_counter() - t0) * 1000
        ir = ni / nm if nm > 0 else 0.0
        return MatchResult(name, n1, n2, nm, ni, ir, re, lat)
    except Exception as e:
        return MatchResult(name, error=str(e))


def print_table(results):
    try:
        from tabulate import tabulate
        rows = []
        for r in results:
            if r.error:
                rows.append([r.method, "ERROR", r.error[:60]])
            else:
                rows.append([r.method, r.n_kp1, r.n_kp2, r.n_matches,
                             r.n_inliers, f"{r.inlier_ratio:.2f}",
                             f"{r.reproj_err:.3f}" if not np.isnan(r.reproj_err) else "nan",
                             f"{r.latency_ms:.1f}"])
        print(tabulate(rows, headers=["method","kp1","kp2","matches","inliers",
                                      "inlier_r","reproj_err_px","ms"],
                       tablefmt="rounded_outline"))
    except ImportError:
        for r in results:
            print(vars(r))


def save_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(MatchResult)])
        w.writeheader()
        for r in results:
            w.writerow(vars(r))
    print(f"Saved → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--methods", default="all",
                    help="Comma-sep list or 'all'. E.g. orb,sift,xfeat")
    ap.add_argument("--out", default="", help="Optional CSV output path")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    assert img1 is not None, f"Cannot read {args.img1}"
    assert img2 is not None, f"Cannot read {args.img2}"

    names = list(METHODS.keys()) if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    seen_fns = set()
    results = []
    warmup = 0 if args.no_warmup else 1

    for name in names:
        if name not in METHODS:
            print(f"[SKIP] unknown method: {name}")
            continue
        fn = METHODS[name]
        canonical = DEDUP.get(fn, name)
        if fn in seen_fns:
            print(f"[SKIP] {name} already ran as {canonical}")
            continue
        seen_fns.add(fn)
        print(f"[RUN ] {canonical} ...", flush=True)
        r = run_one(canonical, fn, img1, img2, warmup=warmup)
        results.append(r)

    print()
    print_table(results)
    if args.out:
        save_csv(results, args.out)


if __name__ == "__main__":
    main()
