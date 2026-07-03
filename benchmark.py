"""
Sparse optical-flow / feature-matching benchmark.
Measures: keypoint count, match count, inlier ratio, mean reprojection error, latency.

Usage:
    python benchmark.py --img1 frame0.png --img2 frame1.png [--methods all] [--out results.csv]

Methods (--methods CSV or 'all'):
    orb, sift, akaze, lkof, superpoint, lightglue, xfeat, aliked, loftr

Each method must implement the Matcher protocol: detect_and_match(img1, img2) -> MatchResult.
"""
import argparse, time, csv, sys
from dataclasses import dataclass, fields
from typing import Optional
import cv2
import numpy as np

from matcher_util import get_matcher

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
    prep_ms:      float = 0.0
    detect_ms:    float = 0.0
    match_ms:     float = 0.0
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
    t0 = time.perf_counter()
    kp1, des1 = detector.detectAndCompute(img1_g, None)
    kp2, des2 = detector.detectAndCompute(img2_g, None)
    t_detect = (time.perf_counter() - t0) * 1000
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0, 0, [], kp1, kp2, t_detect, 0.0
    norm = cv2.NORM_HAMMING if name in ("orb", "akaze") else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=True)
    t0 = time.perf_counter()
    matches = bf.match(des1, des2)
    t_match = (time.perf_counter() - t0) * 1000
    matches = sorted(matches, key=lambda m: m.distance)
    return len(kp1), len(kp2), matches, kp1, kp2, t_detect, t_match


def match_orb(img1, img2):
    det = cv2.ORB_create(nfeatures=2000)
    t0 = time.perf_counter()
    img1_g, img2_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    t_prep = (time.perf_counter() - t0) * 1000
    n1, n2, m, kp1, kp2, t_det, t_mat = _run_classical("orb", det, img1_g, img2_g)
    ni, re = _homography_metrics(kp1, kp2, m, img1.shape)
    return n1, n2, len(m), ni, re, t_prep, t_det, t_mat


def match_sift(img1, img2):
    det = cv2.SIFT_create(nfeatures=2000)
    t0 = time.perf_counter()
    img1_g, img2_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    t_prep = (time.perf_counter() - t0) * 1000
    n1, n2, m, kp1, kp2, t_det, t_mat = _run_classical("sift", det, img1_g, img2_g)
    ni, re = _homography_metrics(kp1, kp2, m, img1.shape)
    return n1, n2, len(m), ni, re, t_prep, t_det, t_mat


def match_akaze(img1, img2):
    det = cv2.AKAZE_create()
    t0 = time.perf_counter()
    img1_g, img2_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    t_prep = (time.perf_counter() - t0) * 1000
    n1, n2, m, kp1, kp2, t_det, t_mat = _run_classical("akaze", det, img1_g, img2_g)
    ni, re = _homography_metrics(kp1, kp2, m, img1.shape)
    return n1, n2, len(m), ni, re, t_prep, t_det, t_mat


def match_lkof(img1, img2):
    """Lucas-Kanade sparse optical flow on ShiTomasi corners."""
    t0 = time.perf_counter()
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    t_prep = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    kp1 = cv2.goodFeaturesToTrack(g1, maxCorners=500, qualityLevel=0.01, minDistance=7)
    t_detect = (time.perf_counter() - t0) * 1000
    if kp1 is None:
        return 0, 0, 0, 0, float('nan'), t_prep, t_detect, 0.0
        
    t0 = time.perf_counter()
    kp2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, kp1, None)
    t_match = (time.perf_counter() - t0) * 1000
    
    good1 = kp1[status.ravel() == 1]
    good2 = kp2[status.ravel() == 1]
    n_match = len(good1)
    if n_match < 4:
        return len(kp1), len(kp1), n_match, 0, float('nan'), t_prep, t_detect, t_match
        
    # ponytail: build fake cv2.DMatch list so _homography_metrics works cleanly
    src = good1.reshape(-1, 1, 2).astype(np.float32)
    dst = good2.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return len(kp1), len(kp1), n_match, 0, float('nan'), t_prep, t_detect, t_match
    ni = int(mask.sum())
    proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
    re = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return len(kp1), len(kp1), n_match, ni, re, t_prep, t_detect, t_match


# ── deep methods ──────────────────────────────────────────────────────────────

def run_deep_matcher(name, img1, img2):
    matcher = get_matcher(name)
    matcher.match_images(img1, img2) # warmup inner components
    matcher.timer.reset()
    
    p1 = matcher.prep(img1)
    p2 = matcher.prep(img2)
    f1 = matcher.detect(p1)
    f2 = matcher.detect(p2)
    kp1, kp2 = matcher.match(f1, f2)
    
    t_p, t_d, t_m = matcher.timer.get_and_reset()
    n_m = len(kp1)
    n_k1 = matcher.get_keypoint_count(f1) if name != "loftr" else n_m
    n_k2 = matcher.get_keypoint_count(f2) if name != "loftr" else n_m
    
    if n_m < 4:
        return n_k1, n_k2, n_m, 0, float('nan'), t_p, t_d, t_m
        
    src = kp1.reshape(-1, 1, 2).astype(np.float32)
    dst = kp2.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ni = int(mask.sum()) if mask is not None else 0
    re = float('nan')
    if H is not None and mask is not None and ni > 0:
        proj = cv2.perspectiveTransform(src[mask.ravel() == 1], H)
        re   = float(np.mean(np.linalg.norm(proj - dst[mask.ravel() == 1], axis=2)))
    return n_k1, n_k2, n_m, ni, re, t_p, t_d, t_m


def match_superpoint_lightglue(img1, img2): return run_deep_matcher("superpoint", img1, img2)
def match_xfeat(img1, img2): return run_deep_matcher("xfeat", img1, img2)
def match_aliked(img1, img2): return run_deep_matcher("aliked", img1, img2)
def match_loftr(img1, img2): return run_deep_matcher("loftr", img1, img2)


# ── registry ──────────────────────────────────────────────────────────────────

METHODS = {
    "orb":        match_orb,
    "sift":       match_sift,
    "akaze":      match_akaze,
    "lkof":       match_lkof,
    "superpoint": match_superpoint_lightglue,
    "lightglue":  match_superpoint_lightglue,
    "xfeat":      match_xfeat,
    "aliked":     match_aliked,
    "loftr":      match_loftr,
}

DEDUP = {
    match_superpoint_lightglue: "superpoint",
}

def run_one(name, fn, img1, img2, warmup=1) -> MatchResult:
    try:
        # warmup (matters for GPU JIT)
        for _ in range(warmup):
            fn(img1, img2)
        t0 = time.perf_counter()
        n1, n2, nm, ni, re, t_prep, t_det, t_mat = fn(img1, img2)
        lat = (time.perf_counter() - t0) * 1000
        ir = ni / nm if nm > 0 else 0.0
        return MatchResult(name, n1, n2, nm, ni, ir, re, lat, t_prep, t_det, t_mat)
    except Exception as e:
        return MatchResult(name, error=str(e))

def print_table(results):
    try:
        from tabulate import tabulate
        rows = []
        for r in results:
            if r.error:
                rows.append([r.method, "ERROR", r.error[:60], "", "", "", "", "", "", "", ""])
            else:
                rows.append([r.method, r.n_kp1, r.n_kp2, r.n_matches,
                             r.n_inliers, f"{r.inlier_ratio:.2f}",
                             f"{r.reproj_err:.3f}" if not np.isnan(r.reproj_err) else "nan",
                             f"{r.latency_ms:.1f}",
                             f"{r.prep_ms:.1f}", f"{r.detect_ms:.1f}", f"{r.match_ms:.1f}"])
        print(tabulate(rows, headers=["method","kp1","kp2","matches","inliers",
                                      "inlier_r","err_px","ms","prep_ms","det_ms","mat_ms"],
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

    t0 = time.perf_counter()
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    t_read = (time.perf_counter() - t0) * 1000
    assert img1 is not None, f"Cannot read {args.img1}"
    assert img2 is not None, f"Cannot read {args.img2}"

    print(f"Image Reading Time: {t_read:.1f} ms")

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
