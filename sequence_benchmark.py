"""
Sequential benchmark for an image dataset.
Reads a directory of images, runs frame-to-frame matching, and reports average FPS and track continuity.

Key flags:
  --gap N    Match every N frames instead of every frame. Larger gap = more parallax,
             better translation recovery but lower FPS for evaluation.
             gap=1 (default): consecutive frames, good for tracking latency.
             gap=5-10: adds parallax, makes translation direction recoverable.
"""
import argparse, time, glob, os
import cv2
import numpy as np

from matcher_util import get_matcher

# ── Pose utilities ──────────────────────────────────────────────────────────

def q_to_rot_mat(q):
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,  2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,  1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,  2*qy*qz + 2*qx*qw,  1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def load_poses(path):
    """
    Loads TartanAir / TUM pose file.
    Format per line:  tx ty tz  qx qy qz qw   (camera-to-world)
    Returns list of 4×4 numpy matrices.
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[:7])
            R = q_to_rot_mat([qx, qy, qz, qw])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses

# ── Error metrics ────────────────────────────────────────────────────────────

def rotation_error_deg(R_est, R_gt):
    """Geodesic rotation error in degrees."""
    R_err = R_gt.T @ R_est
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def translation_angle_deg(t_est, t_gt):
    """
    Angular error between estimated and GT translation directions (degrees).
    NOTE: recoverPose gives only the direction (unit vector), not scale.
    This metric is ONLY meaningful when there is sufficient parallax (gap >= 5).
    At gap=1 (consecutive frames, near-pure-rotation), this will be ~90° randomly.
    """
    t_est = t_est.flatten()
    t_gt  = t_gt.flatten()
    n_est, n_gt = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if n_est < 1e-6 or n_gt < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(t_est, t_gt) / (n_est * n_gt), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  required=True, help="Directory of images")
    ap.add_argument("--method",   default="xfeat", choices=["xfeat", "superpoint", "aliked", "loftr", "roma", "silk", "dedode", "r2d2"])
    ap.add_argument("--poses",    default=None,  help="GT pose file (TartanAir / TUM format)")
    ap.add_argument("--gap",      type=int, default=1,
                    help="Frame gap for matching (default: 1 = consecutive). "
                         "Use --gap 5 or --gap 10 to add parallax for better translation recovery.")
    ap.add_argument("--focal",    type=float, default=320.0, help="Focal length in pixels")
    ap.add_argument("--cx",       type=float, default=320.0, help="Principal point X")
    ap.add_argument("--cy",       type=float, default=240.0, help="Principal point Y")
    args = ap.parse_args()

    # ── Load poses ──
    poses = None
    if args.poses:
        poses = load_poses(args.poses)
        print(f"Loaded {len(poses)} GT poses.")

    # ── Load image list ──
    files = sorted(glob.glob(os.path.join(args.dataset, "*.*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) < 2:
        print("Not enough images in dataset directory.")
        return

    gap = args.gap
    # Pairs: (0, gap), (gap, 2*gap), (2*gap, 3*gap), …
    pairs = [(i, i + gap) for i in range(0, len(files) - gap, gap)]
    print(f"Loading {args.method}...")
    print(f"Benchmarking {len(files)} images → {len(pairs)} pairs at gap={gap}...")

    matcher = get_matcher(args.method)

    # ── Warmup ──
    matcher.match_images(cv2.imread(files[0]), cv2.imread(files[min(gap, len(files)-1)]))
    matcher.timer.reset()

    # ── Precompute features for the very first frame ──
    K = np.array([[args.focal, 0, args.cx], [0, args.focal, args.cy], [0, 0, 1]])

    t_read_total = 0.0
    t_prep_total = t_detect_total = t_match_total = 0.0
    total_ms = 0.0
    total_matches = 0
    rot_errors, trans_errors, inlier_ratios = [], [], []

    # Feature cache: {file_index: feat}
    feat_cache = {}

    def get_feat(idx):
        """Read + prep + detect for file[idx], with single-entry cache."""
        nonlocal t_read_total, t_prep_total, t_detect_total
        if idx in feat_cache:
            return feat_cache[idx]
        t0 = time.perf_counter()
        img = cv2.imread(files[idx])
        t_read_total += (time.perf_counter() - t0) * 1000
        prep = matcher.prep(img)
        feat = matcher.detect(prep)
        tp, td, _ = matcher.timer.get_and_reset()
        t_prep_total += tp
        t_detect_total += td
        feat_cache[idx] = feat
        return feat

    # Pre-warm first frame
    get_feat(0)

    for pair_num, (idx_a, idx_b) in enumerate(pairs):
        feat_a = get_feat(idx_a)
        feat_b = get_feat(idx_b)

        # Only keep one previous frame in cache to save memory
        if idx_a - gap in feat_cache:
            del feat_cache[idx_a - gap]

        t0 = time.perf_counter()
        kp1, kp2 = matcher.match(feat_a, feat_b)
        dt = (time.perf_counter() - t0) * 1000

        _, _, tm = matcher.timer.get_and_reset()
        t_match_total += tm
        total_ms   += dt
        n_matches   = len(kp1)
        total_matches += n_matches

        # ── Pose evaluation ──
        r_err = t_err = -1.0
        n_inliers = 0
        inlier_ratio = 0.0

        if poses is not None and idx_a < len(poses) and idx_b < len(poses) and n_matches >= 8:
            R_a = poses[idx_a][:3, :3]
            R_b = poses[idx_b][:3, :3]
            t_a = poses[idx_a][:3, 3]
            t_b = poses[idx_b][:3, 3]
            # Relative pose: OpenCV recoverPose returns R, t mapping from A to B
            # such that X_b = R * X_a + t.
            # P_world = R_a * X_a + t_a  and  P_world = R_b * X_b + t_b
            # => X_b = R_b.T @ R_a * X_a + R_b.T @ (t_a - t_b)
            R_gt_ned = R_b.T @ R_a
            t_gt_ned = R_b.T @ (t_a - t_b)

            # TartanAir uses NED: X=forward, Y=right, Z=down
            # OpenCV uses CV:     X=right,   Y=down,  Z=forward
            T_ned2cv = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ], dtype=float)
            
            R_gt = T_ned2cv @ R_gt_ned @ T_ned2cv.T
            t_gt = T_ned2cv @ t_gt_ned

            E, emask = cv2.findEssentialMat(kp1, kp2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape == (3, 3):
                n_inliers, R_est, t_est, _ = cv2.recoverPose(E, kp1, kp2, K, mask=emask)
                inlier_ratio = n_inliers / n_matches if n_matches > 0 else 0.0
                r_err = rotation_error_deg(R_est, R_gt)
                t_err = translation_angle_deg(t_est, t_gt)
                rot_errors.append(r_err)
                trans_errors.append(t_err)
                inlier_ratios.append(inlier_ratio)

        if pair_num % 10 == 0:
            pose_str = ""
            if r_err >= 0:
                pose_str = (f"  R_err:{r_err:5.1f}°  t_err:{t_err:5.1f}°"
                            f"  inliers:{n_inliers}/{n_matches} ({inlier_ratio*100:.0f}%)")
            print(f"Pair {pair_num:04d} [{idx_a:04d}→{idx_b:04d}]"
                  f"  matches:{n_matches:4d}  latency:{dt:5.1f} ms{pose_str}")

    n_pairs = len(pairs)
    avg_ms      = total_ms      / n_pairs
    avg_matches = total_matches / n_pairs

    print("\n─── Summary ──────────────────────────────")
    print(f"  Method:      {args.method}")
    print(f"  Frame gap:   {gap}")
    print(f"  Pairs:       {n_pairs}")
    print(f"  Avg Latency: {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
    print(f"  Avg Matches: {avg_matches:.1f} per pair")

    if rot_errors:
        print(f"\n  ── Accuracy (gap={gap}) ─────────────────")
        print(f"  Avg Rot Error:         {np.mean(rot_errors):.2f}°  (med: {np.median(rot_errors):.2f}°)")
        print(f"  Avg Trans Dir Error:   {np.mean(trans_errors):.2f}°  (med: {np.median(trans_errors):.2f}°)")
        print(f"  Avg Inlier Ratio:      {np.mean(inlier_ratios)*100:.1f}%")
        fail = (n_pairs - len(rot_errors)) / n_pairs * 100
        print(f"  Pose Recovery Fails:   {fail:.1f}%")
        if gap == 1:
            print(f"\n  ⚠  NOTE: At gap=1 the translation direction error is unreliable")
            print(f"  (small baseline → near-degenerate Essential matrix).")
            print(f"  Re-run with --gap 5 or --gap 10 for meaningful translation accuracy.")

    print("\n─── Time Statistics ───────────────────────")
    print(f"  Image Reading:      {t_read_total:.1f} ms")
    print(f"  Preprocessing:      {t_prep_total:.1f} ms")
    print(f"  Keypoint Detection: {t_detect_total:.1f} ms")
    print(f"  Matching:           {t_match_total:.1f} ms")
    total_time = t_read_total + t_prep_total + t_detect_total + t_match_total
    print(f"  Total:              {total_time:.1f} ms")

if __name__ == "__main__":
    main()
