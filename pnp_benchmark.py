import argparse, time, glob, os
import cv2
import numpy as np

from matcher_util import get_matcher
from sequence_benchmark import load_poses, rotation_error_deg, translation_angle_deg

def read_depth(path):
    """Read TartanAir float32 depth map encoded in RGBA PNG."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # The PNG is 4 channels of 8-bit, which cast exactly to 1 channel of 32-bit float
    depth = img.view(np.float32).squeeze(-1)
    return depth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="dataset/AbandonedFactory/Data_hard/P003/image_lcam_front", required=False, help="Directory of images (e.g. image_lcam_front)")
    ap.add_argument("--depth",    default="dataset/AbandonedFactory/Data_hard/P003/depth_lcam_front", required=False, help="Directory of depth maps (e.g. depth_lcam_front)")
    ap.add_argument("--poses",    default="dataset/AbandonedFactory/Data_hard/P003/pose_lcam_front.txt", required=False, help="GT pose file (TartanAir format)")
    ap.add_argument("--method",   default="xfeat", choices=["xfeat", "superpoint", "aliked", "loftr", "roma", "silk", "dedode", "r2d2"])
    ap.add_argument("--gap",      type=int, default=1, help="Frame gap for matching")
    ap.add_argument("--focal",    type=float, default=320.0, help="Focal length in pixels")
    ap.add_argument("--cx",       type=float, default=320.0, help="Principal point X")
    ap.add_argument("--cy",       type=float, default=240.0, help="Principal point Y")
    args = ap.parse_args()

    poses = load_poses(args.poses)
    print(f"Loaded {len(poses)} GT poses.")

    img_files = sorted(glob.glob(os.path.join(args.dataset, "*.*")))
    img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    depth_files = sorted(glob.glob(os.path.join(args.depth, "*.png")))

    if len(img_files) != len(depth_files):
        print(f"Warning: {len(img_files)} images but {len(depth_files)} depth maps. Ensure perfectly aligned sequences.")

    gap = args.gap
    pairs = [(i, i + gap) for i in range(0, min(len(img_files), len(depth_files)) - gap, gap)]
    print(f"Benchmarking PnP map expansion with {args.method} at gap={gap}...")

    matcher = get_matcher(args.method)
    
    # Warmup
    matcher.match_images(cv2.imread(img_files[0]), cv2.imread(img_files[min(gap, len(img_files)-1)]))
    matcher.timer.reset()

    K = np.array([[args.focal, 0, args.cx], [0, args.focal, args.cy], [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros(4)

    total_ms = 0.0
    t_read_total = t_prep_total = t_detect_total = t_match_total = t_pnp_total = 0.0
    
    rot_errors, trans_errors, trans_scale_errors = [], [], []
    inlier_ratios = []

    feat_cache = {}

    def get_feat(idx):
        nonlocal t_read_total, t_prep_total, t_detect_total
        if idx in feat_cache: return feat_cache[idx]
        
        t0 = time.perf_counter()
        img = cv2.imread(img_files[idx])
        t_read_total += (time.perf_counter() - t0) * 1000
        
        prep = matcher.prep(img)
        feat = matcher.detect(prep)
        tp, td, _ = matcher.timer.get_and_reset()
        
        t_prep_total += tp
        t_detect_total += td
        feat_cache[idx] = feat
        return feat

    get_feat(0)

    for pair_num, (idx_a, idx_b) in enumerate(pairs):
        feat_a = get_feat(idx_a)
        feat_b = get_feat(idx_b)

        if idx_a - gap in feat_cache:
            del feat_cache[idx_a - gap]

        t0 = time.perf_counter()
        kp1, kp2 = matcher.match(feat_a, feat_b)
        dt = (time.perf_counter() - t0) * 1000
        total_ms += dt
        
        _, _, tm = matcher.timer.get_and_reset()
        t_match_total += tm

        # -- PnP Pose Evaluation --
        # 1. Backproject kp1 using depth map of frame A
        t0_pnp = time.perf_counter()
        depth_a = read_depth(depth_files[idx_a])
        
        pts3d = []
        pts2d = []
        
        for (u1, v1), (u2, v2) in zip(kp1, kp2):
            iu1, iv1 = int(round(u1)), int(round(v1))
            if 0 <= iu1 < depth_a.shape[1] and 0 <= iv1 < depth_a.shape[0]:
                Z = float(depth_a[iv1, iu1])
                if Z > 0.0 and Z < 100.0:  # valid depth check
                    X = (u1 - args.cx) * Z / args.focal
                    Y = (v1 - args.cy) * Z / args.focal
                    pts3d.append([X, Y, Z])
                    pts2d.append([u2, v2])

        pts3d = np.array(pts3d, dtype=np.float32)
        pts2d = np.array(pts2d, dtype=np.float32)

        if len(pts3d) >= 10:
            # Tighten PnP parameters for optimal accuracy
            success, rvec, t_est, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, K, dist_coeffs, 
                reprojectionError=1.5, iterationsCount=1000, flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None:
                # Refine pose using Levenberg-Marquardt on the inliers
                inlier_pts3d = pts3d[inliers].reshape(-1, 3)
                inlier_pts2d = pts2d[inliers].reshape(-1, 2)
                rvec, t_est = cv2.solvePnPRefineLM(inlier_pts3d, inlier_pts2d, K, dist_coeffs, rvec, t_est)
                
                t_pnp_total += (time.perf_counter() - t0_pnp) * 1000
                
                R_est, _ = cv2.Rodrigues(rvec)
                n_inliers = len(inliers)
                inlier_ratio = n_inliers / len(pts3d)

                # Ground Truth
                R_a = poses[idx_a][:3, :3]
                R_b = poses[idx_b][:3, :3]
                t_a = poses[idx_a][:3, 3]
                t_b = poses[idx_b][:3, 3]

                # GT from A to B (TartanAir NED)
                R_gt_ned = R_b.T @ R_a
                t_gt_ned = R_b.T @ (t_a - t_b)

                # NED to OpenCV Camera Frame
                T_ned2cv = np.array([
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]
                ], dtype=float)
                
                R_gt = T_ned2cv @ R_gt_ned @ T_ned2cv.T
                t_gt = T_ned2cv @ t_gt_ned

                r_err = rotation_error_deg(R_est, R_gt)
                
                # Metric translation error (PnP recovers full scale!)
                t_est = t_est.flatten()
                t_gt = t_gt.flatten()
                
                # Direction error
                t_err_dir = translation_angle_deg(t_est, t_gt)
                
                # Absolute magnitude error in meters
                t_err_scale = np.linalg.norm(t_est - t_gt)

                rot_errors.append(r_err)
                trans_errors.append(t_err_dir)
                trans_scale_errors.append(t_err_scale)
                inlier_ratios.append(inlier_ratio)

                if pair_num % 10 == 0:
                    print(f"Pair {pair_num:04d} [{idx_a:04d}→{idx_b:04d}]  inliers: {n_inliers}/{len(pts3d)}  "
                          f"R_err: {r_err:5.2f}°  t_dir_err: {t_err_dir:5.2f}°  t_abs_err: {t_err_scale:5.2f}m")
            else:
                if pair_num % 10 == 0: print(f"Pair {pair_num:04d} PnP Failed")
        else:
             if pair_num % 10 == 0: print(f"Pair {pair_num:04d} Insufficient valid depth points ({len(pts3d)})")

    n_pairs = len(pairs)
    print("\n─── Summary (3D-2D PnP) ──────────────────────")
    print(f"  Method:      {args.method}")
    print(f"  Frame gap:   {gap}")
    print(f"  Pairs:       {n_pairs}")

    if rot_errors:
        print(f"\n  ── Accuracy (gap={gap}) ─────────────────")
        print(f"  Avg Rot Error:         {np.mean(rot_errors):.2f}°  (med: {np.median(rot_errors):.2f}°)")
        print(f"  Avg Trans Dir Error:   {np.mean(trans_errors):.2f}°  (med: {np.median(trans_errors):.2f}°)")
        print(f"  Avg Trans Abs Error:   {np.mean(trans_scale_errors):.3f} m (med: {np.median(trans_scale_errors):.3f} m)")
        print(f"  Avg Inlier Ratio:      {np.mean(inlier_ratios)*100:.1f}%")
        fail = (n_pairs - len(rot_errors)) / n_pairs * 100
        print(f"  Pose Recovery Fails:   {fail:.1f}%")

    print("\n─── Time Statistics ───────────────────────")
    print(f"  Image Reading:      {t_read_total:.1f} ms")
    print(f"  Preprocessing:      {t_prep_total:.1f} ms")
    print(f"  Keypoint Detection: {t_detect_total:.1f} ms")
    print(f"  Matching:           {t_match_total:.1f} ms")
    print(f"  PnP (Solve+Refine): {t_pnp_total:.1f} ms")
    total_time = t_read_total + t_prep_total + t_detect_total + t_match_total + t_pnp_total
    
    # Calculate pure pipeline FPS (Prep + Detect + Match + PnP) excluding IO
    pipe_time = t_prep_total + t_detect_total + t_match_total + t_pnp_total
    avg_pipe_ms = pipe_time / n_pairs if n_pairs > 0 else 0
    print(f"  Total (excl. Read): {pipe_time:.1f} ms")
    if avg_pipe_ms > 0:
        print(f"  Pipeline FPS:       {1000.0/avg_pipe_ms:.1f} FPS")

if __name__ == "__main__":
    main()
