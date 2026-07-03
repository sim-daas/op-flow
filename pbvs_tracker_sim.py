"""
pbvs_tracker_sim.py  –  Strategy B: continuous point-cloud PBVS simulator.

Map frame = OpenCV camera frame at t=0.
PnP solves R,t s.t. x_cam = R @ X_map + t  →  cam-in-map = -R.T @ t.
VIO is simulated: random walk + low-freq Perlin (or sine fallback) + HF Gaussian.
"""
import argparse, glob, os, time
import cv2
import numpy as np

try:
    from perlin_noise import PerlinNoise as _PN
    _HAVE_PERLIN = True
except ImportError:
    _HAVE_PERLIN = False

try:
    import scipy.optimize as opt
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from matcher_util import get_matcher
from sequence_benchmark import load_poses, rotation_error_deg, translation_angle_deg

# ── coordinate helpers ─────────────────────────────────────────────────────────
_T_NED2CV = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=float)

def ned_to_cv(T):
    R = _T_NED2CV @ T[:3,:3] @ _T_NED2CV.T
    t = _T_NED2CV @ T[:3, 3]
    out = np.eye(4); out[:3,:3] = R; out[:3,3] = t
    return out

def read_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None if img is None else img.view(np.float32).squeeze(-1)

def depth_at(depth_img, u, v, var_thresh=0.05, max_z=20.0):
    x, y = int(round(u)), int(round(v))
    if x < 1 or x >= depth_img.shape[1]-1 or y < 1 or y >= depth_img.shape[0]-1:
        return None
    win = depth_img[y-1:y+2, x-1:x+2]
    win = win[win > 0]
    if len(win) < 5 or np.var(win) > var_thresh:
        return None
    z = np.median(win)
    return z if 0.1 < z < max_z else None

def xfeat_match(desc1, desc2, matcher):
    import torch
    with torch.inference_mode():
        d1 = torch.from_numpy(desc1).to(matcher.device)
        d2 = torch.from_numpy(desc2).to(matcher.device)
        idx0, idx1 = matcher.model.match(d1, d2)
    class DummyMatch:
        def __init__(self, q, t):
            self.queryIdx = q
            self.trainIdx = t
            self.distance = 0.0
    return [DummyMatch(q, t) for q, t in zip(idx0.cpu().numpy(), idx1.cpu().numpy())]

def bucket(kps, scores, descs, cell):
    """Keep highest-score keypoint per grid cell."""
    best = {}
    sc = scores if scores is not None else np.ones(len(kps))
    for kp, s, d in zip(kps, sc, descs):
        c = (int(kp[0]//cell), int(kp[1]//cell))
        if c not in best or s > best[c][1]:
            best[c] = (kp, s, d)
    kps_  = np.array([v[0] for v in best.values()], dtype=np.float32)
    descs_= np.array([v[2] for v in best.values()])
    return kps_, descs_

def proj_matrix(K, R, t):
    """P = K @ [R | t]  (t is 3×1, map→cam)"""
    return K @ np.hstack([R, t.reshape(3,1)])

# ── VIO simulator ──────────────────────────────────────────────────────────────
class VIOSim:
    """Caches noisy poses per frame-index so each index is advanced exactly once."""
    def __init__(self, poses_cv, walk_std, hf_std, perlin_scale, perlin_mag, seed=42):
        np.random.seed(seed)
        self.poses = poses_cv
        self.walk_std, self.hf_std, self.pmag = walk_std, hf_std, perlin_mag
        if _HAVE_PERLIN:
            self._pn = [_PN(octaves=perlin_scale, seed=seed+i) for i in range(6)]
        else:
            self._phase = np.random.rand(6) * 2 * np.pi  # ponytail: sine fallback
        self._cache = {}
        self._walk  = np.zeros(6)
        self._prev  = -1

    def _lf(self, idx):
        if _HAVE_PERLIN:
            return np.array([p(idx*0.01) for p in self._pn]) * self.pmag
        return np.sin(idx*0.05 + self._phase) * self.pmag  # ponytail: sine fallback

    def get(self, idx):
        if idx in self._cache: return self._cache[idx]
        if idx >= len(self.poses):  return None
        # advance walk sequentially to avoid double-advancing
        for i in range(self._prev+1, idx+1):
            self._walk += np.random.normal(0, self.walk_std, 6)
        self._prev = idx
        noise = self._walk + self._lf(idx) + np.random.normal(0, self.hf_std, 6)
        T = self.poses[idx].copy()
        T[:3,3] += noise[:3]
        R_n, _ = cv2.Rodrigues(noise[3:])
        T[:3,:3] = R_n @ T[:3,:3]
        self._cache[idx] = T
        return T

# ── weighted PnP refinement (scipy) ───────────────────────────────────────────
def weighted_pnp(pts3d, pts2d, covs, K, dist, rvec0, tvec0):
    w = 1.0 / np.clip(covs, 1e-6, None); w /= w.max()
    def res(p):
        proj, _ = cv2.projectPoints(pts3d, p[:3], p[3:], K, dist)
        return ((proj.reshape(-1,2) - pts2d) * w[:,None]).flatten()
    r = opt.least_squares(res, np.r_[rvec0.flatten(), tvec0.flatten()], method='lm', max_nfev=50)
    return r.x[:3].reshape(3,1), r.x[3:].reshape(3,1)

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="dataset/AbandonedFactory/Data_hard/P000/image_lcam_front")
    ap.add_argument("--depth",    default="dataset/AbandonedFactory/Data_hard/P000/depth_lcam_front")
    ap.add_argument("--poses",    default="dataset/AbandonedFactory/Data_hard/P000/pose_lcam_front.txt")
    ap.add_argument("--method",   default="xfeat")
    ap.add_argument("--gap",      type=int,   default=1)
    ap.add_argument("--max_frames",type=int,  default=None)
    # camera intrinsics  (TartanAir: 640×640, f=320, cx=cy=320)
    ap.add_argument("--focal",    type=float, default=320.0)
    ap.add_argument("--cx",       type=float, default=320.0)
    ap.add_argument("--cy",       type=float, default=320.0)
    ap.add_argument("--max_depth",type=float, default=20.0)
    # pipeline knobs
    ap.add_argument("--subpixel_win",     type=int,   default=5)
    ap.add_argument("--depth_var_thresh", type=float, default=0.2,
                    help="Depth 3x3 window variance gate (m^2). Lower=stricter. 0.05 for real ZED2i, 0.2 for sim.")
    ap.add_argument("--grid_size",        type=int,   default=30,
                    help="Spatial bucket cell size (px). Smaller=denser map.")
    ap.add_argument("--parallax_px",      type=float, default=10.0)
    ap.add_argument("--reproj_err",       type=float, default=4.0)
    ap.add_argument("--ransac_iters",     type=int,   default=200)
    ap.add_argument("--min_inliers",      type=int,   default=4)
    ap.add_argument("--ratio_thresh",     type=float, default=0.75,
                    help="Lowe ratio test threshold for map→frame matching")
    ap.add_argument("--sanity_thresh_m",  type=float, default=0.15,
                    help="Max PnP-VIO delta translation discrepancy (m)")
    ap.add_argument("--custom_pnp", action="store_true",
                    help="Use scipy weighted PnP; falls back to OpenCV LM if scipy absent")
    # VIO noise knobs
    ap.add_argument("--walk_std",    type=float, default=0.003)
    ap.add_argument("--hf_std",      type=float, default=0.001)
    ap.add_argument("--perlin_scale",type=float, default=4.0)
    ap.add_argument("--perlin_mag",  type=float, default=0.015)
    a = ap.parse_args()

    poses_ned = load_poses(a.poses)
    poses_cv  = [ned_to_cv(p) for p in poses_ned]
    print(f"Loaded {len(poses_cv)} GT poses.")

    imgs   = sorted(f for f in glob.glob(os.path.join(a.dataset,"*.*"))
                    if f.lower().endswith(('.png','.jpg','.jpeg')))
    depths = sorted(glob.glob(os.path.join(a.depth,"*.png")))
    N = min(len(imgs), len(depths), len(poses_cv))
    if a.max_frames: N = min(N, a.max_frames)
    frame_idxs = list(range(0, N, a.gap))
    print(f"Frames to process: {len(frame_idxs)}")

    K    = np.array([[a.focal,0,a.cx],[0,a.focal,a.cy],[0,0,1]], dtype=float)
    dist = np.zeros(4)
    vio  = VIOSim(poses_cv, a.walk_std, a.hf_std, a.perlin_scale, a.perlin_mag)
    matcher = get_matcher(a.method)

    # warmup
    matcher.match_images(cv2.imread(imgs[0]), cv2.imread(imgs[min(a.gap, N-1)]))
    matcher.timer.reset()

    bf = cv2.BFMatcher(cv2.NORM_L2)  # kNN + ratio test; crossCheck rejects too many at large viewpoint changes
    criteria_sp = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # global map
    map_pts  = []   # list of (3,) float32
    map_desc = []   # list of descriptor
    map_cov  = []   # list of scalar uncertainty

    # triangulation state (previous frame unmatched features)
    prev_unm_kp   = None
    prev_unm_desc = None
    prev_P        = None   # projection matrix of previous frame (map→cam)
    prev_fidx     = None
    T_vio_0       = None   # VIO pose at frame 0 = map-frame-in-world

    # pose tracking
    T_pnp_prev = np.eye(4)   # cam-in-map starts at identity (map frame = cam-0 frame)

    rot_errs, trans_errs, scale_errs, inl_ratios = [], [], [], []
    n_pnp_attempted = n_pnp_ok = n_sanity_rej = n_triag = 0
    t_match_ms = t_pnp_ms = 0.0

    use_weighted = a.custom_pnp and _HAVE_SCIPY

    print(f"\nPnP mode: {'scipy-weighted' if use_weighted else 'opencv-lm'} | "
          f"VIO: {'perlin' if _HAVE_PERLIN else 'sine-fallback'}\n")

    for fi, idx in enumerate(frame_idxs):
        img  = cv2.imread(imgs[idx])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        feat = matcher.detect(matcher.prep(img))
        kps   = feat["keypoints"].cpu().numpy()
        descs = feat["descriptors"].cpu().numpy()
        scores = feat["scores"].cpu().numpy() if "scores" in feat else None

        # kps, descs = bucket(kps, scores, descs, a.grid_size)
        if not len(kps): continue

        # sub-pixel refinement
        kps = cv2.cornerSubPix(gray, kps.copy(), (a.subpixel_win,)*2, (-1,-1), criteria_sp)

        # ── Init frame 0 ──────────────────────────────────────────────────────
        if fi == 0:
            depth = read_depth(depths[idx])
            for kp, d in zip(kps, descs):
                Z = depth_at(depth, kp[0], kp[1], a.depth_var_thresh, a.max_depth)
                if Z is None: continue
                # ponytail: no artificial noise here; TartanAir depth is GT.
                # For real ZED2i, enable: Z *= np.random.normal(1.0, 0.03)
                map_pts.append([(kp[0]-a.cx)*Z/a.focal, (kp[1]-a.cy)*Z/a.focal, Z])
                map_desc.append(d)
                map_cov.append(0.001)   # Gen-0: lowest uncertainty
            print(f"Frame {idx:04d}: Gen-0 map initialized with {len(map_pts)} anchors.")
            T_vio_0 = vio.get(idx)   # cache: map-frame-in-world (noisy cam-0 pose)
            prev_P = proj_matrix(K, np.eye(3), np.zeros((3,1)))
            prev_unm_kp   = kps.copy()
            prev_unm_desc = descs.copy()
            prev_fidx = idx
            continue

        pts3d_map = np.array(map_pts,  dtype=np.float32)
        desc_map  = np.array(map_desc, dtype=np.float32)
        cov_map   = np.array(map_cov,  dtype=np.float32)

        # ── Frustum culling via VIO ───────────────────────────────────────────
        # Map pts are in cam-0 frame. Need T: map(cam-0) → cam-k.
        # T_vio_0 = cam-0-in-world, T_vio_k = cam-k-in-world
        # map→cam-k = inv(T_vio_k) @ T_vio_0
        T_vio = vio.get(idx)
        if T_vio is None: continue
        T_map2cam = np.linalg.inv(T_vio) @ T_vio_0   # map→cam-k
        R_m2c = T_map2cam[:3,:3]
        t_m2c = T_map2cam[:3, 3:]

        pts_c = (R_m2c @ pts3d_map.T).T + t_m2c.T
        ok_z  = pts_c[:,2] > 0.05
        if not ok_z.any(): continue

        proj2 = (K @ pts_c[ok_z].T).T
        proj2 = proj2[:,:2] / proj2[:,2:]
        h, w  = img.shape[:2]
        in_fov = (proj2[:,0]>=0)&(proj2[:,0]<w)&(proj2[:,1]>=0)&(proj2[:,1]<h)
        cull_idx = np.where(ok_z)[0][in_fov]
        if len(cull_idx) < a.min_inliers: continue

        c_desc  = desc_map[cull_idx]
        c_pts3d = pts3d_map[cull_idx]
        c_covs  = cov_map[cull_idx]

        # ── Match & PnP ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if a.method == "xfeat":
            good = xfeat_match(c_desc, descs, matcher)
        else:
            knn = bf.knnMatch(c_desc, descs, k=2)
            good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < a.ratio_thresh * pair[1].distance]
        t_match_ms += (time.perf_counter()-t0)*1000

        if fi <= 2 and len(good) > 0:  # brief startup diagnostic
            print(f"  [dbg] frame={idx} culled={len(cull_idx)} good_matches={len(good)}")

        n_pnp_attempted += 1
        if len(good) < a.min_inliers: continue

        m3d = np.array([c_pts3d[m.queryIdx] for m in good], dtype=np.float32)
        m2d = np.array([kps[m.trainIdx]     for m in good], dtype=np.float32)
        mcv = np.array([c_covs[m.queryIdx]  for m in good], dtype=np.float32)

        t0 = time.perf_counter()
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            m3d, m2d, K, dist,
            reprojectionError=a.reproj_err, iterationsCount=a.ransac_iters)
        t_pnp_ms += (time.perf_counter()-t0)*1000

        if not ok or inliers is None or len(inliers) < a.min_inliers: continue
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)): continue
        
        inl = inliers.flatten()
        if use_weighted:
            rvec, tvec = weighted_pnp(m3d[inl], m2d[inl], mcv[inl], K, dist, rvec, tvec)
        else:
            rvec, tvec = cv2.solvePnPRefineLM(m3d[inl], m2d[inl], K, dist, rvec, tvec)

        # PnP returns R,t s.t. x_cam = R @ X_map + t
        R_est, _ = cv2.Rodrigues(rvec)
        # cam-in-map = [-R.T | -R.T @ t]
        T_cam_in_map = np.eye(4)
        T_cam_in_map[:3,:3] = R_est.T
        T_cam_in_map[:3, 3] = (-R_est.T @ tvec).flatten()

        # ── Sanity filter: compare |delta_PnP| vs |delta_VIO| ───────────────
        # Translation norms are frame-invariant; avoids world vs map frame mismatch.
        T_vio_prev = vio.get(prev_fidx)
        delta_vio  = np.linalg.inv(T_vio_prev) @ T_vio        # world frame
        delta_pnp  = np.linalg.inv(T_pnp_prev) @ T_cam_in_map  # map frame
        t_pnp_norm = np.linalg.norm(delta_pnp[:3,3])
        t_vio_norm = np.linalg.norm(delta_vio[:3,3])
        t_diff = abs(t_pnp_norm - t_vio_norm)
        
        delta_final = delta_pnp
        if t_diff > a.sanity_thresh_m:
            n_sanity_rej += 1
            # Propagate via VIO: apply VIO relative displacement
            T_cam_in_map = T_pnp_prev @ delta_vio
            delta_final = delta_vio
        
        T_pnp_prev = T_cam_in_map.copy()

        # Guard: if LM refinement produced NaN/inf, skip this frame
        if not np.all(np.isfinite(T_cam_in_map)):
            prev_fidx = idx; continue

        R_final = delta_final[:3,:3]
        try:
            U, _, Vt = np.linalg.svd(R_final); R_final_orth = U @ Vt
        except np.linalg.LinAlgError:
            prev_fidx = idx; continue

        # ── Accuracy vs GT ────────────────────────────────────────────────────
        # GT poses are cam-in-world. Convert to cam-in-map: T_cam_in_map_gt = inv(poses_cv[0]) @ poses_cv[k]
        # delta in map frame: inv(T_cam_map_prev) @ T_cam_map_curr
        T_gt_prev_in_map = np.linalg.inv(poses_cv[0]) @ poses_cv[prev_fidx]
        T_gt_curr_in_map = np.linalg.inv(poses_cv[0]) @ poses_cv[idx]
        delta_gt = np.linalg.inv(T_gt_prev_in_map) @ T_gt_curr_in_map

        rot_errs.append(rotation_error_deg(R_final_orth, delta_gt[:3,:3]))
        trans_errs.append(translation_angle_deg(delta_final[:3,3], delta_gt[:3,3]))
        scale_errs.append(np.linalg.norm(delta_final[:3,3] - delta_gt[:3,3]))
        inl_ratios.append(len(inl)/len(good))
        n_pnp_ok += 1

        if fi % 20 == 0:
            print(f"Frame {idx:04d}: inliers={len(inl)}/{len(good)}  "
                  f"R_err={rot_errs[-1]:.2f}°  t_err={scale_errs[-1]*100:.1f}cm  "
                  f"map={len(map_pts)}  sanity_rej={n_sanity_rej}")

        # ── Triangulate new points ────────────────────────────────────────────
        # Recompute map→cam R, t from the final T_cam_in_map (in case VIO fallback was used)
        R_final_cam = T_cam_in_map[:3, :3].T
        t_final_cam = -R_final_cam @ T_cam_in_map[:3, 3]
        P_curr = proj_matrix(K, R_final_cam, t_final_cam)

        matched_train = {m.trainIdx for m in good}
        unm_mask = np.array([i not in matched_train for i in range(len(kps))])
        unm_kp   = kps[unm_mask]
        unm_desc = descs[unm_mask]

        if prev_unm_kp is not None and len(prev_unm_kp) and len(unm_kp):
            if a.method == "xfeat":
                nm = xfeat_match(prev_unm_desc, unm_desc, matcher)
            else:
                nm_knn = bf.knnMatch(prev_unm_desc, unm_desc, k=2)
                nm = [p[0] for p in nm_knn if len(p) == 2 and p[0].distance < a.ratio_thresh * p[1].distance]
                
            new_pts, new_descs, new_covs = [], [], []
            for m in nm:
                p1, p2 = prev_unm_kp[m.queryIdx], unm_kp[m.trainIdx]
                if np.linalg.norm(p1 - p2) < a.parallax_px: continue
                Xh = cv2.triangulatePoints(prev_P, P_curr, p1.reshape(2,1), p2.reshape(2,1))
                w  = float(Xh[3, 0])
                if abs(w) < 1e-6: continue           # point at infinity
                X  = (Xh[:3, 0] / w)
                if not np.all(np.isfinite(X)): continue
                X_c = P_curr @ np.append(X, 1)
                if X_c[2] <= 0 or X_c[2] > a.max_depth: continue
                new_pts.append(X); new_descs.append(unm_desc[m.trainIdx])
                new_covs.append(0.01)
            if new_pts:
                map_pts.extend(new_pts); map_desc.extend(new_descs)
                map_cov.extend(new_covs); n_triag += len(new_pts)

        prev_unm_kp, prev_unm_desc, prev_P, prev_fidx = unm_kp, unm_desc, P_curr, idx

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n─── Strategy B PBVS Tracker Summary ────────────────────────────────")
    print(f"  Method:        {a.method} | PnP: {'scipy-weighted' if use_weighted else 'opencv-lm'}")
    print(f"  VIO noise:     {'perlin' if _HAVE_PERLIN else 'sine'} | "
          f"walk={a.walk_std} hf={a.hf_std} pmag={a.perlin_mag}")
    print(f"  PnP attempts:  {n_pnp_attempted} | OK: {n_pnp_ok} | "
          f"Sanity rejects: {n_sanity_rej}")
    print(f"  Final map:     {len(map_pts)} pts  "
          f"(Gen-0: {sum(1 for c in map_cov if c<=0.001)} | "
          f"Triangulated: {n_triag})")

    if rot_errs:
        print(f"\n  ── Accuracy ─────────────────────────────────────────────")
        print(f"  Avg Rot Error:       {np.mean(rot_errs):.2f}°  "
              f"(med {np.median(rot_errs):.2f}°)")
        print(f"  Avg Trans Dir Error: {np.mean(trans_errs):.2f}°  "
              f"(med {np.median(trans_errs):.2f}°)")
        print(f"  Avg Trans Abs Error: {np.mean(scale_errs)*100:.1f}cm  "
              f"(med {np.median(scale_errs)*100:.1f}cm)")
        print(f"  Avg Inlier Ratio:    {np.mean(inl_ratios)*100:.1f}%")
        print(f"  Pose recovery rate:  {n_pnp_ok/max(n_pnp_attempted,1)*100:.1f}%")
    else:
        print("\n  No successful PnP frames – check dataset paths and camera params.")

    print(f"\n  ── Timing ───────────────────────────────────────────────────")
    n = max(n_pnp_attempted, 1)
    print(f"  Matching:  {t_match_ms:.0f}ms total  ({t_match_ms/n:.1f}ms/frame)")
    print(f"  PnP:       {t_pnp_ms:.0f}ms total  ({t_pnp_ms/n:.1f}ms/frame)")

if __name__ == "__main__":
    main()
