# PBVS Markerless Aerial Grasping: Tracking & Benchmarking Research

## 1. Project Context
**Goal**: Achieve fully autonomous markerless aerial manipulation during the terminal grasping phase (1.5m → 0.1m).
**Hardware**: NVIDIA Jetson Orin Nano, ZED 2i Stereo Camera, Intel RealSense T265 (VIO).
**Challenge**: 15× scale expansion during descent. Features warp, blur, exit FOV. System must track without external markers or Vicon.

---

## 2. Final Architecture: Strategy B – Continuous Point Cloud PBVS

## System Architecture Evaluation

### Strategy B: Continuous Single-Map Tracking
A single, continuously growing global map of 3D anchors is superior for precision. This structure eliminates "handoff errors" (scale discontinuities when resetting tracking frames) but requires meticulous geometric and state management:

1. **Map Pollution from VIO Fallbacks**: If tracking is lost and the system falls back to VIO propagation, the camera projection matrix $P_c$ must be built exclusively from the *post-fallback* pose. If $P_c$ is built from the rejected, ill-conditioned PnP pose, any on-the-fly triangulation will project points into entirely erroneous spatial regions. As the camera moves, these "garbage anchors" will cause subsequent PnP solves to fail, creating an inescapable failure loop.
2. **Mathematical Correctness in Fallbacks**: Frame transformation algebra is unforgiving. Computing a relative pose update requires exactly matching coordinate derivations (e.g. mapping `curr \rightarrow prev` via $T_{map \rightarrow prev} \cdot (T_{map \rightarrow curr})^{-1}$). A reversed calculation silently inverts displacements, sending the camera violently off-course upon any temporary tracker failure.
3. **Solver Bounds and NaN Safety**: High-noise or co-planar point matches can cause iterative PnP solvers (like OpenCV's `solvePnPRansac`) to yield non-finite `NaN`/`Inf` vectors. These must be rigorously guarded before being passed to precision optimizers (like `scipy.optimize.least_squares`), which will crash completely on ill-posed initial bounds.
4. **Matcher Discrepancies**: While Lowe's ratio test over L2 normalized Euclidean distance works for traditional descriptors, advanced deep matchers like XFeat use internally optimized MLPs for matching. Defaulting to standard `BFMatcher` completely bottlenecks XFeat's ability to maintain high feature density under extreme viewpoint changes.

### VIO Noise Modeling (EKF Fusion Setup)
The simulation employs realistic VIO drift mimicking the T265's noise profile:
- **Random Walk Bias**: Simulated via cumulative integration of `N(0, walk_std)`.
- **High-Frequency Jitter**: Modeled with `N(0, hf_std)`.
- **Perlin Structural Noise**: Adds slow, non-linear harmonic drift simulating visual-inertial scale errors.

### Pipeline (per frame $k$)

1. **XFeat** extracts keypoints. Highest-confidence keypoint per **grid cell** (spatial bucketing) is kept.
2. **Sub-pixel refinement** (`cv2.cornerSubPix`) reduces integer-pixel quantization error before any 3D math.
3. **Frustum culling**: The T265 VIO gives a predicted camera pose. Map points are projected into the predicted FOV; only those landing in-frame are matched (avoids O(N×M) brute-force as map grows).
4. **BFMatcher** matches current-frame descriptors against culled map descriptors.
5. **PnP** (`solvePnPRansac` + `EPNP` init + LM refinement or scipy weighted refinement) gives the camera pose $T_{cam \leftarrow map}$.
6. **Sanity Filter**: $\|\Delta T_{pnp} - \Delta T_{vio}\| > \tau$ → reject PnP, propagate via VIO for this frame.
7. **Triangulation**: Unmatched keypoints are tracked across frames. When pixel displacement exceeds the **parallax gate** (15 px), they are triangulated using `cv2.triangulatePoints` with PnP-derived projection matrices and appended to the map.

---

## 3. Benchmarking Learnings (TartanAir Simulation)

### A. Coordinate Frame Trap
TartanAir poses are **NED** (X-Forward, Y-Right, Z-Down). OpenCV PnP operates in **CV** (X-Right, Y-Down, Z-Forward). Conversion:

```python
T_NED2CV = [[0,1,0],[0,0,1],[1,0,0]]
R_cv = T_NED2CV @ R_ned @ T_NED2CV.T
t_cv = T_NED2CV @ t_ned
```

### B. PnP Convention
`cv2.solvePnPRansac` returns `rvec, tvec` s.t. **`x_cam = R @ X_map + t`** (map→cam).
- Camera position in map frame: `p_cam = -R.T @ t`
- Camera-in-map 4×4: `T_cam_in_map[:3,:3] = R.T; T_cam_in_map[:3,3] = -R.T @ t`

Confusing these is the #1 cause of the "Sanity Filter rejects everything" bug.

---

## 4. Error Sources & Mitigations

| Source | Effect | Fix |
|---|---|---|
| Integer pixel quantization | 1px → ~3cm at 1.5m | `cv2.cornerSubPix` (Opt-1) |
| ZED depth edge bleed | Wildly wrong 3D anchors | 3×3 variance gate (Opt-2) |
| Clustered keypoints | PnP rotation/translation ambiguity | Grid bucketing (Opt-3) |
| Coplanar anchors (downward view) | High rotation error (>20°) due to planar ambiguity | IMU/VIO rotation prior lock |
| VIO drift | Slow pose accumulation | EKF fusing PnP + T265 (Opt-4) |
| VO drift in triangulated pts | Map scale compression after Gen-0 exits FOV | Covariance-weighted PnP + sanity filter |

---

## 5. Strategy B: Known Noise Accumulation & Mitigations

### The Feed-Forward Error Loop

```
[Gen-0 map, σ=0.001] → PnP pose (inherits σ) → Triangulation → [Gen-1, σ=0.01]
          ↑_____________________________________________|
```

Each PnP solved against Gen-N points produces a noisier Gen-(N+1). Over 3-4 generations the covariance compounds.

### Mitigations (implemented)

1. **Covariance tagging**: Gen-0 points have `cov=0.001`, Gen-1+ `cov=0.01`. The scipy weighted PnP (`--custom_pnp`) inversely weights these, forcing the solver to prefer Gen-0 anchors for as long as any remain visible.

2. **Sanity Filter**: Per-frame comparison of $\Delta T_{pnp}$ vs $\Delta T_{vio}$. Threshold (`--sanity_thresh_m`, default 0.15m) is tunable. Rejected frames propagate via VIO, not PnP noise.

3. **Wide-FOV anchor bias**: Spatial bucketing with peripheral cells keeps points near the image edges, which exit the FOV last during a vertical descent (longer Gen-0 lifespan = fewer generations = less drift).

---

## 6. VIO Simulation for EKF Validation

Real T265 drift is **not Gaussian** – it is a slow random walk with low-frequency Perlin-like wandering. Simulated as:

```
P_vio[k] = P_gt[k] + walk[k] + perlin(k) + N(0, σ_hf)
walk[k]  = walk[k-1] + N(0, σ_walk)      # compound drift
perlin(k)                                  # smooth low-freq wander
```

**Key**: `VIOSim.get(idx)` caches the result. Calling it twice for the same frame returns the same value (no double-advancing the walk). This was the critical bug in the first implementation.

Default parameters: `--walk_std 0.003 --hf_std 0.001 --perlin_mag 0.015`.

---

## 7. Script Reference

```bash
# Standard OpenCV LM refinement
python3 pbvs_tracker_sim.py --method xfeat --gap 1

# Scipy weighted PnP (Gen-0 anchors weighted 10× over Gen-1+)
python3 pbvs_tracker_sim.py --method xfeat --gap 1 --custom_pnp

# Widen sanity filter for fast maneuvers
python3 pbvs_tracker_sim.py --sanity_thresh_m 0.3

# Simulate noisier VIO (older T265 hardware)
python3 pbvs_tracker_sim.py --walk_std 0.008 --perlin_mag 0.03
```

---

## 8. ROS2 Node Plan (Next Step)

The `GlobalMap` class in `pbvs_tracker_sim.py` maps 1:1 to a ROS2 node:
- **Subscriptions**: `/zed/rgb/image`, `/zed/depth/image`, `/t265/odom`
- **Publications**: `/pbvs/target_pose` (PnP result), `/pbvs/map_debug` (PointCloud2)
- **Services**: `/pbvs/reset_map` (trigger at 1.5m)
- The sanity filter threshold is a `rclpy` parameter, tunable live via `ros2 param set`.
