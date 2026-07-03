# PBVS Markerless Aerial Grasping: Tracking & Benchmarking Research

## 1. Project Context
**Goal**: Achieve fully autonomous markerless aerial manipulation using a quadrotor during the terminal grasping phase (1.5m down to 0.1m).
**Hardware**: NVIDIA Jetson Orin Nano, ZED 2i Stereo Camera, Intel RealSense T265 (VIO).
**Challenge**: As the drone descends, a massive 15x scale expansion occurs. Features warp, blur, and exit the FOV. The system must track spatial anchors reliably without external markers or Vicon.

---

## 2. The PBVS Pipeline (Keyframe-Based Map Expansion)

The core mechanism for maintaining sub-centimeter accuracy is **Anchor-Based Localization**. 

### The Pipeline Steps:
1. **Anchor Initialization ($t_0$)**: At 1.5m, the drone captures an image. We extract 2D keypoints (XFeat) and use the ZED depth map to instantly back-project them into rigid 3D spatial anchors.
2. **Direct Tracking ($t_k$)**: As the drone moves, we do NOT compute frame-to-frame optical flow. Instead, we match the current frame $t_k$ directly against the original Anchor $t_0$. 
3. **Pose Estimation (PnP)**: We solve the 3D-2D Perspective-n-Point problem (`cv2.solvePnPRansac`).
4. **Anchor Handoff (Map Expansion)**: Once the drone is too close (e.g., at 0.5m) and the original $t_0$ features expand beyond the camera's FOV, we establish a new Anchor $t_{new}$ and compute its pose relative to $t_0$.

### Why This Prevents Drift
By tracking directly against $t_0$, the error is **bounded** to the immediate pixel noise of that frame. If you use frame-to-frame tracking (e.g., matching $t_1 \to t_2$, then $t_2 \to t_3$), you suffer from **random walk drift** (errors accumulate). Your main task is indeed holding onto the Anchor keyframe for as long as visually possible.

---

## 3. Benchmarking Learnings (TartanAir Simulation)

We developed a rigorous testing suite (`sequence_benchmark.py` and `pnp_benchmark.py`) to validate the visual pipeline using the TartanAir dataset (which provides exact depth maps and ground-truth poses).

### A. Coordinate Frame Traps
Our initial pose evaluations yielded impossible rotational errors (15°+) on adjacent frames. 
**Learning**: TartanAir provides poses in the **NED (North-East-Down)** frame. OpenCV's `solvePnP` and `recoverPose` expect the standard **Camera Frame (X-Right, Y-Down, Z-Forward)**. Furthermore, OpenCV computes the pose of Camera 1 relative to Camera 2, which is mathematically the inverse of Camera 2 to Camera 1.
**Fix**: We implemented a strict $T_{ned \to cv}$ transformation and inverted the ground-truth relative pose ($R_{gt} = R_b^T R_a$), dropping our baseline rotation error to ~1°.

### B. 2D-2D Epipolar vs. 3D-2D PnP
- **Essential Matrix (2D-2D)**: Works terribly at small baselines (gap=1). If the camera hovers or barely translates, the epipolar geometry becomes degenerate. The translation vector is completely unreliable (showing 30°+ directional error).
- **PnP (3D-2D)**: By utilizing the depth map, PnP recovers absolute metric translation and locks in the scale. 

### C. SuperPoint vs. XFeat
On synthetic factory datasets, both models detected the exact same geometric corners. XFeat proved mathematically equivalent in pose accuracy but runs an order of magnitude faster (1900+ FPS on GPU vs SuperPoint's 50 FPS). For the Orin Nano, XFeat is the definitive choice.

---

## 4. Where Does The Error Come From? (Optimization Points)

In the current `pnp_benchmark.py`, we see absolute translation errors around 10-15cm. Why?

1. **Pixel Quantization**: CNN extractors output keypoints at integer pixel coordinates. At 1.5m away, 1 pixel of deviation can mean 3cm of spatial error in 3D. 
2. **Depth Map Noise**: In the simulation, depth is perfect. In reality (ZED 2i), depth edges bleed. If a keypoint lands on an edge (e.g., the edge of a pipe), the stereo depth might jump by 0.5m, completely destroying the PnP solver.
3. **Ill-Conditioned Geometry**: If all your matched keypoints are clustered in the center of the image (e.g., only on the target object), PnP cannot mathematically distinguish between a small camera rotation and a small camera translation. 

## 5. How to Make It "Perfect" for Real-World PBVS

To bridge the gap from this general-purpose benchmark to your real-world grasping pipeline:

- **Optimization 1 (Sub-pixel Refinement)**: Pass the XFeat keypoints through `cv2.cornerSubPix`. This will refine the integer pixel coordinates into floating-point coordinates, drastically reducing the 3D back-projection error.
- **Optimization 2 (Depth Variance Filtering)**: Never trust a ZED depth value blindly. Sample a 3x3 pixel window around the keypoint in the depth map. If the variance is high (it's on an edge), discard the keypoint. Only back-project keypoints that sit on flat, continuous surfaces.
- **Optimization 3 (Spatial Bucketing)**: Use Adaptive Non-Maximal Suppression (ANMS) or a simple grid bucket to force keypoints to be evenly distributed across the entire FOV. A wide spread of points geometrically locks the PnP solver and prevents translation/rotation ambiguity.
- **Optimization 4 (EKF Fusion)**: The RealSense T265 provides 200Hz VIO. PnP provides absolute 10Hz Anchor poses. You must fuse them. The T265 handles the high-frequency inertial bumps; the PnP visually corrects the T265's slow drift.
