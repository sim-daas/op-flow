"""
Sequential benchmark for an image dataset.
Reads a directory of images, runs frame-to-frame matching, and reports average FPS and track continuity.
"""
import argparse, time, glob, os
import cv2
import numpy as np
import torch

def build_matcher(name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "xfeat":
        return torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1024).eval().to(device)
    elif name in ("superpoint", "lightglue"):
        from lightglue import LightGlue, SuperPoint
        ext = SuperPoint(max_num_keypoints=1024).eval().to(device)
        mat = LightGlue(features="superpoint").eval().to(device)
        return (ext, mat)
    raise ValueError(f"Method {name} not supported for sequence yet.")

def match_pair(matcher, name, img1, img2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _prep(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
    if name == "xfeat":
        with torch.inference_mode():
            out1 = matcher.detectAndCompute(_prep(img1), top_k=1024)[0]
            out2 = matcher.detectAndCompute(_prep(img2), top_k=1024)[0]
            idxs0, idxs1 = matcher.match(out1["descriptors"], out2["descriptors"])
            return len(idxs0)
    else:
        from lightglue.utils import rbd
        ext, mat = matcher
        with torch.inference_mode():
            f1 = ext.extract(_prep(img1))
            f2 = ext.extract(_prep(img2))
            out = mat({"image0": f1, "image1": f2})
            matches = out["matches"][0].cpu().numpy()
            return len(matches)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Directory containing sequence of images")
    ap.add_argument("--method", default="xfeat", choices=["xfeat", "superpoint"])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset, "*.*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) < 2:
        print("Not enough images found in dataset directory.")
        return

    print(f"Loading {args.method}...")
    matcher = build_matcher(args.method)
    
    print(f"Benchmarking {len(files)} frames...")
    total_ms = 0
    total_matches = 0
    
    # Warmup
    img0 = cv2.imread(files[0])
    img1 = cv2.imread(files[1])
    match_pair(matcher, args.method, img0, img1)
    
    prev_img = img0
    for i in range(1, len(files)):
        curr_img = cv2.imread(files[i])
        t0 = time.perf_counter()
        matches = match_pair(matcher, args.method, prev_img, curr_img)
        dt = (time.perf_counter() - t0) * 1000
        
        total_ms += dt
        total_matches += matches
        prev_img = curr_img
        
        if i % 10 == 0:
            print(f"Frame {i:04d} - matches: {matches:4d}, latency: {dt:.1f} ms")
            
    avg_ms = total_ms / (len(files) - 1)
    avg_matches = total_matches / (len(files) - 1)
    print("\n--- Summary ---")
    print(f"Method: {args.method}")
    print(f"Avg Latency: {avg_ms:.1f} ms ({1000/avg_ms:.1f} FPS)")
    print(f"Avg Matches: {avg_matches:.1f} per frame pair")

if __name__ == "__main__":
    main()
