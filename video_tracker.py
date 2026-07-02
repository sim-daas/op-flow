"""
Video tracker using sparse deep matching.
Reads an input video, matches consecutive frames, and outputs a video with drawn tracks.
"""
import argparse
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
    raise ValueError(f"Method {name} not supported.")

def extract_and_match(matcher, name, img1, img2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _prep(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
    if name == "xfeat":
        with torch.inference_mode():
            out1 = matcher.detectAndCompute(_prep(img1), top_k=1024)[0]
            out2 = matcher.detectAndCompute(_prep(img2), top_k=1024)[0]
            idxs0, idxs1 = matcher.match(out1["descriptors"], out2["descriptors"])
            kp1 = out1["keypoints"][idxs0].cpu().numpy()
            kp2 = out2["keypoints"][idxs1].cpu().numpy()
            return kp1, kp2
    else:
        from lightglue.utils import rbd
        ext, mat = matcher
        with torch.inference_mode():
            f1 = ext.extract(_prep(img1))
            f2 = ext.extract(_prep(img2))
            out = mat({"image0": f1, "image1": f2})
            f1, f2, out = [rbd(x) for x in [f1, f2, out]]
            kp1 = f1["keypoints"].cpu().numpy()
            kp2 = f2["keypoints"].cpu().numpy()
            matches = out["matches"].cpu().numpy()
            if len(matches) == 0:
                return np.empty((0,2)), np.empty((0,2))
            return kp1[matches[..., 0]], kp2[matches[..., 1]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video (e.g. video.mp4)")
    ap.add_argument("--out", required=True, help="Path to output video (e.g. out.mp4)")
    ap.add_argument("--method", default="xfeat", choices=["xfeat", "superpoint"])
    args = ap.parse_args()

    print(f"Loading {args.method}...")
    matcher = build_matcher(args.method)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open {args.video}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Empty video.")
        return
        
    frame_idx = 1
    print(f"Processing video ({w}x{h} @ {fps} FPS)...")
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        kp1, kp2 = extract_and_match(matcher, args.method, prev_frame, curr_frame)
        
        # Draw matches as optical flow lines on the current frame
        viz = curr_frame.copy()
        for p1, p2 in zip(kp1, kp2):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(viz, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(viz, pt2, 2, (0, 0, 255), -1)
            
        cv2.putText(viz, f"{args.method} matches: {len(kp1)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        out.write(viz)
        prev_frame = curr_frame
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")
            
    cap.release()
    out.release()
    print(f"Done. Saved to {args.out}")

if __name__ == "__main__":
    main()
