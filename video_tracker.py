"""
Video tracker using sparse deep matching.
Reads an input video, matches consecutive frames, and outputs a video with drawn tracks.
"""
import argparse
import time
import cv2
import numpy as np

from matcher_util import get_matcher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video (e.g. video.mp4)")
    ap.add_argument("--out", required=True, help="Path to output video (e.g. out.mp4)")
    ap.add_argument("--method", default="xfeat", choices=["xfeat", "superpoint", "aliked", "loftr", "roma", "silk", "dedode", "r2d2"])
    args = ap.parse_args()

    print(f"Loading {args.method}...")
    matcher = get_matcher(args.method)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open {args.video}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    
    t_read_total = 0.0
    
    t0 = time.perf_counter()
    ret, prev_frame = cap.read()
    t_read_total += (time.perf_counter() - t0) * 1000
    if not ret:
        print("Empty video.")
        return
        
    # Process first frame
    t0 = time.perf_counter()
    prev_prep = matcher.prep(prev_frame)
    prev_feat = matcher.detect(prev_prep)
    t_p, t_d, t_m = matcher.timer.get_and_reset()
    
    t_prep_total = t_p
    t_detect_total = t_d
    t_match_total = t_m
    t_viz_total = 0.0
        
    frame_idx = 1
    print(f"Processing video ({w}x{h} @ {fps} FPS)...")
    
    while True:
        t0 = time.perf_counter()
        ret, curr_frame = cap.read()
        t_read_total += (time.perf_counter() - t0) * 1000
        if not ret:
            break
            
        curr_prep = matcher.prep(curr_frame)
        curr_feat = matcher.detect(curr_prep)
        kp1, kp2 = matcher.match(prev_feat, curr_feat)
        
        t_p, t_d, t_m = matcher.timer.get_and_reset()
        t_prep_total += t_p
        t_detect_total += t_d
        t_match_total += t_m
        
        t0 = time.perf_counter()
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
        t_viz_total += (time.perf_counter() - t0) * 1000
        
        prev_feat = curr_feat
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")
            
    cap.release()
    out.release()
    print(f"Done. Saved to {args.out}")
    print("\n--- Time Statistics ---")
    print(f"Video Unpacking (Read): {t_read_total:.1f} ms")
    print(f"Preprocessing:          {t_prep_total:.1f} ms")
    print(f"Keypoint Detection:     {t_detect_total:.1f} ms")
    print(f"Matching:               {t_match_total:.1f} ms")
    print(f"Visualization & Write:  {t_viz_total:.1f} ms")
    total_time = t_read_total + t_prep_total + t_detect_total + t_match_total + t_viz_total
    print(f"Total Processing Time:  {total_time:.1f} ms")

if __name__ == "__main__":
    main()
