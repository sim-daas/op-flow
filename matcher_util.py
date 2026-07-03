import time
import cv2
import numpy as np

class TimingStats:
    def __init__(self):
        self.t_prep = 0.0
        self.t_detect = 0.0
        self.t_match = 0.0

    def reset(self):
        self.t_prep = 0.0
        self.t_detect = 0.0
        self.t_match = 0.0

    def get_and_reset(self):
        res = (self.t_prep, self.t_detect, self.t_match)
        self.reset()
        return res

class BaseMatcher:
    def __init__(self):
        self.timer = TimingStats()
        
    def prep(self, img):
        # Default: just return the image, no prep time overhead unless subclass overrides
        return img
        
    def detect(self, prep_img):
        raise NotImplementedError
        
    def match(self, feat1, feat2):
        raise NotImplementedError

    def get_keypoint_count(self, feat):
        return 0

    def match_images(self, img1, img2):
        """Helper to match two images from scratch (no caching). Useful for benchmark.py"""
        p1 = self.prep(img1)
        p2 = self.prep(img2)
        f1 = self.detect(p1)
        f2 = self.detect(p2)
        return self.match(f1, f2)

class XFeatMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1024).eval().to(self.device)
        
    def prep(self, img):
        import torch
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        p = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return p
        
    def detect(self, prep_img):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.detectAndCompute(prep_img, top_k=1024)[0]
        self.timer.t_detect += (time.perf_counter() - t0) * 1000
        return out
        
    def match(self, feat1, feat2):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            idxs0, idxs1 = self.model.match(feat1["descriptors"], feat2["descriptors"])
            kp1 = feat1["keypoints"][idxs0].cpu().numpy()
            kp2 = feat2["keypoints"][idxs1].cpu().numpy()
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        return kp1, kp2

    def get_keypoint_count(self, feat):
        return len(feat["keypoints"])

class SuperPointLightGlueMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        from lightglue import LightGlue, SuperPoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ext = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.mat = LightGlue(features="superpoint").eval().to(self.device)
        
    def prep(self, img):
        import torch
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        p = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return p
        
    def detect(self, prep_img):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            feat = self.ext.extract(prep_img)
        self.timer.t_detect += (time.perf_counter() - t0) * 1000
        return feat
        
    def match(self, feat1, feat2):
        import torch
        from lightglue.utils import rbd
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.mat({"image0": feat1, "image1": feat2})
            f1, f2, out = [rbd(x) for x in [feat1, feat2, out]]
            kp1 = f1["keypoints"].cpu().numpy()
            kp2 = f2["keypoints"].cpu().numpy()
            matches = out["matches"].cpu().numpy()
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        if len(matches) == 0:
            return np.empty((0,2)), np.empty((0,2))
        return kp1[matches[..., 0]], kp2[matches[..., 1]]

    def get_keypoint_count(self, feat):
        return len(feat["keypoints"][0])

class ALIKEDLightGlueMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        from lightglue import LightGlue, ALIKED
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ext = ALIKED(max_num_keypoints=1024, model_name="aliked-n16rot").eval().to(self.device)
        self.mat = LightGlue(features="aliked").eval().to(self.device)
        
    def prep(self, img):
        import torch
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        p = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return p
        
    def detect(self, prep_img):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            feat = self.ext.extract(prep_img)
        self.timer.t_detect += (time.perf_counter() - t0) * 1000
        return feat
        
    def match(self, feat1, feat2):
        import torch
        from lightglue.utils import rbd
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.mat({"image0": feat1, "image1": feat2})
            f1, f2, out = [rbd(x) for x in [feat1, feat2, out]]
            kp1 = f1["keypoints"].cpu().numpy()
            kp2 = f2["keypoints"].cpu().numpy()
            matches = out["matches"].cpu().numpy()
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        if len(matches) == 0:
            return np.empty((0,2)), np.empty((0,2))
        return kp1[matches[..., 0]], kp2[matches[..., 1]]

    def get_keypoint_count(self, feat):
        return len(feat["keypoints"][0])

class LoFTRMatcher(BaseMatcher):
    # LoFTR doesn't split detect and match well due to dense correlation, so we combine them
    def __init__(self):
        super().__init__()
        import torch
        import kornia.feature as KF
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KF.LoFTR(pretrained="outdoor").eval().to(self.device)
        
    def prep(self, img):
        import torch
        t0 = time.perf_counter()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        p = torch.from_numpy(g)[None, None].to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return p
        
    def detect(self, prep_img):
        # Can't separate detection for LoFTR easily
        return prep_img
        
    def match(self, feat1, feat2):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            inp = {"image0": feat1, "image1": feat2}
            corr = self.model(inp)
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        src = corr["keypoints0"].cpu().numpy()
        dst = corr["keypoints1"].cpu().numpy()
        if len(src) == 0:
            return np.empty((0,2)), np.empty((0,2))
        return src, dst

class RoMaMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        from roma import roma_outdoor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = roma_outdoor(device=self.device)
        
    def prep(self, img):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return tensor

    def detect(self, prep_img):
        return prep_img

    def match(self, feat1, feat2):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            warp, certainty = self.model.match(feat1, feat2, device=self.device)
            matches, certainty = self.model.sample(warp, certainty)
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        matches = matches.cpu().numpy()
        if len(matches) == 0:
            return np.empty((0,2)), np.empty((0,2))
        return matches[:, :2], matches[:, 2:]

class SiLKMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        from silk.models.silk import SiLK
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SiLK(default_outputs=["sparse_positions", "sparse_descriptors"]).eval().to(self.device)

    def prep(self, img):
        import torch
        t0 = time.perf_counter()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return tensor

    def detect(self, prep_img):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            feat = self.model(prep_img)
        self.timer.t_detect += (time.perf_counter() - t0) * 1000
        return feat

    def match(self, feat1, feat2):
        import torch
        t0 = time.perf_counter()
        # SiLK relies on standard mutual nearest neighbor
        kp1 = feat1[0]["sparse_positions"].cpu().numpy()
        desc1 = feat1[0]["sparse_descriptors"].cpu().numpy()
        kp2 = feat2[0]["sparse_positions"].cpu().numpy()
        desc2 = feat2[0]["sparse_descriptors"].cpu().numpy()
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        cv_matches = bf.match(desc1, desc2)
        
        m_kp1 = np.array([kp1[m.queryIdx] for m in cv_matches])
        m_kp2 = np.array([kp2[m.trainIdx] for m in cv_matches])
        
        # SiLK returns y, x instead of x, y natively
        m_kp1 = m_kp1[:, [1, 0]]
        m_kp2 = m_kp2[:, [1, 0]]
        
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        return m_kp1, m_kp2

class DeDoDeMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        from DeDoDe import dedode_detector_L, dedode_descriptor_B
        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = dedode_detector_L(weights=torch.hub.load_state_dict_from_url(
            "https://github.com/Parskatt/DeDoDe/releases/download/dedode_macost/dedode_detector_L.pth",
            map_location=self.device
        )).eval().to(self.device)
        self.descriptor = dedode_descriptor_B(weights=torch.hub.load_state_dict_from_url(
            "https://github.com/Parskatt/DeDoDe/releases/download/dedode_macost/dedode_descriptor_B.pth",
            map_location=self.device
        )).eval().to(self.device)
        self.matcher = DualSoftMaxMatcher().eval().to(self.device)

    def prep(self, img):
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # DeDoDe expects shape to be multiples of 14, but tensor input works
        tensor = transforms.ToTensor()(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return {"tensor": tensor, "shape": img.shape[:2]}

    def detect(self, prep_img):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            batch = {"image": prep_img["tensor"]}
            # detect
            keypoints, P = self.detector(batch, num_keypoints=1024)
            # describe
            batch["keypoints"] = keypoints
            descriptions = self.descriptor(batch)["descriptions"]
        self.timer.t_detect += (time.perf_counter() - t0) * 1000
        return {"keypoints": keypoints, "descriptions": descriptions, "P": P, "shape": prep_img["shape"]}

    def match(self, feat1, feat2):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            matches_A, matches_B, _ = self.matcher.match(
                feat1["keypoints"], feat1["descriptions"],
                feat2["keypoints"], feat2["descriptions"],
                P_A=feat1["P"], P_B=feat2["P"], 
                normalize=True, inv_temp=20, threshold=0.01
            )
            # convert to pixel coords
            H_A, W_A = feat1["shape"]
            H_B, W_B = feat2["shape"]
            matches_A, matches_B = self.matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        return matches_A.cpu().numpy(), matches_B.cpu().numpy()

class R2D2Matcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        import torch
        import warnings
        warnings.filterwarnings("ignore", module="torch.hub")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load from navel/r2d2 torch hub
        self.model = torch.hub.load("naver/r2d2", "r2d2", pretrained=True).eval().to(self.device)

    def prep(self, img):
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        self.timer.t_prep += (time.perf_counter() - t0) * 1000
        return tensor

    def detect(self, prep_img):
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            # R2D2 hub model provides extract_keypoints natively
            # But the standard usage for inference is to just call extract_keypoints?
            # Actually, standard usage on torch hub: `res = model(img)` returns dense
            # It's better to do dense to sparse here.
            # Usually r2d2 provides an extraction script. Let's use standard API:
            res = self.model(prep_img)
            # Just extract top-k from the dense output?
            # Since r2d2 from hub doesn't easily return N x 2 without a custom script,
            # this might crash if the API differs. 
            pass
        self.timer.t_detect += (time.perf_counter() - t0) * 1000
        return res

    def match(self, feat1, feat2):
        import torch
        t0 = time.perf_counter()
        self.timer.t_match += (time.perf_counter() - t0) * 1000
        raise NotImplementedError("R2D2 torch.hub lacks standard matcher. Use custom script.")

def get_matcher(name):
    if name == "xfeat":
        return XFeatMatcher()
    elif name in ("superpoint", "lightglue"):
        return SuperPointLightGlueMatcher()
    elif name == "aliked":
        return ALIKEDLightGlueMatcher()
    elif name == "loftr":
        return LoFTRMatcher()
    elif name == "roma":
        return RoMaMatcher()
    elif name == "silk":
        return SiLKMatcher()
    elif name == "dedode":
        return DeDoDeMatcher()
    elif name == "r2d2":
        return R2D2Matcher()
    else:
        raise ValueError(f"Unknown deep matcher: {name}")

