"""
PyTorch Dataset for Metric Depth Estimation
============================================
Loads RGB images + ground truth depth maps for training.
Handles intrinsic rescaling, normalization, and augmentation.
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import json


class DepthDataset(Dataset):
    """
    Returns per sample:
        rgb         : (3, H, W) tensor, ImageNet-normalized
        depth       : (1, H, W) tensor, meters
        mask        : (1, H, W) tensor, 1 = valid depth
        intrinsics  : (4,) tensor [fx/W, fy/H, cx/W, cy/H] normalized
    """
    
    def __init__(self, data_dir, split="train",
                 image_height=480, image_width=640,
                 max_depth=10.0, min_depth=0.1,
                 camera_params=None, augment=True):
        
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")
        self.H = image_height
        self.W = image_width
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.augment = augment and (split == "train")
        
        # Load file list
        split_file = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.ids = [l.strip() for l in f if l.strip()]
        else:
            self.ids = sorted(
                f.replace('.png', '').replace('.jpg', '')
                for f in os.listdir(self.rgb_dir)
                if f.endswith(('.png', '.jpg'))
            )
            n = int(len(self.ids) * 0.8)
            self.ids = self.ids[:n] if split == "train" else self.ids[n:]
        
        # Camera intrinsics — default to NYU Kinect
        if camera_params is None:
            camera_params = {
                "fx": 518.8579, "fy": 519.4696,
                "cx": 325.5824, "cy": 253.7362,
                "image_width": 640, "image_height": 480
            }
        
        # CRITICAL: Rescale intrinsics when image is resized
        sx = image_width / camera_params["image_width"]
        sy = image_height / camera_params["image_height"]
        self.fx = camera_params["fx"] * sx
        self.fy = camera_params["fy"] * sy
        self.cx = camera_params["cx"] * sx
        self.cy = camera_params["cy"] * sy
        
        # Normalized intrinsics for feeding into the neural network
        self.intrinsics = np.array([
            self.fx / image_width,
            self.fy / image_height,
            self.cx / image_width,
            self.cy / image_height
        ], dtype=np.float32)
        
        # Standard ImageNet normalization
        self.rgb_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        print(f"  [{split}] {len(self.ids)} samples | "
              f"{image_width}x{image_height} | "
              f"depth [{min_depth}, {max_depth}]m")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        fid = self.ids[idx]
        
        # Load RGB
        rgb_path = os.path.join(self.rgb_dir, f"{fid}.png")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.rgb_dir, f"{fid}.jpg")
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.W, self.H))
        
        # Load depth
        depth = np.load(os.path.join(self.depth_dir, f"{fid}.npy")).astype(np.float32)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        
        # Augmentation
        if self.augment:
            rgb, depth = self._augment(rgb, depth)
        
        # Validity mask and clipping
        mask = ((depth > self.min_depth) & (depth < self.max_depth)).astype(np.float32)
        depth = np.clip(depth, self.min_depth, self.max_depth)
        
        # To tensors
        rgb_t = self.rgb_transform(Image.fromarray(rgb))
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        intr_t = torch.from_numpy(self.intrinsics.copy())
        
        return {"rgb": rgb_t, "depth": depth_t, "mask": mask_t,
                "intrinsics": intr_t, "filename": fid}
    
    def _augment(self, rgb, depth):
        """Training-time augmentation applied identically to RGB and depth."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
        
        # Color jitter (RGB only — depth is not affected by lighting)
        if np.random.random() > 0.5:
            rgb = rgb.astype(np.float32)
            rgb *= np.random.uniform(0.8, 1.2)  # brightness
            mean = rgb.mean()
            rgb = (rgb - mean) * np.random.uniform(0.85, 1.15) + mean  # contrast
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # Random crop + resize back
        if np.random.random() > 0.5:
            h, w = rgb.shape[:2]
            ch = int(h * np.random.uniform(0.85, 1.0))
            cw = int(w * np.random.uniform(0.85, 1.0))
            y = np.random.randint(0, h - ch + 1)
            x = np.random.randint(0, w - cw + 1)
            rgb = cv2.resize(rgb[y:y+ch, x:x+cw], (self.W, self.H))
            depth = cv2.resize(depth[y:y+ch, x:x+cw], (self.W, self.H),
                             interpolation=cv2.INTER_NEAREST)
        
        return rgb, depth


def create_dataloaders(config):
    """Build train and val DataLoaders from config dict."""
    
    cam = config.get("camera", {})
    ds = config.get("dataset", {})
    
    # Load camera params from file or config
    camera_params = None
    if cam.get("params_file"):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(project_root, cam["params_file"])) as f:
            camera_params = json.load(f)
    else:
        camera_params = {
            "fx": cam.get("fx", 518.8579), "fy": cam.get("fy", 519.4696),
            "cx": cam.get("cx", 325.5824), "cy": cam.get("cy", 253.7362),
            "image_width": ds.get("image_width", 640),
            "image_height": ds.get("image_height", 480),
        }
    
    data_dir = ds.get("data_dir", "data/nyu_depth_v2")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
    
    kwargs = dict(
        data_dir=data_dir,
        image_height=ds.get("image_height", 480),
        image_width=ds.get("image_width", 640),
        max_depth=ds.get("max_depth", 10.0),
        min_depth=ds.get("min_depth", 0.1),
        camera_params=camera_params,
    )
    
    bs = config.get("training", {}).get("batch_size", 8)
    
    train_ds = DepthDataset(split="train", augment=True, **kwargs)
    val_ds   = DepthDataset(split="val",   augment=False, **kwargs)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
