"""
STEP 4: Predict Metric Depth From Any Image
=============================================
RUN:
    python step4_inference/predict.py --image photo.jpg --model outputs/best_model.pth
    python step4_inference/predict.py --image photo.jpg --model outputs/best_model.pth --camera camera_params.json
    python step4_inference/predict.py --input_dir photos/ --model outputs/best_model.pth
    python step4_inference/predict.py --image photo.jpg --model outputs/best_model.pth --save_pointcloud

OUTPUT:
    <name>_depth.npy          — raw depth values in meters (for computation)
    <name>_depth_colored.png  — colorized depth map (for viewing)
    <name>_comparison.png     — side-by-side RGB vs depth
    <name>_pointcloud.ply     — 3D point cloud (open in MeshLab/CloudCompare)
"""

import os, sys, json, glob, argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from step3_training.model import MetricDepthModel, LightweightDepthModel
from utils.camera_utils import depth_to_3d_points, save_pointcloud_ply
from utils.visualization import colorize_depth, create_side_by_side


def load_model(path, device, lightweight=False):
    """Load trained model from checkpoint."""
    print(f"Loading model: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    
    if lightweight:
        model = LightweightDepthModel()
    else:
        model = MetricDepthModel()
    
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(image_path, H=480, W=640):
    """Load image → resize → normalize → tensor."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    
    resized = cv2.resize(img_rgb, (W, H))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(Image.fromarray(resized))
    return tensor, img_rgb, (orig_h, orig_w)


def get_intrinsics(camera_file, orig_w, orig_h, W=640, H=480):
    """Load and rescale camera intrinsics to model input size."""
    if camera_file and os.path.exists(camera_file):
        with open(camera_file) as f:
            p = json.load(f)
        sx = W / p.get("image_width", orig_w)
        sy = H / p.get("image_height", orig_h)
        intr = np.array([p["fx"]*sx/W, p["fy"]*sy/H,
                         p["cx"]*sx/W, p["cy"]*sy/H], dtype=np.float32)
        return intr, p
    else:
        # Default NYU Kinect
        return np.array([518.8579/640, 519.4696/480,
                         325.5824/640, 253.7362/480], dtype=np.float32), None


@torch.no_grad()
def predict_depth(model, image_path, camera_file=None, device=None,
                  output_dir=None, save_pointcloud=False):
    """
    Run depth prediction on a single image.
    Returns (H, W) numpy array of depth in meters.
    """
    if device is None:
        device = next(model.parameters()).device
    
    tensor, img_rgb, (oh, ow) = preprocess(image_path)
    intr, cam_params = get_intrinsics(camera_file, ow, oh)
    
    pred = model(
        tensor.unsqueeze(0).to(device),
        torch.from_numpy(intr).unsqueeze(0).to(device)
    )
    depth = pred.squeeze().cpu().numpy()
    
    # Resize to original resolution
    depth_full = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_LINEAR)
    
    # Stats
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n  {name}: depth {depth_full.min():.2f}m – {depth_full.max():.2f}m "
          f"(mean {depth_full.mean():.2f}m)")
    
    # Save outputs
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "depth_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Raw depth
    np.save(os.path.join(output_dir, f"{name}_depth.npy"), depth_full)
    
    # Colored depth
    dc = colorize_depth(depth_full)
    cv2.imwrite(os.path.join(output_dir, f"{name}_depth_colored.png"),
                cv2.cvtColor(dc, cv2.COLOR_RGB2BGR))
    
    # Side-by-side comparison
    comp = create_side_by_side(img_rgb, dc)
    cv2.imwrite(os.path.join(output_dir, f"{name}_comparison.png"),
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    
    # 3D point cloud
    if save_pointcloud:
        if cam_params:
            fx, fy = cam_params["fx"], cam_params["fy"]
            cx, cy = cam_params["cx"], cam_params["cy"]
        else:
            fx = 518.8579 * ow / 640
            fy = 519.4696 * oh / 480
            cx = 325.5824 * ow / 640
            cy = 253.7362 * oh / 480
        
        pts, cols = depth_to_3d_points(depth_full, img_rgb, fx, fy, cx, cy)
        ply_path = os.path.join(output_dir, f"{name}_pointcloud.ply")
        save_pointcloud_ply(pts, cols, ply_path)
    
    print(f"  Saved to: {output_dir}/")
    return depth_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model", default="outputs/best_model.pth")
    parser.add_argument("--camera", default=None, help="camera_params.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--save_pointcloud", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)
    
    cam_path = args.camera
    if cam_path and not os.path.isabs(cam_path):
        cam_path = os.path.join(ROOT, cam_path)
    
    model = load_model(model_path, device, args.lightweight)
    
    if args.image:
        predict_depth(model, args.image, cam_path, device,
                      args.output_dir, args.save_pointcloud)
    elif args.input_dir:
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        print(f"Processing {len(files)} images...")
        for f in sorted(files):
            predict_depth(model, f, cam_path, device,
                         args.output_dir, args.save_pointcloud)
    else:
        print("Specify --image or --input_dir")

if __name__ == "__main__":
    main()
