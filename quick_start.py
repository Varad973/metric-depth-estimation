"""
QUICK START — Verify Setup + Run Demo
=======================================
Run this FIRST to make sure everything is installed correctly.

    python quick_start.py

It will:
    1. Check all dependencies are installed
    2. Test camera calibration (database mode)
    3. Create a tiny synthetic dataset
    4. Run one training epoch
    5. Predict depth on a synthetic image
    6. Print summary
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """Verify all required packages are installed."""
    print("=" * 55)
    print(" STEP 0: Checking dependencies")
    print("=" * 55)
    
    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV (opencv-python)",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
        "timm": "timm",
        "matplotlib": "matplotlib",
        "h5py": "h5py",
        "scipy": "scipy",
        "pandas": "pandas",
        "tabulate": "tabulate",
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing.append(name)
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"\n  GPU: None (CPU only — training will be slow)")
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    
    print(f"\n  All dependencies OK!")
    return True


def test_camera_calibration():
    """Test camera database lookup."""
    print("\n" + "=" * 55)
    print(" STEP 1: Camera Calibration (database test)")
    print("=" * 55)
    
    from step1_camera_calibration.calibrate_camera import get_from_database
    
    params = get_from_database("nyu_kinect")
    print(f"  Camera: {params['source']}")
    print(f"  fx={params['fx']:.2f}  fy={params['fy']:.2f}")
    print(f"  cx={params['cx']:.2f}  cy={params['cy']:.2f}")
    
    import json
    out_path = os.path.join(PROJECT_ROOT, "camera_params.json")
    with open(out_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {out_path}")
    return True


def create_synthetic_dataset():
    """Create a tiny fake dataset for testing the pipeline."""
    print("\n" + "=" * 55)
    print(" STEP 2: Creating synthetic test dataset")
    print("=" * 55)
    
    import numpy as np
    from PIL import Image
    
    data_dir = os.path.join(PROJECT_ROOT, "data", "test_synthetic")
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    n_samples = 20
    H, W = 480, 640
    
    for i in range(n_samples):
        # Create synthetic RGB (random colored rectangles)
        img = np.random.randint(100, 200, (H, W, 3), dtype=np.uint8)
        
        # Add some "objects" at different depths
        n_objects = np.random.randint(3, 8)
        depth = np.full((H, W), 8.0, dtype=np.float32)  # background at 8m
        
        for _ in range(n_objects):
            # Random rectangle
            y1 = np.random.randint(0, H - 50)
            x1 = np.random.randint(0, W - 50)
            rh = np.random.randint(40, 200)
            rw = np.random.randint(40, 250)
            y2 = min(y1 + rh, H)
            x2 = min(x1 + rw, W)
            
            obj_depth = np.random.uniform(0.5, 7.0)
            obj_color = np.random.randint(50, 255, 3)
            
            img[y1:y2, x1:x2] = obj_color
            depth[y1:y2, x1:x2] = obj_depth
        
        Image.fromarray(img).save(os.path.join(rgb_dir, f"{i:05d}.png"))
        np.save(os.path.join(depth_dir, f"{i:05d}.npy"), depth)
    
    # Train/val split
    ids = [f"{i:05d}" for i in range(n_samples)]
    split = int(n_samples * 0.8)
    with open(os.path.join(data_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(ids[:split]))
    with open(os.path.join(data_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(ids[split:]))
    
    print(f"  Created {n_samples} synthetic samples")
    print(f"  Train: {split}, Val: {n_samples - split}")
    print(f"  Location: {data_dir}")
    return data_dir


def test_training(data_dir):
    """Run 1 training epoch to verify the full pipeline works."""
    print("\n" + "=" * 55)
    print(" STEP 3: Test training (1 epoch)")
    print("=" * 55)
    
    import torch
    import torch.optim as optim
    from step3_training.model import LightweightDepthModel
    from step3_training.dataset import DepthDataset
    from step3_training.losses import CombinedLoss
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Lightweight model for quick test
    print("  Creating model...")
    model = LightweightDepthModel(max_depth=10.0, min_depth=0.1).to(device)
    
    # Small dataset
    print("  Loading data...")
    train_ds = DepthDataset(data_dir, split="train",
                            image_height=240, image_width=320,
                            max_depth=10.0, min_depth=0.1, augment=False)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    
    criterion = CombinedLoss(w_l1=1.0, w_ssim=0.0, w_grad=0.5, w_si=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train 1 epoch
    model.train()
    total_loss = 0
    n = 0
    
    print("  Training...")
    for batch in train_loader:
        rgb = batch["rgb"].to(device)
        gt = batch["depth"].to(device)
        mask = batch["mask"].to(device)
        intr = batch["intrinsics"].to(device)
        
        optimizer.zero_grad()
        pred = model(rgb, intr)
        loss, ld = criterion(pred, gt, mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n += 1
    
    avg_loss = total_loss / n
    print(f"  Average loss: {avg_loss:.4f}")
    
    # Save test checkpoint
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "test_model.pth")
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0, "loss": avg_loss},
               ckpt_path)
    print(f"  Model saved: {ckpt_path}")
    
    return ckpt_path


def test_inference(ckpt_path, data_dir):
    """Test prediction on a synthetic image."""
    print("\n" + "=" * 55)
    print(" STEP 4: Test inference")
    print("=" * 55)
    
    import torch
    import numpy as np
    from step3_training.model import LightweightDepthModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = LightweightDepthModel(max_depth=10.0, min_depth=0.1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    
    # Pick a test image
    test_img = os.path.join(data_dir, "rgb", "00000.png")
    gt_depth = np.load(os.path.join(data_dir, "depth", "00000.npy"))
    
    from step4_inference.predict import preprocess, get_intrinsics
    
    tensor, img_rgb, (oh, ow) = preprocess(test_img, H=240, W=320)
    intr, _ = get_intrinsics(None, ow, oh, W=320, H=240)
    
    import torch
    with torch.no_grad():
        pred = model(
            tensor.unsqueeze(0).to(device),
            torch.from_numpy(intr).unsqueeze(0).to(device)
        )
    
    depth = pred.squeeze().cpu().numpy()
    
    print(f"  Input image: {test_img}")
    print(f"  Predicted depth range: {depth.min():.2f}m – {depth.max():.2f}m")
    print(f"  Ground truth range:    {gt_depth.min():.2f}m – {gt_depth.max():.2f}m")
    
    # Save visualization
    import cv2
    from utils.visualization import colorize_depth, create_side_by_side
    
    depth_full = cv2.resize(depth, (ow, oh))
    dc = colorize_depth(depth_full)
    comp = create_side_by_side(img_rgb, dc)
    
    out_path = os.path.join(PROJECT_ROOT, "outputs", "test_prediction.png")
    cv2.imwrite(out_path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"  Visualization: {out_path}")
    
    return True


def main():
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║   METRIC DEPTH ESTIMATION — QUICK START  ║")
    print("  ╚══════════════════════════════════════════╝")
    print()
    
    # Step 0: Dependencies
    if not check_dependencies():
        print("\nFix missing dependencies and re-run.")
        sys.exit(1)
    
    # Step 1: Camera calibration
    test_camera_calibration()
    
    # Step 2: Synthetic dataset
    data_dir = create_synthetic_dataset()
    
    # Step 3: Training test
    ckpt_path = test_training(data_dir)
    
    # Step 4: Inference test
    test_inference(ckpt_path, data_dir)
    
    # Summary
    print("\n" + "=" * 55)
    print(" ALL TESTS PASSED!")
    print("=" * 55)
    print()
    print(" Your setup is working. Now do the REAL steps:")
    print()
    print(" 1. CALIBRATE your camera (or use database):")
    print("    python step1_camera_calibration/calibrate_camera.py --use_database --camera iphone_13")
    print()
    print(" 2. GET real training data:")
    print("    python step2_data_collection/download_nyu_dataset.py")
    print()
    print(" 3. TRAIN on real data:")
    print("    python step3_training/train.py")
    print()
    print(" 4. PREDICT on your photos:")
    print("    python step4_inference/predict.py --image photo.jpg --model outputs/best_model.pth")
    print()
    print(" 5. VALIDATE accuracy:")
    print("    python step5_validation/accuracy_metrics.py --model outputs/best_model.pth")
    print("=" * 55)


if __name__ == "__main__":
    main()
