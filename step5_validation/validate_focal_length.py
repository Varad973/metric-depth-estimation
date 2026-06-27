"""
STEP 5A: Validate Focal Length + Depth Accuracy
================================================
Checks whether predicted depth is consistent with camera calibration.

THE KEY TEST:
    If you know an object's real size and real distance:
        focal_length = (pixel_size × real_distance) / real_size
    This estimated focal length should MATCH your calibrated fx/fy!

RUN:
    # With a reference object:
    python step5_validation/validate_focal_length.py \
        --model outputs/best_model.pth \
        --camera camera_params.json \
        --image test_photo.jpg \
        --object_distance 2.0 \
        --object_width 0.3

    # Multiple test photos at known distances:
    python step5_validation/validate_focal_length.py \
        --model outputs/best_model.pth \
        --camera camera_params.json \
        --image_dir test_photos/
"""

import os, sys, json, re, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from step4_inference.predict import load_model, predict_depth


def validate_single(model, image_path, camera_file,
                    object_width_m, object_distance_m, device):
    """
    Validate depth prediction against a known measurement.
    """
    with open(camera_file) as f:
        cam = json.load(f)
    
    calib_fx = cam["fx"]
    
    # Predict
    depth_map = predict_depth(model, image_path, camera_file, device,
                              output_dir="/tmp/val_output")
    
    # Sample center of image (where object is expected)
    h, w = depth_map.shape
    region = depth_map[h//4:3*h//4, w//4:3*w//4]
    pred_depth = float(np.median(region))
    
    # Errors
    depth_err = abs(pred_depth - object_distance_m)
    depth_err_pct = depth_err / object_distance_m * 100
    
    # Estimate focal length from prediction
    # Assume object fills ~50% of image width
    pixel_width = w * 0.5
    est_fx = (pixel_width * pred_depth) / object_width_m
    
    # Scale calibrated fx to this image resolution
    sx = w / cam.get("image_width", w)
    calib_fx_scaled = calib_fx * sx
    
    fx_err = abs(est_fx - calib_fx_scaled)
    fx_err_pct = fx_err / calib_fx_scaled * 100
    
    print(f"\n{'='*55}")
    print(f" VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f" Object distance (real):    {object_distance_m:.2f} m")
    print(f" Object distance (predicted): {pred_depth:.2f} m")
    print(f" Depth error:               {depth_err:.3f} m ({depth_err_pct:.1f}%)")
    print(f"")
    print(f" Calibrated fx:             {calib_fx_scaled:.1f} px")
    print(f" Estimated fx from depth:   {est_fx:.1f} px")
    print(f" Focal length error:        {fx_err:.1f} px ({fx_err_pct:.1f}%)")
    print(f"")
    
    if depth_err_pct < 5:
        print(f" Depth accuracy:  EXCELLENT (< 5%)")
    elif depth_err_pct < 10:
        print(f" Depth accuracy:  GOOD (< 10%)")
    elif depth_err_pct < 20:
        print(f" Depth accuracy:  ACCEPTABLE (< 20%)")
    else:
        print(f" Depth accuracy:  NEEDS WORK (> 20%)")
    
    if fx_err_pct < 10:
        print(f" Focal length:    CONSISTENT (< 10%)")
    elif fx_err_pct < 20:
        print(f" Focal length:    ROUGHLY OK (< 20%)")
    else:
        print(f" Focal length:    INCONSISTENT (> 20%)")
    
    print(f"{'='*55}")
    
    return {"pred_depth": pred_depth, "real_distance": object_distance_m,
            "depth_error_pct": depth_err_pct, "fx_error_pct": fx_err_pct}


def validate_multi(model, image_dir, camera_file, device):
    """Validate using images named with distance: photo_2.0m_001.jpg"""
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    results = []
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        match = re.search(r'(\d+\.?\d*)m', fname)
        if not match:
            continue
        
        real_d = float(match.group(1))
        depth_map = predict_depth(model, fpath, camera_file, device,
                                  output_dir="/tmp/val_output")
        
        h, w = depth_map.shape
        pred_d = float(np.median(depth_map[h//4:3*h//4, w//4:3*w//4]))
        err = abs(pred_d - real_d) / real_d * 100
        
        results.append({"file": fname, "real": real_d, "pred": pred_d, "err_pct": err})
        print(f"  {fname}: real={real_d}m  pred={pred_d:.2f}m  err={err:.1f}%")
    
    if results:
        mean_err = np.mean([r["err_pct"] for r in results])
        print(f"\n  Mean error across {len(results)} images: {mean_err:.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", default="camera_params.json")
    parser.add_argument("--image", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--object_distance", type=float, default=2.0, help="meters")
    parser.add_argument("--object_width", type=float, default=0.3, help="meters")
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mp = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    cp = args.camera if os.path.isabs(args.camera) else os.path.join(ROOT, args.camera)
    
    model = load_model(mp, device, args.lightweight)
    
    if args.image:
        validate_single(model, args.image, cp,
                        args.object_width, args.object_distance, device)
    elif args.image_dir:
        validate_multi(model, args.image_dir, cp, device)
    else:
        print("Specify --image or --image_dir")

if __name__ == "__main__":
    main()
