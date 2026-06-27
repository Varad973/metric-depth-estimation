"""
STEP 5B: Full Accuracy Report on Validation Set
================================================
Runs the trained model on the entire validation set and computes:
    - AbsRel, SqRel, RMSE, RMSE_log, MAE
    - delta1 (< 1.25), delta2 (< 1.56), delta3 (< 1.95)
    - Per-depth-range breakdown
    - Comparison with published benchmarks

RUN:
    python step5_validation/accuracy_metrics.py --model outputs/best_model.pth
    python step5_validation/accuracy_metrics.py --model outputs/best_model.pth --data_dir data/custom
"""

import os, sys, json, argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from step3_training.dataset import DepthDataset
from step4_inference.predict import load_model


def evaluate(model, data_dir, camera_params, device, max_d=10.0, min_d=0.1, bs=4):
    """Full evaluation on validation set."""
    
    dataset = DepthDataset(data_dir, split="val", max_depth=max_d, min_depth=min_d,
                           camera_params=camera_params, augment=False)
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=2)
    
    model.eval()
    metrics_all = {k: [] for k in ["abs_rel","sq_rel","rmse","rmse_log","mae",
                                    "delta1","delta2","delta3"]}
    
    # Per-range buckets
    ranges = [(0,2),(2,4),(4,6),(6,8),(8,10)]
    range_data = {f"{a}-{b}m": {"abs_rel":[], "rmse":[], "n":0} for a,b in ranges}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            rgb  = batch["rgb"].to(device)
            gt   = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            intr = batch["intrinsics"].to(device)
            
            pred = model(rgb, intr)
            
            for i in range(rgb.shape[0]):
                p = pred[i].squeeze()
                g = gt[i].squeeze()
                m = mask[i].squeeze().bool()
                
                if m.sum() < 100:
                    continue
                
                pv = p[m].clamp(min_d, max_d)
                gv = g[m].clamp(min_d, max_d)
                
                diff = torch.abs(pv - gv)
                ratio = torch.max(pv/gv, gv/pv)
                
                metrics_all["abs_rel"].append((diff/gv).mean().item())
                metrics_all["sq_rel"].append(((diff**2)/gv).mean().item())
                metrics_all["rmse"].append((diff**2).mean().sqrt().item())
                metrics_all["rmse_log"].append(
                    ((torch.log(pv)-torch.log(gv))**2).mean().sqrt().item())
                metrics_all["mae"].append(diff.mean().item())
                metrics_all["delta1"].append((ratio<1.25).float().mean().item())
                metrics_all["delta2"].append((ratio<1.25**2).float().mean().item())
                metrics_all["delta3"].append((ratio<1.25**3).float().mean().item())
                
                # Per-range
                for lo, hi in ranges:
                    rm = (gv >= lo) & (gv < hi)
                    if rm.sum() > 10:
                        key = f"{lo}-{hi}m"
                        range_data[key]["abs_rel"].append(
                            (torch.abs(pv[rm]-gv[rm])/gv[rm]).mean().item())
                        range_data[key]["rmse"].append(
                            ((pv[rm]-gv[rm])**2).mean().sqrt().item())
                        range_data[key]["n"] += 1
    
    # Average
    avg = {k: np.mean(v) for k,v in metrics_all.items() if v}
    
    # Print results
    print(f"\n{'='*60}")
    print(f" EVALUATION RESULTS ({len(metrics_all['abs_rel'])} samples)")
    print(f"{'='*60}")
    print(f" {'Metric':<15} {'Value':>10}")
    print(f" {'-'*26}")
    for k in ["abs_rel","sq_rel","rmse","rmse_log","mae","delta1","delta2","delta3"]:
        arrow = "↓" if k not in ["delta1","delta2","delta3"] else "↑"
        unit = " m" if k in ["rmse","mae"] else ""
        print(f" {k:<15} {avg.get(k,0):>9.4f}{unit} {arrow}")
    
    print(f"\n Per-Depth-Range:")
    print(f" {'Range':<10} {'Count':>6} {'AbsRel':>8} {'RMSE':>8}")
    print(f" {'-'*34}")
    for key in range_data:
        rd = range_data[key]
        if rd["n"] > 0:
            print(f" {key:<10} {rd['n']:>6} "
                  f"{np.mean(rd['abs_rel']):>8.4f} "
                  f"{np.mean(rd['rmse']):>7.3f}m")
    
    # Quality summary
    d1 = avg.get("delta1", 0)
    print(f"\n Quality: ", end="")
    if d1 > 0.95:
        print("EXCELLENT")
    elif d1 > 0.85:
        print("GOOD")
    elif d1 > 0.70:
        print("FAIR — more training data or epochs may help")
    else:
        print("NEEDS IMPROVEMENT — check dataset quality and config")
    
    print(f"\n Published benchmarks (NYU Depth V2 for reference):")
    print(f" {'Model':<20} {'AbsRel':>8} {'RMSE':>8} {'delta1':>8}")
    print(f" {'-'*46}")
    print(f" {'Your Model':<20} {avg.get('abs_rel',0):>8.4f} {avg.get('rmse',0):>8.3f} {d1:>8.3f}")
    print(f" {'AdaBins':<20} {'0.103':>8} {'0.364':>8} {'0.903':>8}")
    print(f" {'BTS':<20} {'0.110':>8} {'0.392':>8} {'0.885':>8}")
    print(f" {'Eigen et al.':<20} {'0.158':>8} {'0.641':>8} {'0.769':>8}")
    print(f"{'='*60}")
    
    return {"overall": avg, "per_range": {
        k: {"abs_rel": np.mean(v["abs_rel"]) if v["abs_rel"] else None,
            "rmse": np.mean(v["rmse"]) if v["rmse"] else None}
        for k,v in range_data.items()
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", default="data/nyu_depth_v2")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    mp = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    dd = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)
    
    cam = None
    if args.camera:
        cp = args.camera if os.path.isabs(args.camera) else os.path.join(ROOT, args.camera)
        with open(cp) as f:
            cam = json.load(f)
    
    model = load_model(mp, device, args.lightweight)
    results = evaluate(model, dd, cam, device, args.max_depth, bs=args.batch_size)
    
    op = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved: {op}")

if __name__ == "__main__":
    main()
