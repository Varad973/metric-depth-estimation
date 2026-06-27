"""
STEP 3: Train the Metric Depth Estimation Model
=================================================
RUN:
    python step3_training/train.py                        # Default config
    python step3_training/train.py --config configs/default.yaml
    python step3_training/train.py --lightweight           # Less GPU memory
    python step3_training/train.py --resume outputs/checkpoint_epoch_10.pth

MONITOR:
    tensorboard --logdir outputs/logs
    Then open http://localhost:6006
"""

import os, sys, time, yaml, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from step3_training.model import MetricDepthModel, LightweightDepthModel, create_model
from step3_training.dataset import create_dataloaders
from step3_training.losses import CombinedLoss


# ============================================================
# STANDARD DEPTH EVALUATION METRICS
# ============================================================
def compute_metrics(pred, gt, mask, min_d=0.1, max_d=10.0):
    """
    Compute standard depth estimation benchmarks.
    These are the same metrics used in all published depth papers.
    """
    valid = mask.bool().squeeze()
    if valid.sum() < 100:
        return {}
    
    p = pred.squeeze()[valid].clamp(min_d, max_d)
    g = gt.squeeze()[valid].clamp(min_d, max_d)
    
    diff = torch.abs(p - g)
    ratio = torch.max(p / g, g / p)
    
    return {
        "abs_rel":  (diff / g).mean().item(),
        "sq_rel":   ((diff**2) / g).mean().item(),
        "rmse":     (diff**2).mean().sqrt().item(),
        "rmse_log": ((torch.log(p) - torch.log(g))**2).mean().sqrt().item(),
        "delta1":   (ratio < 1.25).float().mean().item(),
        "delta2":   (ratio < 1.25**2).float().mean().item(),
        "delta3":   (ratio < 1.25**3).float().mean().item(),
    }


# ============================================================
# TRAINING LOOP
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, writer, clip=1.0):
    model.train()
    total_loss, n = 0, 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for i, batch in enumerate(pbar):
        rgb   = batch["rgb"].to(device)
        gt    = batch["depth"].to(device)
        mask  = batch["mask"].to(device)
        intr  = batch["intrinsics"].to(device)
        
        optimizer.zero_grad()
        pred = model(rgb, intr)
        loss, ld = criterion(pred, gt, mask)
        loss.backward()
        
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        step = epoch * len(loader) + i
        if i % 10 == 0:
            writer.add_scalar("train/loss", loss.item(), step)
    
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, config):
    model.eval()
    total_loss, n = 0, 0
    all_metrics = []
    
    ds = config.get("dataset", {})
    
    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} [val]")):
        rgb  = batch["rgb"].to(device)
        gt   = batch["depth"].to(device)
        mask = batch["mask"].to(device)
        intr = batch["intrinsics"].to(device)
        
        pred = model(rgb, intr)
        loss, _ = criterion(pred, gt, mask)
        total_loss += loss.item()
        n += 1
        
        m = compute_metrics(pred, gt, mask,
                           ds.get("min_depth", 0.1), ds.get("max_depth", 10.0))
        if m:
            all_metrics.append(m)
    
    avg_loss = total_loss / n
    avg_m = {}
    if all_metrics:
        for k in all_metrics[0]:
            avg_m[k] = np.mean([m[k] for m in all_metrics])
    
    writer.add_scalar("val/loss", avg_loss, epoch)
    for k, v in avg_m.items():
        writer.add_scalar(f"val/{k}", v, epoch)
    
    return avg_loss, avg_m


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    # Load config
    cfg_path = os.path.join(ROOT, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU — training will be slow!")
    
    # Directories
    save_dir = os.path.join(ROOT, cfg["output"]["save_dir"])
    log_dir  = os.path.join(ROOT, cfg["output"]["log_dir"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # Data
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(cfg)
    
    # Model
    print("\nCreating model...")
    if args.lightweight:
        model = LightweightDepthModel(
            max_depth=cfg["dataset"].get("max_depth", 10.0),
            min_depth=cfg["dataset"].get("min_depth", 0.1))
    else:
        model = create_model(cfg)
    model = model.to(device)
    
    # Loss
    lc = cfg.get("loss", {})
    criterion = CombinedLoss(
        w_l1=lc.get("l1_weight", 1.0), w_ssim=lc.get("ssim_weight", 0.5),
        w_grad=lc.get("gradient_weight", 0.5), w_si=lc.get("scale_invariant_weight", 0.5))
    
    # Optimizer + scheduler
    tc = cfg.get("training", {})
    optimizer = optim.AdamW(model.parameters(),
                           lr=tc.get("learning_rate", 1e-4),
                           weight_decay=tc.get("weight_decay", 0.01))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.get("epochs", 50), eta_min=1e-6)
    
    # Resume
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")
    
    # ==================== TRAIN ====================
    epochs = tc.get("epochs", 50)
    clip = tc.get("gradient_clip", 1.0)
    
    print(f"\n{'='*55}")
    print(f" Training: {epochs} epochs, batch_size={tc.get('batch_size',8)}")
    print(f" Monitor:  tensorboard --logdir {log_dir}")
    print(f"{'='*55}\n")
    
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, device, epoch, writer, clip)
        val_loss, val_m = validate(model, val_loader, criterion,
                                    device, epoch, writer, cfg)
        scheduler.step()
        
        dt = time.time() - t0
        print(f"\nEpoch {epoch}/{epochs-1} ({dt:.0f}s)")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f}")
        if val_m:
            print(f"  AbsRel: {val_m.get('abs_rel',0):.4f}  "
                  f"RMSE: {val_m.get('rmse',0):.3f}m  "
                  f"delta1: {val_m.get('delta1',0):.3f}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch, "loss": val_loss, "metrics": val_m,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  ★ Best model saved (loss={val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % cfg["output"].get("save_every", 5) == 0:
            torch.save({
                "epoch": epoch, "loss": val_loss, "metrics": val_m,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    writer.close()
    print(f"\n{'='*55}")
    print(f" TRAINING COMPLETE — best loss: {best_loss:.4f}")
    print(f" Models: {save_dir}/")
    print(f" Next:   python step4_inference/predict.py --model {save_dir}/best_model.pth")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
