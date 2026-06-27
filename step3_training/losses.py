"""
Loss Functions for Metric Depth Estimation
==========================================
We combine four losses, each targeting a different aspect of depth quality:

1. L1 Loss         — overall absolute accuracy in meters
2. SSIM Loss       — structural patterns (edges, textures)
3. Gradient Loss   — sharp depth boundaries at object edges
4. SI-Log Loss     — scale-invariant log error (handles near AND far equally)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """Absolute error on valid pixels."""
    def forward(self, pred, gt, mask):
        return (torch.abs(pred - gt) * mask).sum() / (mask.sum() + 1e-8)


class SSIMLoss(nn.Module):
    """Structural Similarity — penalizes loss of texture/edge structure."""
    def __init__(self, win=11):
        super().__init__()
        self.win = win
        self.C1, self.C2 = 0.01**2, 0.03**2
    
    def forward(self, pred, gt, mask):
        mx = gt[mask.bool()].max() if mask.sum() > 0 else 10.0
        p, g = pred / (mx + 1e-8) * mask, gt / (mx + 1e-8) * mask
        pad = self.win // 2
        
        mu_p = F.avg_pool2d(p, self.win, 1, pad)
        mu_g = F.avg_pool2d(g, self.win, 1, pad)
        
        s_pp = F.avg_pool2d(p*p, self.win, 1, pad) - mu_p*mu_p
        s_gg = F.avg_pool2d(g*g, self.win, 1, pad) - mu_g*mu_g
        s_pg = F.avg_pool2d(p*g, self.win, 1, pad) - mu_p*mu_g
        
        ssim = ((2*mu_p*mu_g + self.C1) * (2*s_pg + self.C2)) / \
               ((mu_p**2 + mu_g**2 + self.C1) * (s_pp + s_gg + self.C2))
        
        valid = (F.avg_pool2d(mask, self.win, 1, pad) > 0.5).float()
        return ((1 - ssim) * valid).sum() / (valid.sum() + 1e-8)


class GradientLoss(nn.Module):
    """Edge-aware loss — depth boundaries should be sharp."""
    def forward(self, pred, gt, mask):
        dx_p = pred[:,:,:,:-1] - pred[:,:,:,1:]
        dy_p = pred[:,:,:-1,:] - pred[:,:,1:,:]
        dx_g = gt[:,:,:,:-1] - gt[:,:,:,1:]
        dy_g = gt[:,:,:-1,:] - gt[:,:,1:,:]
        
        mx = mask[:,:,:,:-1] * mask[:,:,:,1:]
        my = mask[:,:,:-1,:] * mask[:,:,1:,:]
        
        lx = (torch.abs(dx_p - dx_g) * mx).sum() / (mx.sum() + 1e-8)
        ly = (torch.abs(dy_p - dy_g) * my).sum() / (my.sum() + 1e-8)
        return lx + ly


class ScaleInvariantLogLoss(nn.Module):
    """
    SI-Log Loss (Eigen et al. 2014) — THE key loss for metric depth.
    Works in log-space so near objects (1m) and far objects (10m) are treated equally.
    
    L = mean(d²) - λ·mean(d)²  where d = log(pred) - log(gt)
    """
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, pred, gt, mask):
        valid = mask.bool()
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device)
        
        d = torch.log(pred[valid].clamp(1e-3)) - torch.log(gt[valid].clamp(1e-3))
        return (d**2).mean() - self.lambd * d.mean()**2


class CombinedLoss(nn.Module):
    """Weighted combination of all four losses."""
    
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_grad=0.5, w_si=0.5):
        super().__init__()
        self.l1 = L1Loss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()
        self.si = ScaleInvariantLogLoss()
        self.w = {"l1": w_l1, "ssim": w_ssim, "grad": w_grad, "si": w_si}
        print(f"  Loss weights: {self.w}")
    
    def forward(self, pred, gt, mask):
        losses = {}
        total = torch.tensor(0.0, device=pred.device)
        
        for name, fn in [("l1", self.l1), ("ssim", self.ssim),
                          ("grad", self.grad), ("si", self.si)]:
            if self.w[name] > 0:
                v = fn(pred, gt, mask)
                losses[name] = v.item()
                total = total + self.w[name] * v
        
        losses["total"] = total.item()
        return total, losses
