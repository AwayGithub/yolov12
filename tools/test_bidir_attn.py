import sys
sys.path.insert(0, r"E:\Yan-Unifiles\lab\exp\yolov12\.claude\worktrees\magical-burnell")

import torch
from ultralytics.nn.modules.block import BidirLinearCrossAttnBlock, BidirCrossModalA2C2f

B, C, H, W = 2, 64, 15, 20
x_rgb = torch.randn(B, C, H, W)
x_ir  = torch.randn(B, C, H, W)

# --- BidirLinearCrossAttnBlock ---
blk = BidirLinearCrossAttnBlock(channels=64, num_heads=2)
out_r, out_i = blk(x_rgb, x_ir)
assert out_r.shape == (B, C, H, W), f"shape mismatch: {out_r.shape}"
assert out_i.shape == (B, C, H, W), f"shape mismatch: {out_i.shape}"
print(f"BidirLinearCrossAttnBlock OK  {tuple(out_r.shape)}")

# init check: out_proj 全零 → attn 分支输出 0 → out ≈ conv_bypass(x)
blk.eval()
with torch.no_grad():
    out_r2, _ = blk(x_rgb, x_ir)
    diff = (out_r2 - blk.conv_rgb(x_rgb)).abs().max().item()
print(f"  init check max|delta|={diff:.2e}  (expect ~0)")

# --- BidirCrossModalA2C2f ---
mod = BidirCrossModalA2C2f(c1=128, c2=128, n=2, e=0.5)
x128 = torch.randn(B, 128, H, W)
out_r3, out_i3 = mod(x128, x128)
assert out_r3.shape == (B, 128, H, W), f"shape mismatch: {out_r3.shape}"
assert out_i3.shape == (B, 128, H, W), f"shape mismatch: {out_i3.shape}"
print(f"BidirCrossModalA2C2f OK  {tuple(out_r3.shape)}")

print(f"  BidirLinearCrossAttnBlock params: {sum(p.numel() for p in blk.parameters()):,}")
print(f"  BidirCrossModalA2C2f    params: {sum(p.numel() for p in mod.parameters()):,}")

# gradient flow
(out_r3.sum() + out_i3.sum()).backward()
n_grad  = sum(1 for p in mod.parameters() if p.grad is not None)
n_total = sum(1 for _ in mod.parameters())
print(f"  grad-bearing params: {n_grad}/{n_total}")

print("ALL PASS")
