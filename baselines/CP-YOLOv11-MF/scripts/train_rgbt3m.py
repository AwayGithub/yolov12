"""
CP-YOLOv11-MF training — compatible with ultralytics v8DetectionLoss.
"""
import argparse, sys, copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np, cv2

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import CPYOLOv11MF, FPNNeck
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import SimpleClass
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


class Args:
    box, cls, dfl = 7.5, 0.1, 1.5


class CPModel(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.inner = CPYOLOv11MF(nc)
        detect = self.inner.neck.detect
        self.nc = detect.nc
        self.reg_max = detect.reg_max
        self.no = detect.no
        self.nl = detect.nl
        self.stride = detect.stride
        self.args = Args()
        # v8DetectionLoss expects model.model[-1] == Detect
        self.model = nn.ModuleList([detect])

    def forward(self, x):
        return self.inner(x)

    @property
    def device(self):
        return next(self.parameters()).device


class RGBT3MDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="train", imgsz=(480, 640)):
        self.imgsz = imgsz
        root = Path(data_root)
        with open(root / f"{split}.txt") as f:
            self.stems = [l.strip() for l in f if l.strip()]
        self.rgb_dir = root / "RGB" / split
        self.ir_dir = root / "IR" / split
        self.lbl_dir = root / "labels_fire_person" / split
        print(f"  {split}: {len(self.stems)} images")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        h, w = self.imgsz

        rgb = cv2.imread(str(self.rgb_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(self.ir_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        rgb = cv2.resize(rgb, (w, h)) if rgb is not None else np.zeros((h, w, 3), np.uint8)
        ir = cv2.resize(ir, (w, h)) if ir is not None else np.zeros((h, w, 3), np.uint8)

        rgb = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        ir = ir.astype(np.float32).transpose(2, 0, 1) / 255.0
        img = np.concatenate([rgb, ir], axis=0)  # 6, H, W

        # Load YOLO labels: class cx cy w h
        labels = []
        lbl = self.lbl_dir / f"{stem}.txt"
        if lbl.exists():
            for line in lbl.open():
                p = line.split()
                if len(p) >= 5:
                    labels.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])

        return torch.tensor(img, dtype=torch.float32), labels

    @staticmethod
    def collate_fn(batch):
        imgs, all_labels = zip(*batch)
        imgs = torch.stack(imgs)
        B = len(all_labels)

        # Build batch_idx, cls, bboxes for v8DetectionLoss
        batch_idx_list, cls_list, bboxes_list = [], [], []
        for b, labels in enumerate(all_labels):
            if labels:
                arr = np.array(labels)
                batch_idx_list.extend([b] * len(arr))
                cls_list.extend(arr[:, 0].tolist())
                bboxes_list.extend(arr[:, 1:5].tolist())

        if batch_idx_list:
            batch_idx = torch.tensor(batch_idx_list, dtype=torch.float32).view(-1, 1)
            cls = torch.tensor(cls_list, dtype=torch.float32).view(-1, 1)
            bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
        else:
            batch_idx = torch.zeros((0, 1))
            cls = torch.zeros((0, 1))
            bboxes = torch.zeros((0, 4))

        return imgs, {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes, "img": imgs}


def train(args):
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root = Path("/data/xwh/dataset/RGBT-3M/RGBT-3M")

    print("Loading data...")
    trn = RGBT3MDataset(root, "train")
    val = RGBT3MDataset(root, "val")
    trn_ld = DataLoader(trn, args.batch, True, collate_fn=RGBT3MDataset.collate_fn,
                        num_workers=0, pin_memory=True)
    val_ld = DataLoader(val, args.batch, False, collate_fn=RGBT3MDataset.collate_fn,
                        num_workers=0)

    model = CPModel(nc=args.nc).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params ({params/1e6:.1f}M)")

    # Compute stride from model
    # For YOLOv11-n: stride = [8, 16, 32]
    stride = torch.tensor([8.0, 16.0, 32.0], device=dev)
    model.inner.neck.detect.stride = stride
    model.stride = stride

    # v8DetectionLoss
    criterion = v8DetectionLoss(model)
    criterion.hyp.box = args.box
    criterion.hyp.cls = args.cls
    criterion.hyp.dfl = args.dfl
    model.args.box = args.box
    model.args.cls = args.cls
    model.args.dfl = args.dfl

    opt = optim.SGD(model.parameters(), lr=args.lr0, momentum=0.937, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr0 * 0.01)

    out = Path(REPO_ROOT) / "runs" / "detect" / "train_cp_yolo11mf"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Save: {out} | {args.epochs} ep, batch={args.batch}, lr={args.lr0}")

    for ep in range(args.epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=dev, requires_grad=True)
        nb = 0
        for bi, (imgs, batch) in enumerate(trn_ld):
            imgs = imgs.to(dev)
            batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            preds = model(imgs)
            loss, loss_items = criterion(preds, batch)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            total_loss = total_loss + loss
            nb += 1
            if bi % 50 == 0:
                print(f"  E{ep+1} B{bi}/{len(trn_ld)} Loss:{loss.item():.4f} "
                      f"box:{loss_items[0].item():.4f} cls:{loss_items[1].item():.4f} dfl:{loss_items[2].item():.4f}")

        sch.step()
        avg = (total_loss / max(nb, 1)).item()
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep+1}/{args.epochs} avg_loss={avg:.4f} lr={lr:.6f}")

        if (ep + 1) % args.save_period == 0 or ep + 1 == args.epochs:
            torch.save({"epoch": ep+1, "model": model.state_dict(), "loss": avg}, out / f"epoch{ep+1}.pt")
        torch.save({"epoch": ep+1, "model": model.state_dict(), "loss": avg}, out / "last.pt")

    print(f"\nDone: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nc", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--cls", type=float, default=0.1)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--dfl", type=float, default=1.5)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--save_period", type=int, default=50)
    train(p.parse_args())
