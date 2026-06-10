# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Runtime P4 Parallel Cross ablation on A1-detected/B2-missed smoke targets."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from analyze_smoke_feature_delta import load_input, patch_legacy, trace
from ultralytics import YOLO


def set_mode(layer, mode):
    if not hasattr(layer, '_diag_original'):
        layer._diag_original = (layer.gamma_rgb.detach().clone(), layer.gamma_ir.detach().clone(), layer.cross_scale_rgb.detach().clone(), layer.cross_scale_ir.detach().clone())
    g1,g2,s1,s2=layer._diag_original
    with torch.no_grad():
        layer.gamma_rgb.copy_(g1); layer.gamma_ir.copy_(g2); layer.cross_scale_rgb.copy_(s1); layer.cross_scale_ir.copy_(s2)
        if mode=='gamma_zero': layer.gamma_rgb.zero_(); layer.gamma_ir.zero_()
        elif mode=='cross_zero': layer.cross_scale_rgb.zero_(); layer.cross_scale_ir.zero_()
        elif mode=='rgb_cross_zero': layer.cross_scale_rgb.zero_()
        elif mode=='ir_cross_zero': layer.cross_scale_ir.zero_()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',default='runs/detect/adr003/smoke_delta/A1_vs_B2_smoke_delta.json');ap.add_argument('--weights',default='runs/detect/train2/weights/last.pt');ap.add_argument('--output',default='runs/detect/adr003/smoke_delta/B2_p4_runtime_ablation.json');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    device=torch.device(args.device);model=YOLO(args.weights).model.to(device).eval();patch_legacy(model);layer=model.backbone_rgb[6]
    records=[x['b'] for x in json.load(open(args.report))['a_detected_b_missed']]
    modes=['baseline','gamma_zero','cross_zero','rgb_cross_zero','ir_cross_zero'];rows=[]
    with torch.inference_mode():
        for idx,r in enumerate(records):
            image=load_input(r['image'],device); item={'image':r['image'],'gt_box':r['gt_box'],'reason':r['reason'],'size_bin':r['size_bin'],'modes':{}}
            for mode in modes:
                set_mode(layer,mode);t=trace(model,image,r['gt_box']);item['modes'][mode]={k:t[k] for k in ['backbone/p4','fused/p4','neck/p4','neck/p3','neck/p2','head/p2','head/p3','head/p4','head/p5']}
            rows.append(item)
            if (idx+1)%20==0: print(idx+1,'/',len(records),flush=True)
    out={'weights':args.weights,'gamma_rgb':float(layer._diag_original[0]),'gamma_ir':float(layer._diag_original[1]),'records':rows}
    Path(args.output).write_text(json.dumps(out,indent=2));print(args.output)
if __name__=='__main__':main()
