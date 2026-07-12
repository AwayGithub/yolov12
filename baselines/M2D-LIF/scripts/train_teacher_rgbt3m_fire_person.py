import argparse

from ultralytics.models.yolo.detect import DetectionTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--name', required=True)
    parser.add_argument('--project', default='./runs/teachers')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_val', action='store_true')
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--augment', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = dict(
        ch=3,
        model='./ultralytics/cfg/models/v8/yolov8m.yaml',
        data=args.data,
        amp=not args.no_amp,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        lr0=0.01,
        lrf=0.01,
        optimizer='SGD',
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        augment=args.augment,
        workers=args.workers,
        rect=False,
        deterministic=True,
        seed=args.seed,
        save=True,
        val=not args.no_val,
        plots=not args.no_plots,
        project=args.project,
        name=args.name,
    )
    DetectionTrainer(overrides=overrides).train()


if __name__ == '__main__':
    main()
