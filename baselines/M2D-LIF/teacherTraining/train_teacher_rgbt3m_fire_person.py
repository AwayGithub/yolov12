import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='yolov8m.yaml')
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
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=[480, 640],
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
        cls=0.1,
        copy_paste=0.1,
        augment=False,
        workers=args.workers,
        rect=False,
        save=True,
        amp=not args.no_amp,
        deterministic=True,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        val=not args.no_val,
        plots=not args.no_plots,
        save_period=-1,
    )


if __name__ == '__main__':
    main()
