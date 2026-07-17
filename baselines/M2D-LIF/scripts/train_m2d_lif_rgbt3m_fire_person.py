import argparse

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='./model_yaml/yolov8-LIF.yaml')
    parser.add_argument('--teacher_rgb', required=True)
    parser.add_argument('--teacher_ir', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--name', required=True)
    parser.add_argument('--project', default='./runs/m2d_lif')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_val', action='store_true')
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--augment', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    _, teacher_rgb = attempt_load_one_weight(args.teacher_rgb)
    _, teacher_ir = attempt_load_one_weight(args.teacher_ir)
    overrides = dict(
        ch=6,
        model=args.model,
        data=args.data,
        Distillation='MultiDistillation',
        distill_weight=0.8,
        Teacher_Model_RGB=teacher_rgb['model'],
        Teacher_Model_IR=teacher_ir['model'],
        loss_type='CWD',
        amp=not args.no_amp,
        imgsz=[480, 640],
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        lr0=0.01,
        lrf=0.01,
        cls=0.1,
        copy_paste=0.1,
        optimizer='SGD',
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        online=False,
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
        exist_ok=True,
    )
    DetectionTrainer(overrides=overrides).train()


if __name__ == '__main__':
    main()
