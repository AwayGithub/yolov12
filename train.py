from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv12n模型（yaml中已配置ch=6, nc=1）
    model = YOLO("yolov12n.yaml")

    # 开始训练
    results = model.train(
    data="ultralytics/cfg/datasets/FLAME2.yaml",
    epochs=3,
    imgsz=254,
    batch=64,
    workers=0,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.0,
    cos_lr=False,
)