from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv12n模型（yaml中已配置ch=6, nc=1）
    model = YOLO("yolov12n.yaml")

    # 开始训练
    results = model.train(
        data="ultralytics/cfg/datasets/FLAME2.yaml",
        epochs=100,
        imgsz=254,
        batch=64,
        workers=0,
        device=0
    )