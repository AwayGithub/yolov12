from ultralytics import YOLO

# 加载模型
model = YOLO("yolov12s.yaml")
# 训练（指定6通道数据集配置）
results = model.train(data="ultralytics/cfg/datasets/FLAME2.yaml", epochs=100, imgsz=254, batch=16)