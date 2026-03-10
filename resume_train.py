from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/last.pt")
    model.train(resume=True, epochs=1000)
