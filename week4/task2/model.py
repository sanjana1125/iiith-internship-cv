from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # start from pretrained base

model.train(
    data   = 'dataset/data.yaml',  # your yaml with train/val paths
    epochs = 30,
    imgsz  = 640,
    batch  = 8,
    name   = 'my_custom_model'
)
# Best weights saved to:
# runs/detect/my_custom_model/weights/best.pt