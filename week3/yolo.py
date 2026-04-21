from ultralytics import YOLO

model = YOLO("yolo26n.pt")
video_path = "video.mp4"

# Run inference on video
results = model.track(source=video_path, save=True, project="outputs", name="video_results", exist_ok=True)