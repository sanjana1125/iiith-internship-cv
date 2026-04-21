from ultralytics import YOLO
import os

# Load the latest YOLO26n model (nano version for speed)
model = YOLO("yolo26n.pt")
output_dir = "outputs"

result = []
# Run inference on an image from a URL
for i in range(1981):
    img_path = f"inputs/image{i}.png"
    if not os.path.exists(img_path):
        print(f"Image {img_path} does not exist. Skipping.")
        continue
    output = model(img_path, save=True, project=output_dir, name="results", exist_ok=True)
    result.append(output)

# Display the results with bounding boxes
