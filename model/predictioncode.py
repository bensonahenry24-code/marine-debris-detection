import os
import glob
from ultralytics import YOLO
from IPython.display import Image, display


weight_files = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
if not weight_files:
    raise FileNotFoundError("No trained weights found in runs/detect/")

latest_weights = max(weight_files, key=os.path.getctime)
print("Loading weights from:", latest_weights)
model = YOLO(latest_weights)


search_dirs = [
    "/content/dataset/final_dataset/images/test",
    "/content/dataset/final_dataset/images/val",
    "/content/dataset/final_dataset/images/train",
    "/content/dataset/final_dataset/images",
    "/content/dataset/final_dataset"
]

valid_source = None
for folder in search_dirs:
    if os.path.exists(folder):
        has_images = any(
            f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            for _, _, files in os.walk(folder)
            for f in files
        )
        if has_images:
            valid_source = folder
            break

if not valid_source:
    raise FileNotFoundError("No images found in /content/dataset/final_dataset.")

print(f"Running inference on images from: {valid_source}")


results = model.predict(
    source=valid_source,
    conf=0.25,
    save=True
)

# 4. Display the first 3 predicted output images
predict_dirs = glob.glob("runs/detect/predict*")
latest_predict_dir = max(predict_dirs, key=os.path.getctime)
output_images = glob.glob(f"{latest_predict_dir}/*.jpg") + glob.glob(f"{latest_predict_dir}/*.png")

print(f"\nShowing predictions saved in {latest_predict_dir}:")
for img_path in output_images[:30]:
    display(Image(filename=img_path, width=600))