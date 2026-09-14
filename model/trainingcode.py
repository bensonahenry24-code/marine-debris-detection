import yaml
from ultralytics import YOLO


yaml_path = "/content/dataset/final_dataset/data.yaml"


with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

config["path"] = "/content/dataset/final_dataset"

with open(yaml_path, "w") as f:
    yaml.dump(config, f)

print(f"Successfully updated data.yaml path to: {config['path']}")
print("Classes in dataset:", config.get("names"))


model = YOLO("yolov8s.pt")  

model.train(
    data=yaml_path,
    epochs=30,   
    imgsz=640,   
    batch=16,    
    device=0     
)