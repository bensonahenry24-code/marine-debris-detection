from ultralytics import YOLO

MODEL_PATH = "model/best.pt"

# Load model once
model = YOLO(MODEL_PATH)

# Class names
CLASS_NAMES = {
    0: "aircraft",
    1: "fish",
    2: "other",
    3: "shipwreck"
}

def predict(image_path, confidence=0.25):
    results = model.predict(
        source=image_path,
        conf=confidence,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": round(conf, 3),
                "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                "is_anomaly": conf < 0.40
            })

    return detections
