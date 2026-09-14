from ultralytics import YOLO

MODEL_PATH = "model/best.pt"

model = YOLO(MODEL_PATH)

def predict(image_path, confidence=0.25):
    results = model.predict(
        source=image_path,
        conf=confidence
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "confidence": confidence_score,
                "bbox": coordinates
            })

    return detections
