import streamlit as st
import requests
from PIL import Image, ImageDraw

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Marine Debris Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Marine Debris Detection")
st.write(
    "AI-powered detection of underwater objects from "
    "Side-Scan Sonar imagery."
)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Side-Scan Sonar Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Sonar Image")
    st.image(image, use_container_width=True)

    # -----------------------------
    # DETECTION BUTTON
    # -----------------------------

    if st.button("🔍 Detect Objects"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/detect",
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                detections = result["detections"]

                # -----------------------------
                # DRAW BOXES
                # -----------------------------

                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)

                anomaly_found = False

                for detection in detections:

                    object_class = detection["class"]
                    confidence = detection["confidence"]

                    x = detection["x"]
                    y = detection["y"]
                    width = detection["width"]
                    height = detection["height"]

                    # Anomaly rule
                    if confidence < 0.40:
                        anomaly_found = True
                        label = "ANOMALY"
                    else:
                        label = object_class.upper()

                    # Bounding box
                    draw.rectangle(
                        [
                            x,
                            y,
                            x + width,
                            y + height
                        ],
                        outline="red",
                        width=4
                    )

                    # Label
                    draw.text(
                        (x, max(0, y - 20)),
                        f"{label} {confidence * 100:.0f}%"
                    )

                # -----------------------------
                # DISPLAY RESULT
                # -----------------------------

                st.subheader("🎯 Detection Results")

                st.image(
                    annotated_image,
                    caption="Detected objects",
                    use_container_width=True
                )

                # -----------------------------
                # ANOMALY WARNING
                # -----------------------------

                if anomaly_found:

                    st.warning(
                        "⚠️ ANOMALY DETECTED — "
                        "Low-confidence object requires human review."
                    )

                else:

                    st.success(
                        "✅ No low-confidence anomalies detected."
                    )

                # -----------------------------
                # RESULT SUMMARY
                # -----------------------------

                st.subheader("📊 Detection Summary")

                for detection in detections:

                    object_class = detection["class"]
                    confidence = detection["confidence"]

                    if confidence < 0.40:

                        st.write(
                            f"⚠️ **ANOMALY** — "
                            f"{confidence * 100:.0f}% confidence"
                        )

                    else:

                        st.write(
                            f"✅ **{object_class.upper()}** — "
                            f"{confidence * 100:.0f}% confidence"
                        )

            else:

                st.error("Detection API returned an error.")

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to FastAPI. "
                "Make sure the backend server is running."
            )