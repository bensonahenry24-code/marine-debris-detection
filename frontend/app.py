import streamlit as st
import requests
from PIL import Image, ImageDraw

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Marine Debris Detection",
    page_icon="",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------

st.title(" Marine Debris Detection")
st.markdown(
    "### AI-powered Side-Scan Sonar Analysis"
)

st.write(
    "Upload a Side-Scan Sonar image to detect underwater "
    "objects and flag low-confidence anomalies for human review."
)

st.divider()

# -----------------------------
# UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Side-Scan Sonar Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    if st.button("🔍 Analyze Sonar Image", use_container_width=True):

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
                # DRAW DETECTIONS
                # -----------------------------

                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)

                anomaly_count = 0
                total_confidence = 0

                for detection in detections:

                    object_class = detection["class"]
                    confidence = detection["confidence"]

                    x = detection["x"]
                    y = detection["y"]
                    width = detection["width"]
                    height = detection["height"]

                    total_confidence += confidence

                    # Low confidence = anomaly
                    if confidence < 0.40:

                        label = "ANOMALY"
                        anomaly_count += 1

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
                # METRICS
                # -----------------------------

                detection_count = len(detections)

                if detection_count > 0:
                    average_confidence = (
                        total_confidence / detection_count
                    ) * 100
                else:
                    average_confidence = 0

                st.divider()

                st.subheader("📊 Analysis Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Objects Detected",
                        detection_count
                    )

                with col2:
                    st.metric(
                        "Average Confidence",
                        f"{average_confidence:.0f}%"
                    )

                with col3:
                    st.metric(
                        "Anomalies",
                        anomaly_count
                    )

                # -----------------------------
                # ANOMALY WARNING
                # -----------------------------

                if anomaly_count > 0:

                    st.warning(
                        f"⚠️ {anomaly_count} low-confidence "
                        "object(s) flagged for human review."
                    )

                else:

                    st.success(
                        "✅ No low-confidence anomalies detected."
                    )

                # -----------------------------
                # ANNOTATED IMAGE
                # -----------------------------

                st.subheader(" Detection Result")

                st.image(
                    annotated_image,
                    caption="Detected objects and anomalies",
                    use_container_width=True
                )

                # -----------------------------
                # DETAILS
                # -----------------------------

                st.subheader("Detection Details")

                for i, detection in enumerate(detections, 1):

                    object_class = detection["class"]
                    confidence = detection["confidence"]

                    if confidence < 0.40:

                        st.write(
                            f"⚠️ **Object {i}: ANOMALY** — "
                            f"{confidence * 100:.0f}% confidence"
                        )

                    else:

                        st.write(
                            f"✅ **Object {i}: "
                            f"{object_class.upper()}** — "
                            f"{confidence * 100:.0f}% confidence"
                        )

            else:

                st.error(
                    "❌ Detection API returned an error."
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to FastAPI. "
                "Make sure the backend server is running."
            )