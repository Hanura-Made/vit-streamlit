import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model_D0.pth"
LABELS_PATH = BASE_DIR / "labels.json"
DEVICE = torch.device("cpu")

with LABELS_PATH.open("r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
        output_attentions=False,
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Format checkpoint tidak didukung.")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_detectors():
    face_detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    eye_detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")
    )
    return face_detector, eye_detector


def detect_and_crop_eyes(image):
    """Deteksi wajah, cari mata, lalu crop area kedua mata setiap wajah."""
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_detector, eye_detector = load_detectors()

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    annotated = image_rgb.copy()
    eye_crops = []

    for face_number, (x, y, width, height) in enumerate(faces, start=1):
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (37, 99, 235), 3)

        # Mata biasanya berada di 65% bagian atas wajah.
        face_gray = gray[y:y + int(height * 0.65), x:x + width]
        eyes = eye_detector.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(15, 15),
        )
        eyes = sorted(eyes, key=lambda eye: eye[0])[:2]

        if len(eyes) < 2:
            continue

        left_x, left_y, left_w, left_h = eyes[0]
        right_x, right_y, right_w, right_h = eyes[1]
        crop_x1 = max(x + min(left_x, right_x) - 20, 0)
        crop_y1 = max(y + min(left_y, right_y) - 20, 0)
        crop_x2 = min(x + max(left_x + left_w, right_x + right_w) + 20, image_rgb.shape[1])
        crop_y2 = min(y + max(left_y + left_h, right_y + right_h) + 20, image_rgb.shape[0])

        for eye_x, eye_y, eye_w, eye_h in eyes:
            start = (x + eye_x, y + eye_y)
            end = (x + eye_x + eye_w, y + eye_y + eye_h)
            cv2.rectangle(annotated, start, end, (22, 163, 74), 2)

        cv2.rectangle(annotated, (crop_x1, crop_y1), (crop_x2, crop_y2), (234, 88, 12), 2)
        eye_crop = Image.fromarray(image_rgb[crop_y1:crop_y2, crop_x1:crop_x2])
        eye_crops.append((face_number, eye_crop))

    return Image.fromarray(annotated), eye_crops


st.set_page_config(page_title="Klasifikasi Gender - ViT", layout="centered")

st.markdown(
    """
    <style>
        .block-container { max-width: 900px; padding-top: 3rem; }
        .app-header { border-bottom: 1px solid #e5e7eb; margin-bottom: 2rem; padding-bottom: 1.25rem; }
        .app-subtitle { color: #6b7280; font-size: 1rem; margin-top: -0.5rem; }
        .result-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; margin: 1.25rem 0; }
        .result-label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .result-value { color: #0f172a; font-size: 1.8rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Klasifikasi Gambar</h1>
        <p class="app-subtitle">Klasifikasi berbasis Vision Transformer</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Informasi model", expanded=False):
    st.write("Model: ViT Base Patch16 224")
    st.write("Kelas: pria dan wanita")
    st.write("Input: grayscale, 224 × 224 piksel")

uploaded_file = st.file_uploader(
    "Pilih gambar untuk dianalisis",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    try:
        model = load_model()
        annotated_image, eye_crops = detect_and_crop_eyes(image)

        if not eye_crops:
            st.warning("Wajah dengan dua mata yang jelas tidak ditemukan pada gambar.")
            st.stop()

        image_column, result_column = st.columns([1, 1], gap="large")
        with image_column:
            st.image(annotated_image, caption="Deteksi wajah dan area mata", use_container_width=True)
        with result_column:
            st.metric("Jumlah wajah terdeteksi", len(eye_crops))
            st.caption("Kotak oranye menunjukkan area mata yang dianalisis.")

        st.subheader("Hasil analisis area mata")
        for face_number, eye_crop in eye_crops:
            input_tensor = IMAGE_TRANSFORM(eye_crop).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor).logits, dim=1)[0]

            predicted_index = int(torch.argmax(probabilities).item())
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index].item())
            probability_data = {
                class_name.capitalize(): float(probabilities[index].item())
                for index, class_name in enumerate(CLASS_NAMES)
            }

            with st.expander(f"Wajah {face_number}: {predicted_class.capitalize()} ({confidence * 100:.2f}%)", expanded=True):
                crop_column, result_column = st.columns([1, 2])
                with crop_column:
                    st.image(eye_crop, caption="Crop mata", use_container_width=True)
                with result_column:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Prediksi</div>'
                        f'<div class="result-value">{predicted_class.capitalize()}</div>'
                        f'<div>Keyakinan: {confidence * 100:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.bar_chart(probability_data, horizontal=True, y_label="Kelas", x_label="Probabilitas")

        st.caption("Hasil prediksi merupakan keluaran model dan sebaiknya digunakan sebagai informasi pendukung.")
    except Exception as error:
        st.error("Model gagal dimuat atau prediksi gagal.")
        with st.expander("Detail error"):
            st.exception(error)
