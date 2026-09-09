import json
from pathlib import Path

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
    image = Image.open(uploaded_file).convert("L")
    try:
        model = load_model()
        input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(model(input_tensor).logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        image_column, result_column = st.columns([1, 1], gap="large")
        with image_column:
            st.image(image, caption="Gambar masukan", use_container_width=True)
        with result_column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Hasil prediksi</div>
                    <div class="result-value">{predicted_class.capitalize()}</div>
                    <div>Keyakinan: {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Probabilitas kelas")
        probability_data = {
            class_name.capitalize(): float(probabilities[index].item())
            for index, class_name in enumerate(CLASS_NAMES)
        }
        st.bar_chart(probability_data, horizontal=True, y_label="Kelas", x_label="Probabilitas")
        st.caption("Hasil prediksi merupakan keluaran model dan sebaiknya digunakan sebagai informasi pendukung.")
    except Exception as error:
        st.error("Model gagal dimuat atau prediksi gagal.")
        with st.expander("Detail error"):
            st.exception(error)
