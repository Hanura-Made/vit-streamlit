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


# Harus sama dengan val_test_transform pada notebook:
# grayscale -> 3 channel -> resize 224x224 -> normalize 0.5
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

    # Notebook menyimpan checkpoint dalam format:
    # {'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Mendukung file yang langsung berisi state_dict.
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Format checkpoint tidak didukung. Model harus berupa state_dict "
            "atau dictionary yang memiliki key 'model_state_dict'."
        )

    # Jika pernah dilatih memakai DataParallel, hilangkan prefix 'module.'.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


st.set_page_config(
    page_title="Klasifikasi Gender - ViT",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Klasifikasi Gambar")
st.caption("Model Vision Transformer — klasifikasi pria atau wanita")

uploaded_file = st.file_uploader(
    "Unggah gambar",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    try:
        model = load_model()
        input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(input_tensor).logits
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        st.success(
            f"Prediksi: {predicted_class.upper()} "
            f"({confidence * 100:.2f}%)"
        )

        st.subheader("Probabilitas")
        for index, class_name in enumerate(CLASS_NAMES):
            score = float(probabilities[index].item())
            st.write(f"{class_name}: {score * 100:.2f}%")
            st.progress(score)

    except Exception as error:
        st.error("Model gagal dimuat atau prediksi gagal.")
        st.exception(error)
