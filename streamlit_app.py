"""
Streamlit UI for waste classification.

Run (from directory containing this file):

    streamlit run streamlit_app.py

Needs: class_names.json (bundled), deployment_config.json, checkpoint .pth
Dataset folder is optional if class_names.json is present.
"""
from __future__ import annotations

import streamlit as st
from PIL import Image

from waste_model_loader import get_preprocess, load_trained_model, predict_image

st.set_page_config(
    page_title="Waste Class Predictor",
    page_icon="♻️",
    layout="centered",
)


def _upload_signature(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


@st.cache_resource(show_spinner="Loading model…")
def cached_predictor():
    return load_trained_model()


try:
    model, class_names, device, architecture, checkpoint_path, load_error = cached_predictor()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

preprocess = get_preprocess()

if "upload_sig" not in st.session_state:
    st.session_state.upload_sig = None
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None

st.title("Waste class predictor")
st.caption("Recyclable & household waste — upload a photo to classify.")

if load_error:
    st.error(load_error)
    st.info(
        f"Expected checkpoint: `{checkpoint_path}` · "
        f"`deployment_config.json` architecture: `{architecture}`"
    )
else:
    st.success(
        f"Model loaded · **{architecture}** · {len(class_names)} classes · {device}"
    )

uploaded = st.file_uploader(
    "Drag and drop an image here, or click to browse",
    type=["png", "jpg", "jpeg", "webp"],
    help="PNG, JPEG, or WebP",
)

sig = _upload_signature(uploaded)
if sig != st.session_state.upload_sig:
    st.session_state.upload_sig = sig
    st.session_state.pred_result = None

image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

if image is not None:
    st.image(image, caption="Preview", width=280)

    if model is None:
        st.warning("Add a valid checkpoint to enable predictions.")
    else:
        if st.button("Predict", type="primary"):
            with st.spinner("Running model…"):
                st.session_state.pred_result = predict_image(
                    model, class_names, device, preprocess, image
                )

if st.session_state.pred_result is not None and uploaded is not None:
    if _upload_signature(uploaded) == st.session_state.upload_sig:
        pred, conf, top_k = st.session_state.pred_result
        st.divider()
        st.metric("Predicted class", pred.replace("_", " "))
        st.metric("Confidence", f"{conf * 100:.2f}%")
        st.subheader("Top predictions")
        for i, row in enumerate(top_k, start=1):
            label = (
                f"{i}. {row['class_name'].replace('_', ' ')} — "
                f"{row['confidence'] * 100:.1f}%"
            )
            st.write(label)
            st.progress(min(1.0, float(row["confidence"])))
