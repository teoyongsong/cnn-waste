"""
Streamlit UI for waste classification.

Run (from directory containing this file):

    streamlit run streamlit_app.py

Needs: class_names.json (bundled), deployment_config.json, checkpoint .pth
Dataset folder is optional if class_names.json is present.

On Streamlit Cloud: add secret CHECKPOINT_URL pointing to a hosted .pth file,
or commit best_waste_model.pth (see .gitignore exception).
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


def _ensure_checkpoint_from_url() -> bool:
    """If checkpoint is missing and CHECKPOINT_URL is in secrets, download to /tmp (Cloud app dir is often read-only)."""
    from pathlib import Path
    import tempfile
    from urllib.error import URLError, HTTPError
    from urllib.request import urlretrieve

    from waste_model_loader import load_deployment_config, resolve_checkpoint_path

    cfg = load_deployment_config()
    if resolve_checkpoint_path(cfg).is_file():
        return False
    try:
        url = st.secrets.get("CHECKPOINT_URL", "") or ""
    except Exception:
        url = ""
    url = url.strip()
    if not url:
        return False
    dest = Path(tempfile.gettempdir()) / cfg["checkpoint"]
    try:
        with st.spinner("Downloading model checkpoint (one-time, may take a minute)…"):
            urlretrieve(url, str(dest))
        return True
    except (URLError, HTTPError, OSError, ValueError) as e:
        st.error(f"Could not download checkpoint from CHECKPOINT_URL: {e}")
        return False


def _upload_signature(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


@st.cache_resource(show_spinner="Loading model…")
def cached_predictor():
    return load_trained_model()


if _ensure_checkpoint_from_url():
    cached_predictor.clear()

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
    st.markdown(
        """
**Fix on Streamlit Cloud (pick one):**

1. **Secrets (recommended for large files)** — In the app → **Settings → Secrets**, add:
   ```toml
   CHECKPOINT_URL = "https://…/best_waste_model.pth"
   ```
   Use a **direct download** link (e.g. GitHub Release asset, Hugging Face). The file is saved under **/tmp** (app folder is read-only on Cloud).

2. **Commit the weights** — Allow and commit `best_waste_model.pth` (see `.gitignore`), then push. GitHub allows files under 100 MB.

3. **Git LFS** — Track `*.pth` with Git LFS if the file is large.
"""
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
