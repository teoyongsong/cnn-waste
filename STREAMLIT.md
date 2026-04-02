# Streamlit deployment checklist

## Local smoke test

```bash
cd cnn   # directory containing streamlit_app.py
conda activate waste-cnn   # or your env with PyTorch + Streamlit
python verify_streamlit_ready.py   # must use the same env as Streamlit (needs `torch`)
streamlit run streamlit_app.py
```

If `verify_streamlit_ready.py` warns about imports, you are using a Python without PyTorch; switch to your conda/venv before verifying.

Open http://localhost:8501 — you should see the model status banner and be able to upload + **Predict** if `best_waste_model.pth` exists.

## Required files (in this folder)

| File | Purpose |
|------|---------|
| `streamlit_app.py` | App entrypoint |
| `waste_model_loader.py` | Loads config, model, inference |
| `class_names.json` | 30 class names (same order as training) |
| `deployment_config.json` | `checkpoint` filename + `architecture` string |
| `best_waste_model.pth` | Trained weights (or path set in config) |

Optional: dataset under `data/` — not required for inference if `class_names.json` is present.

## Streamlit Community Cloud

1. Push this folder (or repo containing `cnn/`) to GitHub.
2. New app → GitHub repo → **Main file**: path to `streamlit_app.py` (e.g. `cnn/streamlit_app.py` if repo root is above `cnn`).
3. **Requirements file**: use `cnn/requirements-streamlit.txt` (faster install than full `requirements.txt`).
4. **Python**: 3.10 or 3.11. If your repo root is not the `cnn/` folder, set the version in the Cloud UI or add a `runtime.txt` at the **repository root** (you can copy `cnn/runtime.txt` or mirror `python-3.10.12`).
5. **Large `.pth` file**: Git LFS or host externally; if repo size limits apply, upload the checkpoint via release asset / Drive and document manual download, or use [Streamlit secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management) to point to a URL (would require a small code change to download on startup).

## Architecture strings

Must match how you trained and saved weights: `resnet18`, `resnet34`, `efficientnet_b0`, `densenet121`, `convnext_tiny`.

## After training (notebook)

Run the save cell that writes `best_waste_model.pth` and `deployment_config.json`, commit both (or LFS for `.pth`), then redeploy.
