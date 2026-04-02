# Waste Classifier Web UI

This folder contains a simple HTML frontend for image upload and class prediction.

## 1) Save model checkpoint from notebook

After training your best model in `cnn_experiments.ipynb`, save its `state_dict`:

```python
import torch
torch.save(waste_model.state_dict(), "best_waste_model.pth")
```

Place that file at:

- `cnn/best_waste_model.pth`

If this file is missing, `uvicorn` still starts, but `GET /health` shows `"ready": false` and `/predict` returns 503 until you save the checkpoint.

## 2) Install dependencies

From `cnn/`:

```bash
pip install fastapi uvicorn python-multipart torch torchvision pillow
```

## 3) Start inference API

From `cnn/`:

```bash
uvicorn inference_app:app --host 127.0.0.1 --port 8000
```

## 4) Open HTML page

Open `cnn/web/index.html` in your browser, upload an image, and click **Predict**.

The page calls `http://127.0.0.1:8000/predict` and shows predicted class + confidence + top-5.
