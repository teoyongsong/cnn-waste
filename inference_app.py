from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from waste_model_loader import get_preprocess, load_deployment_config, load_trained_model, predict_image

model, class_names, device, _architecture, CHECKPOINT_PATH, MODEL_LOAD_ERROR = load_trained_model()
preprocess = get_preprocess()

app = FastAPI(title="Waste Classification Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    ready = model is not None
    return {
        "status": "ok" if ready else "no_checkpoint",
        "ready": ready,
        "num_classes": len(class_names),
        "device": str(device),
        "architecture": load_deployment_config()["architecture"],
        "checkpoint_path": str(CHECKPOINT_PATH),
        "error": MODEL_LOAD_ERROR,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=MODEL_LOAD_ERROR or "Model not loaded.",
        )

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    predicted_class, confidence, top_k = predict_image(
        model, class_names, device, preprocess, image
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_k,
    }
