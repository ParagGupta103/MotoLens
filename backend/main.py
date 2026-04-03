from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
import os
from model_utils import load_model, predict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load classes
CLASSES_FILE = "classes.txt"
if not os.path.exists(CLASSES_FILE):
    # Try parent directory if not found (during development)
    CLASSES_FILE = os.path.join(os.path.dirname(__file__), "classes.txt")

with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load model
MODEL_PATH = "moto_lens_model.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "moto_lens_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, len(classes), DEVICE)

@app.get("/")
async def root():
    return {"message": "MotoLens API is running"}

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    image_bytes = io.BytesIO(await file.read())
    label, confidence = predict(model, image_bytes, classes, DEVICE)
    return {
        "prediction": label,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
