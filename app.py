import os
import uuid
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import open_clip_torch as open_clip  # Corrected import

# Setup
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RECOMMEND_DIR = os.path.join(BASE_DIR, "static", "recommendations")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECOMMEND_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
device = "cpu"
print("[INFO] Loading model...")
model, _, preprocess = open_clip.load("ViT-B/32", device=device)
model.eval()
print("[INFO] Model loaded.")

# Util
def get_image_features(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return features
    except Exception as e:
        print(f"[ERROR] Failed to get image features for {image_path}: {e}")
        return None

def calculate_similarity(query_features, candidate_features):
    return torch.nn.functional.cosine_similarity(query_features, candidate_features).item()

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": []})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    print(f"[INFO] Uploading file: {file.filename}")
    try:
        filename = f"{uuid.uuid4()}_{file.filename}"
        uploaded_path = os.path.join(UPLOAD_DIR, filename)

        with open(uploaded_path, "wb") as f:
            f.write(await file.read())
        print(f"[INFO] File saved to: {uploaded_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save uploaded file: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": [],
            "error": f"Failed to save file: {e}"
        })

    query_features = get_image_features(uploaded_path)
    if query_features is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": [],
            "error": "Could not process uploaded image."
        })

    similarities = []
    for fname in os.listdir(RECOMMEND_DIR):
        fpath = os.path.join(RECOMMEND_DIR, fname)
        candidate_features = get_image_features(fpath)
        if candidate_features is not None:
            sim = calculate_similarity(query_features, candidate_features)
            similarities.append((fname, sim))
        else:
            print(f"[WARN] Skipping image: {fpath}")

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = [os.path.join("static", "recommendations", fname) for fname, _ in similarities[:6]]

    print(f"[INFO] Found {len(top_matches)} matches.")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": top_matches,
        "uploaded": os.path.join("static", "uploads", filename)
    })
