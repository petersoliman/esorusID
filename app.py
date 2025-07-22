import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import open_clip
from PIL import Image
import numpy as np
import faiss

# === CONFIG ===
RECOMMEND_DIR = "static/recommendations"
UPLOAD_DIR = "static/uploads"
TOP_K = 5

# === SETUP ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

index = None
recommendation_images = []

# === UTILS ===
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor).cpu().numpy()
    return features / np.linalg.norm(features)

def build_faiss_index():
    global index, recommendation_images
    print("[INFO] Building FAISS index...")
    image_paths = sorted([f for f in os.listdir(RECOMMEND_DIR) if f.lower().endswith(("jpg", "jpeg", "png"))])
    vectors = []
    recommendation_images = []

    for filename in image_paths:
        path = os.path.join(RECOMMEND_DIR, filename)
        try:
            vec = extract_features(path)
            vectors.append(vec)
            recommendation_images.append(filename)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    if vectors:
        vectors_np = np.vstack(vectors).astype("float32")
        index = faiss.IndexFlatL2(vectors_np.shape[1])
        index.add(vectors_np)
        print(f"[INFO] FAISS index built with {len(vectors)} images.")
    else:
        print("[WARNING] No images indexed.")

# === ROUTES ===
@app.on_event("startup")
def startup_event():
    build_faiss_index()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[INFO] Uploaded: {file.filename}")
    try:
        query_vec = extract_features(upload_path).astype("float32")
        D, I = index.search(query_vec, TOP_K)
        results = [recommendation_images[i] for i in I[0]]
        print(f"[INFO] Top-{TOP_K} results: {results}")
    except Exception as e:
        print(f"[ERROR] Failed to search: {e}")
        results = []

    return templates.TemplateResponse("index.html", {
        "request": request,
        "uploaded": file.filename,
        "results": results
    })

@app.get("/images", response_class=HTMLResponse)
async def manage_images(request: Request):
    images = os.listdir(RECOMMEND_DIR)
    return templates.TemplateResponse("images.html", {"request": request, "images": images})
