from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import open_clip
import torch
import faiss
import numpy as np
import io
import os

print("✅ Starting app setup...")

app = FastAPI()

# Serve static files (frontend)
print("✅ Mounting static directory...")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
print("✅ Loading OpenCLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.to(device).eval()
print("✅ Model loaded and ready.")

# Embed static images
IMAGE_DIR = "static/images"
print(f"✅ Reading images from: {IMAGE_DIR}")
IMAGE_PATHS = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'png'))]
print(f"✅ Found {len(IMAGE_PATHS)} image(s).")

image_embeddings = []

with torch.no_grad():
    for path in IMAGE_PATHS:
        print(f"🔍 Processing image: {path}")
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        image_embeddings.append(emb.cpu().numpy())

image_embeddings = np.vstack(image_embeddings)
print(f"✅ Image embeddings shape: {image_embeddings.shape}")

# FAISS index setup
print("✅ Creating FAISS index...")
d = image_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(image_embeddings)
print("✅ FAISS index created and populated.")

@app.get("/", response_class=HTMLResponse)
async def root():
    print("📥 Received GET / request for homepage.")
    try:
        with open("static/index.html") as f:
            print("✅ index.html found and loaded.")
            return f.read()
    except Exception as e:
        print(f"❌ Error loading index.html: {e}")
        return HTMLResponse(content="Error loading index.html", status_code=500)

@app.post("/search")
async def search(file: UploadFile = File(...)):
    print(f"📥 Received POST /search request with file: {file.filename}")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        print("🔍 Encoding uploaded image...")
        with torch.no_grad():
            emb = model.encode_image(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        query_vector = emb.cpu().numpy().astype("float32")
        D, I = index.search(query_vector, k=5)

        print(f"✅ Top 5 matches found: {I[0]}")
        result_paths = [IMAGE_PATHS[i].replace("static/", "") for i in I[0]]
        print(f"✅ Result paths: {result_paths}")

        return JSONResponse({"results": result_paths})
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
