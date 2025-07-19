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

app = FastAPI()

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.to(device).eval()

# Dummy image embeddings and image paths
IMAGE_DIR = "static/images"
IMAGE_PATHS = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'png'))]
image_embeddings = []

# Embed all images in folder
with torch.no_grad():
    for path in IMAGE_PATHS:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        image_embeddings.append(emb.cpu().numpy())

image_embeddings = np.vstack(image_embeddings)

# Index with FAISS
d = image_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(image_embeddings)

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/search")
async def search(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    query_vector = emb.cpu().numpy().astype("float32")
    D, I = index.search(query_vector, k=5)

    result_paths = [IMAGE_PATHS[i].replace("static/", "") for i in I[0]]
    return JSONResponse({"results": result_paths})
