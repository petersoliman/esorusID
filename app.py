from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, shutil
import torch
import open_clip
from PIL import Image
import numpy as np
import faiss

# Directories
UPLOAD_DIR = "static/uploads"
RECOMMEND_DIR = "static/recommendations"

# App setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)

# FAISS setup
def get_image_features(image_paths):
    images = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
    with torch.no_grad():
        image_input = torch.cat(images)
        image_features = model.encode_image(image_input).cpu().numpy()
    return image_features

def build_faiss_index():
    image_paths = [os.path.join(RECOMMEND_DIR, img) for img in os.listdir(RECOMMEND_DIR) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not image_paths:
        return None, []
    features = get_image_features(image_paths)
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index, image_paths

# Main page: upload and get recommendations
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/search", response_class=HTMLResponse)
async def upload_and_recommend(request: Request, file: UploadFile = File(...)):
    # Save uploaded image
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    uploaded_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(uploaded_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Encode uploaded image
    image = preprocess(Image.open(uploaded_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        uploaded_feature = model.encode_image(image).cpu().numpy()

    # Search recommendations
    index, image_paths = build_faiss_index()
    if index is None:
        results = []
    else:
        D, I = index.search(uploaded_feature, k=min(5, len(image_paths)))
        results = [os.path.basename(image_paths[i]) for i in I[0]]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "uploaded": os.path.basename(uploaded_path)
    })

# ---------------------------
# CRUD for Recommendation Images
# ---------------------------

@app.get("/images", response_class=HTMLResponse)
async def list_images(request: Request):
    images = [f for f in os.listdir(RECOMMEND_DIR) if f.lower().endswith(("jpg", "png", "jpeg"))]
    return templates.TemplateResponse("list_images.html", {"request": request, "images": images})

@app.get("/images/add", response_class=HTMLResponse)
async def add_image_form(request: Request):
    return templates.TemplateResponse("add_image.html", {"request": request})

@app.post("/images/add")
async def add_image(file: UploadFile = File(...)):
    os.makedirs(RECOMMEND_DIR, exist_ok=True)
    path = os.path.join(RECOMMEND_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse(url="/images", status_code=302)

@app.get("/images/edit/{filename}", response_class=HTMLResponse)
async def edit_image_form(request: Request, filename: str):
    return templates.TemplateResponse("edit_image.html", {"request": request, "filename": filename})

@app.post("/images/edit/{filename}")
async def edit_image(filename: str, file: UploadFile = File(...)):
    os.remove(os.path.join(RECOMMEND_DIR, filename))
    new_path = os.path.join(RECOMMEND_DIR, file.filename)
    with open(new_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse(url="/images", status_code=302)

@app.get("/images/delete/{filename}")
async def delete_image(filename: str):
    os.remove(os.path.join(RECOMMEND_DIR, filename))
    return RedirectResponse(url="/images", status_code=302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
