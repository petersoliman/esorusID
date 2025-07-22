from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from shutil import copyfile
import os
import uuid
import torch
import faiss
from PIL import Image
import numpy as np
import open_clip
import logging

logging.basicConfig(level=logging.INFO)


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = Path("static/uploads")
RECOMMEND_FOLDER = Path("static/recommendations")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RECOMMEND_FOLDER.mkdir(parents=True, exist_ok=True)

# Global index and image metadata
index = None
image_features = []
image_filenames = []
model = None
preprocess = None
device = "cpu"


def load_model():
    global model, preprocess
    print("Loading model...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.exception("Error loading model")
        raise e  # Make sure it raises so Railway logs the error

    model.eval()
    model.to(device)


def extract_features(image: Image.Image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor).squeeze().cpu().numpy()
    return features.astype("float32")


def build_index():
    global index, image_features, image_filenames
    print("Building FAISS index from recommendation images...")
    image_features = []
    image_filenames = []
    for image_path in RECOMMEND_FOLDER.glob("*.*"):
        try:
            img = Image.open(image_path).convert("RGB")
            feature = extract_features(img)
            image_features.append(feature)
            image_filenames.append(image_path.name)
        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")
    if image_features:
        features_array = np.vstack(image_features)
        index = faiss.IndexFlatL2(features_array.shape[1])
        index.add(features_array)
        print(f"Indexed {len(image_filenames)} images.")
    else:
        index = None
        print("No images indexed.")


@app.on_event("startup")
async def startup_event():
    try:
        logging.info("Loading model...")
        global model, preprocess
        load_model()
        logging.info("Model loaded successfully.")

        logging.info("Indexing recommendation images...")
        build_index()
        logging.info("Images indexed successfully.")
    except Exception as e:
        logging.exception("Startup failed due to exception:")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile):
    contents = await file.read()
    ext = file.filename.split(".")[-1]
    name = f"{uuid.uuid4()}.{ext}"
    path = UPLOAD_FOLDER / name
    with open(path, "wb") as f:
        f.write(contents)

    img = Image.open(path).convert("RGB")
    query_feat = extract_features(img)

    if index is None or len(image_filenames) == 0:
        return templates.TemplateResponse("index.html", {"request": request, "uploaded": name, "results": []})

    D, I = index.search(np.expand_dims(query_feat, axis=0), k=5)
    results = [image_filenames[i] for i in I[0]]
    return templates.TemplateResponse("index.html", {"request": request, "uploaded": name, "results": results})


@app.get("/images", response_class=HTMLResponse)
async def list_images(request: Request):
    images = [f.name for f in RECOMMEND_FOLDER.iterdir() if f.is_file()]
    return templates.TemplateResponse("list_images.html", {"request": request, "images": images})


@app.get("/images/add", response_class=HTMLResponse)
async def add_image_form(request: Request):
    return templates.TemplateResponse("add_image.html", {"request": request})


@app.post("/images/add")
async def add_image(request: Request, file: UploadFile):
    contents = await file.read()
    ext = file.filename.split(".")[-1]
    name = f"{uuid.uuid4()}.{ext}"
    path = RECOMMEND_FOLDER / name
    with open(path, "wb") as f:
        f.write(contents)
    build_index()
    return RedirectResponse(url="/images", status_code=302)


@app.get("/images/edit/{image_name}", response_class=HTMLResponse)
async def edit_image_form(request: Request, image_name: str):
    return templates.TemplateResponse("edit_image.html", {"request": request, "image": image_name})


@app.post("/images/edit/{image_name}")
async def edit_image(image_name: str, file: UploadFile):
    path = RECOMMEND_FOLDER / image_name
    if path.exists():
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    build_index()
    return RedirectResponse(url="/images", status_code=302)


@app.get("/images/delete/{image_name}")
async def delete_image(image_name: str):
    path = RECOMMEND_FOLDER / image_name
    if path.exists():
        os.remove(path)
    build_index()
    return RedirectResponse(url="/images", status_code=302)
