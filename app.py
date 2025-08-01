from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import uuid
import torch
import faiss
from PIL import Image
import numpy as np
import open_clip
import logging
import json
from index_images import index_images

logging.basicConfig(level=logging.INFO)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

app = FastAPI()

# Use Railway persistent storage if available, otherwise use local paths
if os.path.exists('/app/data'):
    # Railway persistent storage
    DATA_DIR = Path('/app/data')
    STATIC_DIR = Path('/app/static')
else:
    # Local development
    DATA_DIR = Path('data')
    STATIC_DIR = Path('static')

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = STATIC_DIR / "uploads"
RECOMMEND_FOLDER = STATIC_DIR / "recommendations"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RECOMMEND_FOLDER.mkdir(parents=True, exist_ok=True)

# Index file paths (created by index_images.py)
INDEX_PATH = DATA_DIR / "image_index.faiss"
MAPPING_PATH = DATA_DIR / "image_paths.json"

# Global variables
index = None
image_filenames = []
model = None
preprocess = None
device = "cpu"


def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.filename:
        return False
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check file size (if available)
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        return False
    
    return True


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


def load_index():
    """Load the pre-built index from files"""
    global index, image_filenames
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
            index = faiss.read_index(INDEX_PATH)
            with open(MAPPING_PATH, "r") as f:
                image_filenames = json.load(f)
            print(f"✅ Loaded index with {len(image_filenames)} images")
            return True
        else:
            print("❌ Index files not found. Run index_images.py first.")
            index = None
            image_filenames = []
            return False
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        index = None
        image_filenames = []
        return False


@app.on_event("startup")
async def startup_event():
    try:
        logging.info("Loading model...")
        global model, preprocess
        load_model()
        logging.info("Model loaded successfully.")

        logging.info("Loading pre-built index...")
        load_index()
        logging.info("Index loading completed.")
    except Exception as e:
        logging.exception("Startup failed due to exception:")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile):
    try:
        # Validate file
        if not validate_image_file(file):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Please upload a valid image file (JPG, PNG, GIF, BMP, WebP) under 10MB."
            })
        
        contents = await file.read()
        
        # Check file size after reading
        if len(contents) > MAX_FILE_SIZE:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "File size too large. Please upload an image under 10MB."
            })
        
        ext = file.filename.split(".")[-1]
        name = f"{uuid.uuid4()}.{ext}"
        path = UPLOAD_FOLDER / name
        
        with open(path, "wb") as f:
            f.write(contents)

        # Validate image can be opened
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            os.remove(path)  # Clean up invalid file
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Invalid image file. Please upload a valid image."
            })

        query_feat = extract_features(img)

        # Check if index exists and has images
        if index is None:
            return templates.TemplateResponse("results.html", {
                "request": request, 
                "uploaded": name, 
                "results": [],
                "message": "No search index available. Please run index_images.py to build the index."
            })
        
        if len(image_filenames) == 0:
            return templates.TemplateResponse("results.html", {
                "request": request, 
                "uploaded": name, 
                "results": [],
                "message": "No images in the index. Please run index_images.py to build the index."
            })

        D, I = index.search(np.expand_dims(query_feat, axis=0), k=12)
        results = [image_filenames[i] for i in I[0]]
        return templates.TemplateResponse("results.html", {"request": request, "uploaded": name, "results": results})
        
    except Exception as e:
        logging.error(f"Search error: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "An error occurred during search. Please try again."
        })


@app.get("/results/{image_name}", response_class=HTMLResponse)
async def show_results(request: Request, image_name: str):
    # Check if the image exists
    image_path = UPLOAD_FOLDER / image_name
    if not image_path.exists():
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse("results.html", {"request": request, "uploaded": image_name, "results": []})


@app.get("/images", response_class=HTMLResponse)
async def list_images(request: Request):
    images = [f.name for f in RECOMMEND_FOLDER.iterdir() if f.is_file()]
    return templates.TemplateResponse("list_images.html", {"request": request, "images": images})


@app.get("/images/add", response_class=HTMLResponse)
async def add_image_form(request: Request):
    return templates.TemplateResponse("add_image.html", {"request": request})


@app.post("/images/add")
async def add_image(request: Request, file: UploadFile):
    try:
        # Validate file
        if not validate_image_file(file):
            return RedirectResponse(url="/images?error=Please upload a valid image file under 10MB", status_code=302)
        
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            return RedirectResponse(url="/images?error=File size too large. Please upload an image under 10MB", status_code=302)
        
        ext = file.filename.split(".")[-1]
        name = f"{uuid.uuid4()}.{ext}"
        path = RECOMMEND_FOLDER / name
        
        with open(path, "wb") as f:
            f.write(contents)
            
        return RedirectResponse(url="/images?message=Image added successfully. Run index_images.py to update the search index.", status_code=302)
        
    except Exception as e:
        logging.error(f"Add image error: {e}")
        return RedirectResponse(url="/images?error=Failed to add image", status_code=302)


@app.get("/images/edit/{image_name}", response_class=HTMLResponse)
async def edit_image_form(request: Request, image_name: str):
    return templates.TemplateResponse("edit_image.html", {"request": request, "image": image_name})


@app.post("/images/edit/{image_name}")
async def edit_image(image_name: str, file: UploadFile):
    try:
        # Validate file
        if not validate_image_file(file):
            return RedirectResponse(url="/images?error=Please upload a valid image file under 10MB", status_code=302)
        
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            return RedirectResponse(url="/images?error=File size too large. Please upload an image under 10MB", status_code=302)
        
        path = RECOMMEND_FOLDER / image_name
        if path.exists():
            with open(path, "wb") as f:
                f.write(contents)
            return RedirectResponse(url="/images?message=Image updated successfully. Run index_images.py to update the search index.", status_code=302)
        else:
            return RedirectResponse(url="/images?error=Image not found", status_code=302)
            
    except Exception as e:
        logging.error(f"Edit image error: {e}")
        return RedirectResponse(url="/images?error=Failed to update image", status_code=302)


@app.get("/images/delete/{image_name}")
async def delete_image(image_name: str):
    try:
        path = RECOMMEND_FOLDER / image_name
        if path.exists():
            os.remove(path)
            return RedirectResponse(url="/images?message=Image deleted successfully. Run index_images.py to update the search index.", status_code=302)
        else:
            return RedirectResponse(url="/images?error=Image not found", status_code=302)
    except Exception as e:
        logging.error(f"Delete image error: {e}")
        return RedirectResponse(url="/images?error=Failed to delete image", status_code=302)


@app.get("/reindex")
def reindex():
    try:
        index_images()
        return {"status": "success", "message": "Reindexing complete"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
