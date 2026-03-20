from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import uuid
import logging
import json
from contextlib import asynccontextmanager

# Try to import ML dependencies, but don't fail if they're not available
try:
    import faiss
    from PIL import Image
    import numpy as np
    from utils import get_image_embedding
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML dependencies not available: {e}")
    ML_AVAILABLE = False

try:
    from index_images import index_images
    INDEXING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Indexing not available: {e}")
    INDEXING_AVAILABLE = False

try:
    from detect_objects import detect_products, draw_annotations, DETECTION_COLORS
    DETECTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Object detection not available: {e}")
    DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
CSS_VERSION = "1.2"
SIMILARITY_THRESHOLD = 0.2     # min cosine similarity to include a FAISS result
YOLO_CONF_THRESHOLD = 0.3      # min YOLO confidence to accept a detection
MAX_RESULTS_PER_OBJECT = 5

# API key for the /reindex endpoint — set REINDEX_API_KEY env var to enable auth
REINDEX_API_KEY = os.environ.get("REINDEX_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: clear temporary uploads and load index
    _clear_uploads()
    try:
        logging.info("Loading pre-built index...")
        load_index()
        logging.info("Index loading completed.")
    except Exception as e:
        logging.exception("Startup failed due to exception:")
        logging.warning("Startup encountered issues but continuing...")

    yield

    # Shutdown
    logging.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Use Railway persistent storage if available, otherwise use local paths
if os.path.exists('/app/data'):
    DATA_DIR = Path('/app/data')
    STATIC_DIR = Path('/app/static')
else:
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


def _clear_uploads():
    """Delete all temporary search uploads on startup."""
    if UPLOAD_FOLDER.exists():
        for f in UPLOAD_FOLDER.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass


def _safe_recommend_path(image_name: str) -> Path:
    """Resolve image_name inside RECOMMEND_FOLDER and reject path traversal."""
    resolved = (RECOMMEND_FOLDER / image_name).resolve()
    try:
        resolved.relative_to(RECOMMEND_FOLDER.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image path")
    return resolved


def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.filename:
        return False
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        return False
    return True


def extract_features(image):
    """Extract CLIP features from a PIL Image."""
    if not ML_AVAILABLE:
        raise ImportError("ML dependencies not available")
    if not hasattr(image, 'convert'):
        raise ValueError("Image must be a PIL Image object")
    return get_image_embedding(image)


def load_index():
    """Load the pre-built FAISS index from files"""
    global index, image_filenames
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
            index = faiss.read_index(str(INDEX_PATH))
            with open(MAPPING_PATH, "r") as f:
                image_filenames = json.load(f)
            logging.info(f"Loaded index with {len(image_filenames)} images")
            return True
        else:
            logging.warning("Index files not found. Run index_images.py first.")
            index = None
            image_filenames = []
            return False
    except Exception as e:
        logging.error(f"Error loading index: {e}")
        index = None
        image_filenames = []
        return False


def _template(name: str, request: Request, **kwargs):
    return templates.TemplateResponse(
        name, {"request": request, "css_version": CSS_VERSION,
               "detection_colors": DETECTION_COLORS if DETECTION_AVAILABLE else [], **kwargs}
    )


def _faiss_search(feat, k: int) -> tuple:
    """Search the FAISS index and return (distances, indices)."""
    k = min(k, len(image_filenames))
    return index.search(np.expand_dims(feat, axis=0), k=k)


def _single_image_search(img, uploaded_name: str, request: Request):
    """Fall-back: treat the whole uploaded image as one query."""
    try:
        feat = extract_features(img)
    except Exception:
        logging.error("Feature extraction error", exc_info=True)
        return _template("index.html", request,
                         error="Error processing image. Please try again.")

    if index is None or len(image_filenames) == 0:
        return _template("results.html", request,
                         uploaded=uploaded_name, annotated=uploaded_name,
                         detections=[], multi_mode=False,
                         message="No search index available. Please run index_images.py first.")

    D, I = _faiss_search(feat, k=MAX_RESULTS_PER_OBJECT)
    results = [image_filenames[i] for i in I[0] if i < len(image_filenames)]

    # Wrap as a single detection group so results.html can use the same template path
    detection_group = {
        'id': 1,
        'label': 'image',
        'confidence': 1.0,
        'crop_filename': uploaded_name,
        'results': results,
        'bbox': None,
        'color': DETECTION_COLORS[0] if DETECTION_AVAILABLE else '#4ECDC4',
    }
    return _template("results.html", request,
                     uploaded=uploaded_name, annotated=uploaded_name,
                     detections=[detection_group], multi_mode=False)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return _template("index.html", request)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "detection_available": DETECTION_AVAILABLE,
        "indexing_available": INDEXING_AVAILABLE,
        "index_loaded": index is not None,
        "images_count": len(image_filenames) if image_filenames else 0,
    }


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile):
    try:
        if not validate_image_file(file):
            return _template("index.html", request,
                             error="Please upload a valid image file (JPG, PNG, GIF, BMP, WebP) under 10MB.")

        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            return _template("index.html", request,
                             error="File size too large. Please upload an image under 10MB.")

        ext = Path(file.filename).suffix.lstrip(".")
        name = f"{uuid.uuid4()}.{ext}"
        path = UPLOAD_FOLDER / name

        with open(path, "wb") as f:
            f.write(contents)

        # Open and validate image
        try:
            if not ML_AVAILABLE:
                return _template("index.html", request,
                                 error="ML functionality not available. Please install required dependencies.")
            img = Image.open(path).convert("RGB")
        except Exception:
            os.remove(path)
            return _template("index.html", request,
                             error="Invalid image file. Please upload a valid image.")

        # Check index readiness
        if index is None or len(image_filenames) == 0:
            return _template("results.html", request,
                             uploaded=name, annotated=name,
                             detections=[], multi_mode=False,
                             message="No search index available. Please run index_images.py first.")

        # ── Multi-product detection path ─────────────────────────────────────
        if DETECTION_AVAILABLE:
            try:
                raw_detections = detect_products(img, conf_threshold=YOLO_CONF_THRESHOLD)
            except Exception:
                logging.error("Object detection failed, falling back to single-image search", exc_info=True)
                raw_detections = []
        else:
            raw_detections = []

        if not raw_detections:
            # Fall back to whole-image search
            return _single_image_search(img, name, request)

        # ── Process each detected object ─────────────────────────────────────
        result_groups = []
        group_id = 1

        for i, det in enumerate(raw_detections):
            try:
                feat = extract_features(det['crop'])
            except Exception:
                logging.warning(f"Could not embed crop {i} ({det['label']}), skipping", exc_info=True)
                continue

            D, I = _faiss_search(feat, k=10)

            good_results = [
                image_filenames[I[0][j]]
                for j in range(len(I[0]))
                if I[0][j] < len(image_filenames) and D[0][j] >= SIMILARITY_THRESHOLD
            ][:MAX_RESULTS_PER_OBJECT]

            if not good_results:
                continue  # no sufficiently similar products — skip this detection

            # Save crop thumbnail
            crop_name = f"crop_{uuid.uuid4()}_{i}.jpg"
            try:
                det['crop'].convert("RGB").save(UPLOAD_FOLDER / crop_name, "JPEG")
            except Exception:
                logging.warning(f"Failed to save crop {i}", exc_info=True)
                crop_name = name  # fall back to original image

            result_groups.append({
                'id': group_id,
                'label': det['label'],
                'confidence': det['confidence'],
                'crop_filename': crop_name,
                'results': good_results,
                'bbox': list(det['bbox']),
                'color': DETECTION_COLORS[(group_id - 1) % len(DETECTION_COLORS)],
            })
            group_id += 1

        if not result_groups:
            # All detections were filtered out — fall back to whole-image search
            return _single_image_search(img, name, request)

        # Draw bounding boxes only for matched detections
        try:
            annotated_img = draw_annotations(img, result_groups)
            annotated_name = f"annotated_{uuid.uuid4()}.jpg"
            annotated_img.convert("RGB").save(UPLOAD_FOLDER / annotated_name, "JPEG")
        except Exception:
            logging.warning("Failed to draw annotations", exc_info=True)
            annotated_name = name  # fall back to original

        return _template("results.html", request,
                         uploaded=name,
                         annotated=annotated_name,
                         detections=result_groups,
                         multi_mode=True)

    except Exception:
        logging.error("Search error", exc_info=True)
        return _template("index.html", request,
                         error="An unexpected error occurred. Please try again.")


@app.get("/results/{image_name:path}", response_class=HTMLResponse)
async def show_results(request: Request, image_name: str):
    image_path = UPLOAD_FOLDER / image_name
    if not image_path.exists():
        return RedirectResponse(url="/", status_code=302)
    return _template("results.html", request, uploaded=image_name,
                     annotated=image_name, detections=[], multi_mode=False)


@app.get("/images", response_class=HTMLResponse)
async def list_images(request: Request):
    images = []
    for root, dirs, files in os.walk(RECOMMEND_FOLDER):
        for file in files:
            if file.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(RECOMMEND_FOLDER)
                images.append(str(rel_path))
    return _template("list_images.html", request, images=images)


@app.get("/images/add", response_class=HTMLResponse)
async def add_image_form(request: Request):
    return _template("add_image.html", request)


@app.post("/images/add")
async def add_image(request: Request, file: UploadFile):
    try:
        if not validate_image_file(file):
            return RedirectResponse(url="/images?error=Please upload a valid image file under 10MB", status_code=302)

        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            return RedirectResponse(url="/images?error=File size too large. Please upload an image under 10MB", status_code=302)

        ext = Path(file.filename).suffix.lstrip(".")
        name = f"{uuid.uuid4()}.{ext}"
        path = RECOMMEND_FOLDER / name

        with open(path, "wb") as f:
            f.write(contents)

        return RedirectResponse(url="/images?message=Image added successfully. Run index_images.py to update the search index.", status_code=302)

    except Exception:
        logging.error("Add image error", exc_info=True)
        return RedirectResponse(url="/images?error=Failed to add image", status_code=302)


@app.get("/images/edit/{image_name:path}", response_class=HTMLResponse)
async def edit_image_form(request: Request, image_name: str):
    _safe_recommend_path(image_name)
    return _template("edit_image.html", request, image=image_name)


@app.post("/images/edit/{image_name:path}")
async def edit_image(image_name: str, file: UploadFile):
    try:
        path = _safe_recommend_path(image_name)

        if not validate_image_file(file):
            return RedirectResponse(url="/images?error=Please upload a valid image file under 10MB", status_code=302)

        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            return RedirectResponse(url="/images?error=File size too large. Please upload an image under 10MB", status_code=302)

        if path.exists():
            with open(path, "wb") as f:
                f.write(contents)
            return RedirectResponse(url="/images?message=Image updated successfully. Run index_images.py to update the search index.", status_code=302)
        else:
            return RedirectResponse(url="/images?error=Image not found", status_code=302)

    except HTTPException:
        return RedirectResponse(url="/images?error=Invalid image path", status_code=302)
    except Exception:
        logging.error("Edit image error", exc_info=True)
        return RedirectResponse(url="/images?error=Failed to update image", status_code=302)


@app.post("/images/delete/{image_name:path}")
async def delete_image(image_name: str):
    try:
        path = _safe_recommend_path(image_name)
        if path.exists():
            os.remove(path)
            return RedirectResponse(url="/images?message=Image deleted successfully. Run index_images.py to update the search index.", status_code=302)
        else:
            return RedirectResponse(url="/images?error=Image not found", status_code=302)
    except HTTPException:
        return RedirectResponse(url="/images?error=Invalid image path", status_code=302)
    except Exception:
        logging.error("Delete image error", exc_info=True)
        return RedirectResponse(url="/images?error=Failed to delete image", status_code=302)


@app.get("/reindex")
def reindex(api_key: str = ""):
    if REINDEX_API_KEY and api_key != REINDEX_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    try:
        if not INDEXING_AVAILABLE:
            return {"status": "error", "message": "Indexing functionality not available. ML dependencies may not be installed."}

        if not ML_AVAILABLE:
            return {"status": "error", "message": "ML dependencies not available. Please install required packages."}

        index_images()

        global index, image_filenames
        if load_index():
            return {"status": "success", "message": f"Reindexing complete. Loaded {len(image_filenames)} images."}
        else:
            return {"status": "error", "message": "Reindexing completed but failed to load the index."}
    except Exception:
        logging.error("Reindex error", exc_info=True)
        return {"status": "error", "message": "Reindexing failed. Check server logs for details."}
