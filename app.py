from fastapi import FastAPI, Request, UploadFile, HTTPException, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List
import os
import io
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
CSS_VERSION = "1.3"
SIMILARITY_THRESHOLD = 0.2     # min cosine similarity to include a FAISS result
YOLO_CONF_THRESHOLD = 0.35     # min YOLO confidence to accept a detection
MAX_RESULTS_PER_OBJECT = 5

# API key for the /reindex endpoint — set REINDEX_API_KEY env var to enable auth
REINDEX_API_KEY = os.environ.get("REINDEX_API_KEY")

# API key for /api/search — set API_KEY env var to require X-API-Key header.
# If unset, /api/search is open (for dev).
API_KEY = os.environ.get("API_KEY")


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
        name,
        {"request": request, "css_version": CSS_VERSION,
         "detection_colors": DETECTION_COLORS if DETECTION_AVAILABLE else [], **kwargs}
    )


def _faiss_search(feat, k: int) -> tuple:
    """Search the FAISS index and return (distances, indices)."""
    k = min(k, len(image_filenames))
    return index.search(np.expand_dims(feat, axis=0), k=k)


def _product_id_from_filename(filename):
    """Extract int product_id from '{id}_{variant}.ext' filenames; None otherwise."""
    try:
        return int(str(filename).split('_', 1)[0])
    except (ValueError, AttributeError):
        return None


def _make_matches(D_row, I_row, threshold, limit):
    """Build the JSON-shape `matches` list from a FAISS search row.

    threshold=None means no similarity filtering (used for the whole-image fallback,
    which matches the pre-refactor behaviour of _single_image_search).
    """
    matches = []
    for j in range(len(I_row)):
        idx = int(I_row[j])
        if idx < 0 or idx >= len(image_filenames):
            continue
        score = float(D_row[j])
        if threshold is not None and score < threshold:
            continue
        fname = image_filenames[idx]
        matches.append({
            "product_id": _product_id_from_filename(fname),
            "similarity_score": round(score, 4),
            "image_filename": fname,
        })
        if len(matches) >= limit:
            break
    return matches


def _run_search_pipeline(img):
    """Shared YOLO → embed → FAISS → filter pipeline used by both /search and /api/search.

    Callers must pre-check ML_AVAILABLE and that the index is loaded.

    Returns a list of detection groups with keys:
        label (str), confidence (float), matches (list of JSON-shape dicts),
        bbox (list[int] or None), crop (PIL.Image or None).

    `bbox`/`crop` are populated for YOLO detections and are None for the
    whole-image fallback group. Returns [] only if feature extraction fails
    on the whole-image fallback.
    """
    raw_detections = []
    if DETECTION_AVAILABLE:
        try:
            raw_detections = detect_products(img, conf_threshold=YOLO_CONF_THRESHOLD)
        except Exception:
            logging.error("Object detection failed, falling back to single-image search", exc_info=True)
            raw_detections = []

    groups = []
    for i, det in enumerate(raw_detections):
        try:
            feat = extract_features(det['crop'])
        except Exception:
            logging.warning(f"Could not embed crop {i} ({det['label']}), skipping", exc_info=True)
            continue

        D, I = _faiss_search(feat, k=10)
        matches = _make_matches(D[0], I[0], SIMILARITY_THRESHOLD, MAX_RESULTS_PER_OBJECT)
        if not matches:
            continue

        groups.append({
            "label": det['label'],
            "confidence": float(det['confidence']),
            "matches": matches,
            "bbox": list(det['bbox']),
            "crop": det['crop'],
        })

    if groups:
        return groups

    # No YOLO detections (or all filtered) — fall back to whole-image search.
    try:
        feat = extract_features(img)
    except Exception:
        logging.error("Feature extraction error on whole image", exc_info=True)
        return []

    D, I = _faiss_search(feat, k=MAX_RESULTS_PER_OBJECT)
    matches = _make_matches(D[0], I[0], threshold=None, limit=MAX_RESULTS_PER_OBJECT)
    return [{
        "label": "image",
        "confidence": 1.0,
        "matches": matches,
        "bbox": None,
        "crop": None,
    }]


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
        "api_search_available": True,
    }


@app.post("/api/search")
async def api_search(request: Request, file: UploadFile = File(...)):
    """Machine-readable search endpoint for marketplace integration."""
    if API_KEY:
        provided = request.headers.get("X-API-Key")
        if provided != API_KEY:
            return JSONResponse(
                {"status": "error", "message": "Unauthorized"},
                status_code=401,
            )

    try:
        if not validate_image_file(file):
            return JSONResponse(
                {"status": "error", "message": "Invalid image file"},
                status_code=400,
            )

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return JSONResponse(
                {"status": "error", "message": "Invalid image file"},
                status_code=400,
            )

        if not ML_AVAILABLE:
            return JSONResponse(
                {"status": "error", "message": "ML not available"},
                status_code=503,
            )

        if index is None or len(image_filenames) == 0:
            return JSONResponse(
                {"status": "error", "message": "No search index available"},
                status_code=503,
            )

        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return JSONResponse(
                {"status": "error", "message": "Invalid image file"},
                status_code=400,
            )

        groups = _run_search_pipeline(img)

        detections = [
            {
                "label": g["label"],
                "confidence": round(float(g["confidence"]), 4),
                "matches": g["matches"],
            }
            for g in groups
        ]

        return JSONResponse({"status": "ok", "detections": detections})

    except Exception:
        logging.error("api_search error", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": "Server error"},
            status_code=500,
        )


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

        groups = _run_search_pipeline(img)

        if not groups:
            return _template("index.html", request,
                             error="Error processing image. Please try again.")

        multi_mode = groups[0]["bbox"] is not None

        result_groups = []
        for i, g in enumerate(groups):
            gid = i + 1

            if g["crop"] is not None:
                crop_name = f"crop_{uuid.uuid4()}_{i}.jpg"
                try:
                    g["crop"].convert("RGB").save(UPLOAD_FOLDER / crop_name, "JPEG")
                except Exception:
                    logging.warning(f"Failed to save crop {i}", exc_info=True)
                    crop_name = name
            else:
                crop_name = name

            color = (
                DETECTION_COLORS[(gid - 1) % len(DETECTION_COLORS)]
                if DETECTION_AVAILABLE and DETECTION_COLORS
                else '#4ECDC4'
            )

            result_groups.append({
                'id': gid,
                'label': g['label'],
                'confidence': g['confidence'],
                'crop_filename': crop_name,
                'results': [m['image_filename'] for m in g['matches']],
                'bbox': g['bbox'],
                'color': color,
            })

        annotated_name = name
        if multi_mode:
            try:
                annotated_img = draw_annotations(img, result_groups)
                annotated_name = f"annotated_{uuid.uuid4()}.jpg"
                annotated_img.convert("RGB").save(UPLOAD_FOLDER / annotated_name, "JPEG")
            except Exception:
                logging.warning("Failed to draw annotations", exc_info=True)
                annotated_name = name

        return _template("results.html", request,
                         uploaded=name,
                         annotated=annotated_name,
                         detections=result_groups,
                         multi_mode=multi_mode)

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
    return _template("list_images.html", request, images=images,
                    index_loaded=index is not None, images_indexed=len(image_filenames))


@app.post("/images/bulk-add")
async def bulk_add_images(files: List[UploadFile] = File(...)):
    added = []
    errors = []

    for file in files:
        try:
            if not validate_image_file(file):
                errors.append({"filename": file.filename, "error": "Invalid file type or size"})
                continue

            contents = await file.read()

            if len(contents) > MAX_FILE_SIZE:
                errors.append({"filename": file.filename, "error": "File too large"})
                continue

            ext = Path(file.filename).suffix.lstrip(".")
            name = f"{uuid.uuid4()}.{ext}"
            path = RECOMMEND_FOLDER / name

            with open(path, "wb") as f:
                f.write(contents)

            added.append(name)
        except Exception as e:
            logging.error(f"Error processing {file.filename}: {e}")
            errors.append({"filename": file.filename, "error": str(e)})

    return JSONResponse({"added": added, "errors": errors})


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


@app.get("/images-json")
async def images_json():
    images = []
    for root, dirs, files in os.walk(RECOMMEND_FOLDER):
        for file in files:
            if file.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(RECOMMEND_FOLDER)
                images.append(str(rel_path))
    return JSONResponse({
        "images": images,
        "index_loaded": index is not None,
        "images_indexed": len(image_filenames)
    })


@app.post("/images/delete/{image_name:path}")
async def delete_image(image_name: str):
    try:
        path = _safe_recommend_path(image_name)
        if path.exists():
            os.remove(path)
            return JSONResponse({"status": "success", "message": "Image deleted successfully"})
        else:
            return JSONResponse({"status": "error", "message": "Image not found"}, status_code=404)
    except HTTPException:
        return JSONResponse({"status": "error", "message": "Invalid image path"}, status_code=400)
    except Exception as e:
        logging.error("Delete image error", exc_info=True)
        return JSONResponse({"status": "error", "message": "Failed to delete image"}, status_code=500)


@app.post("/search-crops")
async def search_crops(request: Request):
    """
    Accept manually selected crop coordinates and search for similar products.

    Expected JSON body:
        {
            "image": "uuid.jpg",          # filename inside uploads/
            "crops": [
                {"x1": 100, "y1": 50, "x2": 300, "y2": 200, "label": "Selection 1"},
                ...
            ]
        }
    Returns JSON with a list of result groups, same structure as multi-mode.
    """
    try:
        data = await request.json()
        image_name = data.get('image', '')
        crops = data.get('crops', [])

        if not image_name or not crops:
            return JSONResponse({'error': 'Missing image or crops'}, status_code=400)

        # Security: ensure the image is inside the uploads folder
        image_path = UPLOAD_FOLDER / image_name
        try:
            image_path.resolve().relative_to(UPLOAD_FOLDER.resolve())
        except ValueError:
            return JSONResponse({'error': 'Invalid image path'}, status_code=400)

        if not image_path.exists():
            return JSONResponse({'error': 'Image not found'}, status_code=404)

        if not ML_AVAILABLE:
            return JSONResponse({'error': 'ML not available'}, status_code=503)

        if index is None or len(image_filenames) == 0:
            return JSONResponse({'error': 'No search index available'}, status_code=503)

        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size

        # Colors for manual selections (distinct from auto-detection colors)
        MANUAL_COLORS = ['#E67E22', '#8E44AD', '#16A085', '#C0392B', '#2980B9', '#27AE60']

        groups = []
        for i, crop_data in enumerate(crops):
            x1 = max(0, int(crop_data.get('x1', 0)))
            y1 = max(0, int(crop_data.get('y1', 0)))
            x2 = min(img_w, int(crop_data.get('x2', img_w)))
            y2 = min(img_h, int(crop_data.get('y2', img_h)))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            try:
                feat = extract_features(crop)
            except Exception:
                logging.warning("Could not embed manual crop %d", i, exc_info=True)
                continue

            D, I = _faiss_search(feat, k=10)
            good_results = [
                image_filenames[I[0][j]]
                for j in range(len(I[0]))
                if I[0][j] < len(image_filenames) and D[0][j] >= SIMILARITY_THRESHOLD
            ][:MAX_RESULTS_PER_OBJECT]

            if not good_results:
                continue

            # Save crop thumbnail
            crop_name = f"manual_crop_{uuid.uuid4()}_{i}.jpg"
            try:
                crop.convert('RGB').save(UPLOAD_FOLDER / crop_name, 'JPEG')
            except Exception:
                logging.warning("Failed to save manual crop %d", i, exc_info=True)
                crop_name = image_name  # fallback to original

            groups.append({
                'id': i + 1,
                'label': crop_data.get('label', f'Selection {i + 1}'),
                'crop_filename': crop_name,
                'results': good_results,
                'color': MANUAL_COLORS[i % len(MANUAL_COLORS)],
                'bbox': [x1, y1, x2, y2],
            })

        return JSONResponse({'groups': groups})

    except Exception:
        logging.error("search-crops error", exc_info=True)
        return JSONResponse({'error': 'Server error'}, status_code=500)


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
