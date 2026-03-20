import logging
import os
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

# Colors for bounding boxes (one per detected object, cycling)
DETECTION_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#FF8C42', '#A8E6CF',
]

# Roboflow config — set these in your .env or environment
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')
ROBOFLOW_MODEL_ID = os.environ.get('ROBOFLOW_MODEL_ID', 'furniture-detection/2')

_yolo_model = None
_roboflow_client = None


def _get_yolo():
    """Lazy-load YOLOv8l model (downloads weights on first call)."""
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed")
        logging.info("Loading YOLOv8l model...")
        _yolo_model = YOLO('yolov8l.pt')
        logging.info("YOLOv8l model loaded.")
    return _yolo_model


def _get_roboflow_client():
    """Lazy-load Roboflow inference client (only if API key is set)."""
    global _roboflow_client
    if _roboflow_client is None:
        if not ROBOFLOW_AVAILABLE:
            return None
        if not ROBOFLOW_API_KEY:
            return None
        try:
            _roboflow_client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=ROBOFLOW_API_KEY,
            )
            logging.info("Roboflow client initialized (model: %s).", ROBOFLOW_MODEL_ID)
        except Exception as e:
            logging.warning("Failed to init Roboflow client: %s", e)
            return None
    return _roboflow_client


def _compute_iou(b1, b2):
    """Compute IoU between two boxes given as (x1, y1, x2, y2)."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def detect_products(image: Image.Image, conf_threshold: float = 0.2) -> list:
    """
    Detect furniture objects in a PIL image.
    Uses YOLOv8l (COCO classes) + Roboflow furniture model (if ROBOFLOW_API_KEY is set).
    Deduplicates overlapping boxes from both models via IoU.

    Returns a list of dicts:
        {
            'label':      str,
            'confidence': float,
            'bbox':       (x1, y1, x2, y2),
            'crop':       PIL.Image,
            'source':     str,   # 'yolo' or 'roboflow'
        }
    """
    w, h = image.size
    detections = []

    # ── YOLOv8l ───────────────────────────────────────────────────────────────
    try:
        model = _get_yolo()
        results = model(image, conf=conf_threshold, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                crop = image.crop((x1, y1, x2, y2))
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'crop': crop,
                    'source': 'yolo',
                })
    except Exception as e:
        logging.error("YOLOv8 detection failed: %s", e, exc_info=True)

    # ── Roboflow furniture model ───────────────────────────────────────────────
    try:
        client = _get_roboflow_client()
        if client:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name, 'JPEG')
                tmp_path = tmp.name
            try:
                rf_result = client.infer(tmp_path, model_id=ROBOFLOW_MODEL_ID)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            for pred in rf_result.get('predictions', []):
                label = pred['class']
                confidence = pred['confidence']
                if confidence < conf_threshold:
                    continue
                cx, cy = pred['x'], pred['y']
                bw, bh = pred['width'], pred['height']
                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(w, int(cx + bw / 2))
                y2 = min(h, int(cy + bh / 2))
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = (x1, y1, x2, y2)
                # Skip if it significantly overlaps an existing detection
                if any(_compute_iou(bbox, d['bbox']) > 0.5 for d in detections):
                    continue
                crop = image.crop(bbox)
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': bbox,
                    'crop': crop,
                    'source': 'roboflow',
                })
    except Exception as e:
        logging.warning("Roboflow detection skipped: %s", e)

    return detections


def draw_annotations(image: Image.Image, matched_detections: list) -> Image.Image:
    """
    Draw numbered, colored bounding boxes on a copy of the image.

    matched_detections must contain dicts with keys:
        'id', 'label', 'confidence', 'bbox'
    (i.e. only the detections that have search results, already numbered)
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for det in matched_detections:
        idx = det['id'] - 1  # 0-based for color selection
        color = DETECTION_COLORS[idx % len(DETECTION_COLORS)]
        x1, y1, x2, y2 = det['bbox']

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Label pill background + text
        label_text = f"{det['id']}. {det['label']} ({det['confidence']:.0%})"
        pill_w = len(label_text) * 7 + 8
        pill_h = 20
        draw.rectangle([x1, y1, x1 + pill_w, y1 + pill_h], fill=color)
        draw.text((x1 + 4, y1 + 3), label_text, fill='white')

    return annotated
