import logging
import os
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLOWorld
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Colors for bounding boxes (one per detected object, cycling)
DETECTION_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#FF8C42', '#A8E6CF',
]

# Furniture classes for YOLOWorld — covers common interior items
FURNITURE_CLASSES = [
    "chair", "armchair", "sofa", "couch", "loveseat",
    "dining table", "coffee table", "side table", "end table", "console table",
    "desk", "writing desk", "office desk",
    "floor lamp", "table lamp", "pendant lamp", "chandelier",
    "bed", "headboard", "nightstand", "dresser", "wardrobe", "cabinet",
    "bookshelf", "bookcase", "shelf", "tv stand", "media console",
    "ottoman", "footstool", "bench", "stool", "bar stool",
    "rug", "carpet", "curtain", "mirror", "artwork", "plant",
]

_yolo_model = None


def _get_yolo():
    """Lazy-load YOLOWorld model (downloads weights on first call)."""
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed")
        logging.info("Loading YOLOWorld model...")
        _yolo_model = YOLOWorld('yolov8x-worldv2.pt')
        _yolo_model.set_classes(FURNITURE_CLASSES)
        logging.info("YOLOWorld model loaded with %d furniture classes.", len(FURNITURE_CLASSES))
    return _yolo_model


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


def detect_products(image: Image.Image, conf_threshold: float = 0.1) -> list:
    """
    Detect furniture objects in a PIL image using YOLOWorld (fully local, no API).

    Returns a list of dicts:
        {
            'label':      str,
            'confidence': float,
            'bbox':       (x1, y1, x2, y2),
            'crop':       PIL.Image,
            'source':     str,   # always 'yoloworld'
        }
    """
    w, h = image.size
    detections = []

    try:
        model = _get_yolo()
        results = model.predict(image, conf=conf_threshold, verbose=False)
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
                    'source': 'yoloworld',
                })
    except Exception as e:
        logging.error("YOLOWorld detection failed: %s", e, exc_info=True)

    # Deduplicate heavily overlapping boxes (IoU > 0.5)
    deduped = []
    for det in detections:
        if not any(_compute_iou(det['bbox'], d['bbox']) > 0.5 for d in deduped):
            deduped.append(det)

    return deduped


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
