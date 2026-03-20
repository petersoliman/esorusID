import logging
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Colors for bounding boxes (one per detected object, cycling)
DETECTION_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#FF8C42', '#A8E6CF',
]

_yolo_model = None


def _get_yolo(conf_threshold: float = 0.3):
    """Lazy-load YOLOv8 nano model (downloads weights on first call)."""
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed")
        logging.info("Loading YOLOv8 model...")
        _yolo_model = YOLO('yolov8n.pt')
        logging.info("YOLOv8 model loaded.")
    return _yolo_model


def detect_products(image: Image.Image, conf_threshold: float = 0.3) -> list:
    """
    Detect objects in a PIL image using YOLOv8.

    Returns a list of dicts:
        {
            'label':      str,          # COCO class name (e.g. 'chair')
            'confidence': float,        # detection confidence 0–1
            'bbox':       (x1,y1,x2,y2),
            'crop':       PIL.Image,    # cropped region
        }
    """
    model = _get_yolo()
    results = model(image, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Clamp to image bounds
            w, h = image.size
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
            })

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
