from PIL import Image
import torch
import open_clip

# Load model once at module level for efficiency
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()

def get_image_embedding(img):
    """
    Generate embedding for an image using OpenCLIP model.

    Args:
        img: PIL Image object (RGB)

    Returns:
        numpy array: Image embedding (normalized for cosine similarity)

    Raises:
        Exception: propagated directly so callers can handle failures explicitly
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    image_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype("float32").flatten()