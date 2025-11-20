import numpy as np
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
    """
    try:
        # Ensure image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess the image
        image_tensor = preprocess(img).unsqueeze(0)
        
        # Generate embedding with no gradient computation
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            # Normalize for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().astype("float32").flatten()
            
    except Exception as e:
        print(f"Error in image embedding: {e}")
        # Return a zero vector as fallback
        return np.zeros(512, dtype='float32')