import numpy as np
from PIL import Image
import cv2

def get_image_embedding(image):
    """Get simple image embedding using basic features"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Resize to standard size
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert to grayscale and normalize
        if len(img_resized.shape) == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_resized
            
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        # Flatten and create a simple feature vector
        features = gray.flatten()
        
        # Pad or truncate to 512 dimensions
        if len(features) > 512:
            features = features[:512]
        elif len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), 'constant')
            
        # Normalize the feature vector
        features = features / np.linalg.norm(features)
        
        return features
        
    except Exception as e:
        print(f"Error in image embedding: {e}")
        # Return a zero vector as fallback
        return np.zeros(512)