import os
import json
import faiss
import numpy as np
from utils import get_image_embedding
from PIL import Image
from tqdm import tqdm
import gc
from pathlib import Path

# Use Railway persistent storage if available, otherwise use local paths
if os.path.exists('/app/data'):
    # Railway persistent storage
    DATA_DIR = Path('/app/data')
    STATIC_DIR = Path('/app/static')
else:
    # Local development
    DATA_DIR = Path('data')
    STATIC_DIR = Path('static')

RECOMMENDATION_DIR = STATIC_DIR / "recommendations"
INDEX_PATH = DATA_DIR / "image_index.faiss"
MAPPING_PATH = DATA_DIR / "image_paths.json"

def index_images():
    print("Starting image indexing...")
    
    # Get list of image files recursively
    image_files = []
    for root, dirs, files in os.walk(RECOMMENDATION_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                # Store relative path from RECOMMENDATION_DIR
                full_path = Path(root) / file
                rel_path = full_path.relative_to(RECOMMENDATION_DIR)
                image_files.append(str(rel_path))
    
    print(f"Found {len(image_files)} images to index")
    
    if not image_files:
        print("No images found to index!")
        return
    
    embeddings = []
    mapping = []
    
    # Process images in smaller batches to avoid memory issues
    batch_size = 5
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        for file in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
            path = RECOMMENDATION_DIR / file
            try:
                print(f"Processing {file}...")
                img = Image.open(path).convert("RGB")
                emb = get_image_embedding(img)
                embeddings.append(emb)
                mapping.append(file)
                print(f"✅ Successfully processed {file}")
                
                # Clean up memory
                del img
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error indexing {file}: {e}")
                continue
    
    if embeddings:
        print(f"Creating FAISS index with {len(embeddings)} images...")
        features_array = np.vstack(embeddings)
        index = faiss.IndexFlatL2(features_array.shape[1])
        index.add(features_array.astype("float32"))

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))

        with open(MAPPING_PATH, "w") as f:
            json.dump(mapping, f)
        print(f"✅ Indexing complete! Indexed {len(mapping)} images.")
    else:
        print("❌ No images were successfully indexed!")

if __name__ == "__main__":
    try:
        index_images()
    except Exception as e:
        print(f"❌ Fatal error during indexing: {e}")
        import traceback
        traceback.print_exc()
