import os
import json
import faiss
import numpy as np
from utils import get_image_embedding
from PIL import Image
from tqdm import tqdm
import gc

RECOMMENDATION_DIR = "static/recommendations"
INDEX_PATH = "data/image_index.faiss"
MAPPING_PATH = "data/image_paths.json"

def index_images():
    print("Starting image indexing...")
    
    # Get list of image files
    image_files = [f for f in os.listdir(RECOMMENDATION_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
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
            path = os.path.join(RECOMMENDATION_DIR, file)
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

        os.makedirs("data", exist_ok=True)
        faiss.write_index(index, INDEX_PATH)

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
