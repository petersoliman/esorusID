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

ALLOWED_EXTS = (".jpg", ".png", ".jpeg", ".bmp", ".webp")


def _collect_image_files():
    """Recursively collect relative paths of all images in RECOMMENDATION_DIR."""
    image_files = []
    for root, _dirs, files in os.walk(RECOMMENDATION_DIR):
        for file in files:
            if file.lower().endswith(ALLOWED_EXTS):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(RECOMMENDATION_DIR)
                image_files.append(str(rel_path))
    return image_files


def _embed_files(files):
    """Embed a list of image files in batches. Returns (embeddings, mapping) — lists aligned by index."""
    embeddings = []
    mapping = []
    batch_size = 5

    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}")

        for file in tqdm(batch, desc=f"Batch {i // batch_size + 1}"):
            path = RECOMMENDATION_DIR / file
            try:
                img = Image.open(path).convert("RGB")
                emb = get_image_embedding(img)
                embeddings.append(emb)
                mapping.append(file)
                del img
                gc.collect()
            except Exception as e:
                print(f"❌ Error indexing {file}: {e}")
                continue

    return embeddings, mapping


def index_images():
    print("Starting image indexing...")

    # Decide mode: incremental if both index files already exist.
    incremental = INDEX_PATH.exists() and MAPPING_PATH.exists()

    existing_mapping = []
    existing_index = None
    if incremental:
        try:
            existing_index = faiss.read_index(str(INDEX_PATH))
            with open(MAPPING_PATH, "r") as f:
                existing_mapping = json.load(f)
            print(f"Incremental run: loaded existing index with {len(existing_mapping)} images.")
        except Exception as e:
            print(f"⚠️ Failed to load existing index ({e}). Falling back to full rebuild.")
            incremental = False
            existing_mapping = []
            existing_index = None
    else:
        print("No existing index found — performing full build.")

    # Discover current catalog.
    found_files = _collect_image_files()
    print(f"Found {len(found_files)} images in {RECOMMENDATION_DIR}.")

    if not found_files and not incremental:
        print("No images found to index!")
        return

    existing_set = set(existing_mapping)
    found_set = set(found_files)

    # Warn on entries in the mapping that no longer exist on disk.
    missing = [f for f in existing_mapping if f not in found_set]
    if missing:
        print(f"⚠️ {len(missing)} file(s) in the index are missing from {RECOMMENDATION_DIR}:")
        for f in missing[:20]:
            print(f"    - {f}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more.")
        print("   The index was NOT rebuilt to remove them. To clean up deletions, delete")
        print(f"   {INDEX_PATH} and {MAPPING_PATH}, then rerun this script.")

    if incremental:
        new_files = [f for f in found_files if f not in existing_set]
        if not new_files:
            print(f"✅ Index is already up to date. {len(existing_mapping)} images indexed.")
            return

        print(f"Embedding {len(new_files)} new image(s)...")
        embeddings, mapping_new = _embed_files(new_files)

        if not embeddings:
            print("❌ No new images were successfully embedded.")
            return

        features_array = np.vstack(embeddings).astype("float32")
        existing_index.add(features_array)

        combined_mapping = list(existing_mapping) + mapping_new

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(existing_index, str(INDEX_PATH))
        with open(MAPPING_PATH, "w") as f:
            json.dump(combined_mapping, f)

        print(f"✅ Incremental update complete. Added {len(mapping_new)} new images. Total indexed: {len(combined_mapping)}.")
    else:
        print(f"Embedding {len(found_files)} image(s)...")
        embeddings, mapping = _embed_files(found_files)

        if not embeddings:
            print("❌ No images were successfully indexed!")
            return

        features_array = np.vstack(embeddings).astype("float32")
        index = faiss.IndexFlatIP(features_array.shape[1])  # inner product = cosine sim on normalized vecs
        index.add(features_array)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))
        with open(MAPPING_PATH, "w") as f:
            json.dump(mapping, f)

        print(f"✅ Full index built. Total indexed: {len(mapping)}.")


if __name__ == "__main__":
    try:
        index_images()
    except Exception as e:
        print(f"❌ Fatal error during indexing: {e}")
        import traceback
        traceback.print_exc()
