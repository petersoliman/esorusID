"""Build the FAISS visual-search index from the catalog image manifest.

Images are NOT copied here. The catalog already stores every product image on
disk, and duplicating them cost 25 GB on production for 66,467 images — the copy
existed only because this service originally kept its own small image folder.

Instead each index entry records:

    gpid  — the identity of the product the image belongs to
    path  — the catalog-relative path, e.g. sp/esorus/<domain>/<id>/<file>.jpg
    url   — the public URL the results page renders

Embedding reads the file in place when CATALOG_PUBLIC_DIR points at the catalog's
public/ directory (same machine), and otherwise downloads it to a temp file.

Usage:
    python index_images.py                       # everything in the manifest
    python index_images.py --supplier example.com  # one supplier
    python index_images.py --limit 200             # first N images
"""

import argparse
import gc
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import get_image_embedding

# Use Railway persistent storage if available, otherwise use local paths
if os.path.exists('/app/data'):
    DATA_DIR = Path('/app/data')
    STATIC_DIR = Path('/app/static')
else:
    DATA_DIR = Path('data')
    STATIC_DIR = Path('static')

RECOMMENDATION_DIR = STATIC_DIR / "recommendations"
INDEX_PATH = DATA_DIR / "image_index.faiss"
MAPPING_PATH = DATA_DIR / "image_paths.json"

ALLOWED_EXTS = (".jpg", ".png", ".jpeg", ".bmp", ".webp")

# Catalog's public/ directory. When set, images are read straight off the shared
# filesystem — no download, no copy.
CATALOG_PUBLIC_DIR = os.environ.get("CATALOG_PUBLIC_DIR", "").rstrip("/")
# Base URL the catalog serves images from; also what the results page links to.
CATALOG_ASSET_BASE = os.environ.get(
    "CATALOG_ASSET_BASE", os.environ.get("CATALOG_API_URL", "")
).rstrip("/")


def build_entries(supplier=None, limit=0):
    """Fetch the manifest and turn it into index entries.

    `supplier` filters on the domain segment of the catalog path, so a run can be
    scoped to one site for a quick end-to-end check instead of the whole catalog.
    """
    from catalog import image_manifest

    manifest = image_manifest()
    if not manifest:
        print("No catalog manifest available — nothing to index.")
        return []

    entries = []
    for item in manifest:
        gpid = item.get("gpid")
        rel = item.get("relative_path")
        if not gpid or not rel:
            continue
        if supplier and f"/{supplier}/" not in f"/{rel}":
            continue
        entries.append({
            "gpid": gpid,
            "path": rel,
            "url": f"{CATALOG_ASSET_BASE}/{rel}" if CATALOG_ASSET_BASE else rel,
        })
        if limit and len(entries) >= limit:
            break

    scope = f" for supplier {supplier}" if supplier else ""
    print(f"Manifest: {len(manifest)} image(s); {len(entries)} selected{scope}.")
    return entries


def _open_image(entry):
    """Return a PIL image for an entry, reading locally when possible.

    Falls back to downloading into a temp file that is deleted immediately after
    the embedding is taken, so nothing accumulates on disk.
    """
    rel = entry["path"]

    if CATALOG_PUBLIC_DIR:
        local = Path(CATALOG_PUBLIC_DIR) / rel
        if local.is_file():
            return Image.open(local).convert("RGB"), None

    if not entry["url"]:
        raise FileNotFoundError(f"{rel} not found locally and no CATALOG_ASSET_BASE set")

    suffix = Path(rel).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    # Same reason as in catalog.py: urllib's default User-Agent is 403'd by
    # Cloudflare, so image downloads must identify themselves too.
    from catalog import USER_AGENT
    req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp.name, "wb") as out:
        out.write(resp.read())
    return Image.open(tmp.name).convert("RGB"), tmp.name


def _embed_entries(entries):
    """Embed entries in batches. Returns (embeddings, mapping) aligned by index."""
    embeddings = []
    mapping = []
    batch_size = 5

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(entries) + batch_size - 1) // batch_size}")

        for entry in tqdm(batch, desc=f"Batch {i // batch_size + 1}"):
            tmp_name = None
            try:
                img, tmp_name = _open_image(entry)
                embeddings.append(get_image_embedding(img))
                mapping.append(entry)
                del img
                gc.collect()
            except Exception as e:
                print(f"❌ Error indexing {entry['path']}: {e}")
                continue
            finally:
                if tmp_name:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass

    return embeddings, mapping


def index_images(supplier=None, limit=0):
    """Rebuild the index from the catalog manifest.

    Always a full rebuild. The mapping is positional — FAISS returns index i and
    the caller reads mapping[i] — so patching it in place risks mis-attributing
    every hit after a changed row. A rebuild is the only safe update.
    """
    print("Starting image indexing...")

    entries = build_entries(supplier=supplier, limit=limit)
    if not entries:
        print("Nothing to index.")
        return

    print(f"Embedding {len(entries)} image(s)...")
    embeddings, mapping = _embed_entries(entries)

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

    print(f"✅ Index built. Total indexed: {len(mapping)}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the visual-search index from the catalog.")
    ap.add_argument("--supplier", default=None,
                    help="Index only one supplier domain, e.g. bespokefurnitureeg.com")
    ap.add_argument("--limit", type=int, default=0,
                    help="Index at most N images (0 = no limit)")
    args = ap.parse_args()

    try:
        index_images(supplier=args.supplier, limit=args.limit)
    except Exception as e:
        print(f"❌ Fatal error during indexing: {e}")
        import traceback
        traceback.print_exc()
