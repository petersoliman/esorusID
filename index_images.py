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


def prune_orphan_files():
    """Delete staged images whose GPID the catalog no longer knows.

    Keeps static/recommendations in step with the catalog so a rebuild does not
    re-index products that have since been removed or re-minted.
    """
    live = _live_gpids()
    if not live:
        return 0

    removed = 0
    for f in RECOMMENDATION_DIR.glob("*"):
        if not f.is_file() or not f.name.lower().endswith(ALLOWED_EXTS):
            continue
        gpid = _gpid_from_filename(f.name)
        if gpid is not None and gpid not in live:
            f.unlink()
            removed += 1
    if removed:
        print(f"Pruned {removed} staged image(s) whose GPID is no longer in the catalog.")
    return removed


def sync_from_catalog():
    """Pull the catalog image manifest and stage each image under its GPID.

    Files are named `{gpid}_{order}.{ext}` so the index never has to infer
    identity from a path. Source images are read from the shared filesystem when
    CATALOG_PUBLIC_DIR points at the catalog's public/ directory, otherwise
    fetched over HTTP from CATALOG_ASSET_BASE.

    Returns the number of images staged. A no-op when the catalog is unreachable
    or unconfigured, so a local-only index still works.
    """
    import shutil
    import urllib.request

    from catalog import image_manifest

    manifest = image_manifest()
    if not manifest:
        print("No catalog manifest available — leaving static/recommendations as-is.")
        return 0

    public_dir = os.environ.get("CATALOG_PUBLIC_DIR", "").rstrip("/")
    asset_base = os.environ.get("CATALOG_ASSET_BASE", "").rstrip("/")
    RECOMMENDATION_DIR.mkdir(parents=True, exist_ok=True)

    staged = 0
    for entry in manifest:
        gpid = entry.get("gpid")
        rel = entry.get("relative_path")
        if not gpid or not rel:
            continue
        ext = Path(rel).suffix.lstrip(".").lower() or "jpg"
        dest = RECOMMENDATION_DIR / f"{gpid}_{entry.get('order', 0)}.{ext}"
        if dest.exists():
            staged += 1
            continue
        try:
            if public_dir:
                shutil.copy2(Path(public_dir) / rel, dest)
            elif asset_base:
                urllib.request.urlretrieve(f"{asset_base}/{rel}", dest)
            else:
                continue
            staged += 1
        except Exception as e:
            print(f"⚠️  Could not stage {rel}: {e}")
    print(f"Staged {staged} catalog image(s) into {RECOMMENDATION_DIR}.")
    return staged


def _live_gpids():
    """GPIDs the catalog currently knows, from the image manifest. Cached.

    An empty set means the catalog is unreachable — in that case nothing is
    treated as stale, so a network blip can never trigger a needless rebuild.
    """
    global _LIVE_GPIDS
    if _LIVE_GPIDS is None:
        from catalog import image_manifest
        _LIVE_GPIDS = {e["gpid"] for e in image_manifest() if e.get("gpid")}
        if not _LIVE_GPIDS:
            print("   (catalog unreachable — skipping stale-GPID check)")
    return _LIVE_GPIDS


_LIVE_GPIDS = None


def _gpid_from_filename(name):
    """GPID prefix of a staged filename, or None if it isn't GPID-named.

    This is a lookup of a name this indexer itself wrote — not identity
    inference. Files not written by sync_from_catalog() carry no GPID and are
    indexed with gpid=None so they remain searchable but unresolvable.
    """
    stem = Path(name).stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    # UUID4 canonical form: 8-4-4-4-12
    parts = prefix.split("-")
    if len(parts) == 5 and [len(p) for p in parts] == [8, 4, 4, 4, 12]:
        return prefix
    return None


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
                # Identity is recorded explicitly alongside the file. Consumers
                # read `gpid` from here and never parse the filename.
                mapping.append({"file": file, "gpid": _gpid_from_filename(file)})
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

    # Stage catalog images under their GPID, and drop any whose product the
    # catalog no longer knows, before discovering what's on disk.
    sync_from_catalog()
    prune_orphan_files()

    # Discover current catalog.
    found_files = _collect_image_files()
    print(f"Found {len(found_files)} images in {RECOMMENDATION_DIR}.")

    if not found_files and not incremental:
        print("No images found to index!")
        return

    # Mapping entries are {"file", "gpid"} objects. Pre-GPID indexes stored bare
    # filename strings, so tolerate both when reading an existing mapping.
    existing_files = [
        e["file"] if isinstance(e, dict) else e for e in existing_mapping
    ]
    existing_set = set(existing_files)
    found_set = set(found_files)

    # Entries in the mapping whose file is gone.
    #
    # These used to be tolerated with a warning, which was a correctness bug: the
    # mapping is positional — FAISS returns index i and the caller reads
    # mapping[i] — so a stale row silently shifts every later entry and a search
    # hit then reports a DIFFERENT product than the image it matched. Any drift
    # between disk and mapping therefore forces a full rebuild.
    missing = [f for f in existing_files if f not in found_set]
    if missing and incremental:
        print(f"⚠️ {len(missing)} indexed file(s) no longer exist on disk:")
        for f in missing[:10]:
            print(f"    - {f}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more.")
        print("   Falling back to a FULL rebuild — a positional mapping cannot be")
        print("   patched safely, and a stale row would mis-attribute search hits.")
        incremental = False
        existing_mapping = []
        existing_index = None
        existing_set = set()

    # Entries the catalog no longer recognises. After a GPID re-mint the old ids
    # resolve to nothing, so results render with no supplier and no link.
    if incremental and existing_mapping:
        stale_gpid = [
            e["file"] for e in existing_mapping
            if isinstance(e, dict) and e.get("gpid") and e["gpid"] not in _live_gpids()
        ]
        if stale_gpid:
            print(f"⚠️ {len(stale_gpid)} indexed image(s) carry a GPID the catalog no longer knows.")
            print("   Falling back to a FULL rebuild so they are dropped.")
            incremental = False
            existing_mapping = []
            existing_index = None
            existing_set = set()

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
