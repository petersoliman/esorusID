"""Client for the seo-catalog-generator catalog API.

esorusID owns images and embeddings only. Product facts — supplier, name, price,
and which storefront product a match corresponds to — live in the catalog and are
looked up by GPID.

The GPID is the *only* key used here. Nothing in this module derives identity
from a filename: the on-disk layout mixes two id spaces
(`sp/<marketplace>/<domain>/<catalog product id>/<scraper id>-<n>.jpg`), and
parsing it is exactly how this service ended up holding unresolvable ids.

Every lookup fails soft. If the catalog is unreachable, search results still
render — just without catalog detail.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

CATALOG_API_URL = os.environ.get("CATALOG_API_URL", "").rstrip("/")
CATALOG_API_KEY = os.environ.get("CATALOG_API_KEY", "")
CATALOG_MARKETPLACE = os.environ.get("CATALOG_MARKETPLACE", "esorus")
CATALOG_API_TIMEOUT = float(os.environ.get("CATALOG_API_TIMEOUT", "3"))
# The manifest is a bulk call — 66k images is 34 pages of 2000 — so it needs far
# longer than the per-search resolve timeout, which is deliberately short so a
# slow catalog never stalls a user's search.
CATALOG_MANIFEST_TIMEOUT = float(os.environ.get("CATALOG_MANIFEST_TIMEOUT", "120"))
CATALOG_CACHE_TTL = float(os.environ.get("CATALOG_CACHE_TTL", "3600"))

# Identify ourselves. urllib's default ("Python-urllib/3.x") is blocked outright
# by Cloudflare's bot rules — every catalog call came back 403 in production
# while the same URL served fine from curl.
USER_AGENT = os.environ.get("CATALOG_USER_AGENT", "esorusID/1.0 (+catalog-client)")

# Matches MAX_GPIDS in CatalogResolveController.
MAX_GPIDS_PER_REQUEST = 500

# gpid -> (payload_or_None, expires_at). Misses are cached too, so an unknown
# GPID does not trigger a request on every search.
_cache: Dict[str, Tuple[Optional[dict], float]] = {}


def _request(path: str, params: dict, timeout: float = None) -> Optional[dict]:
    """GET a catalog endpoint. Returns the decoded `data` block, or None."""
    if not CATALOG_API_URL:
        return None

    url = f"{CATALOG_API_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if CATALOG_API_KEY:
        req.add_header("X-API-Key", CATALOG_API_KEY)

    try:
        with urllib.request.urlopen(req, timeout=timeout or CATALOG_API_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        logging.warning(f"Catalog request failed ({path}): {e}")
        return None

    if not payload.get("success"):
        logging.warning(f"Catalog request rejected ({path}): {payload.get('error')}")
        return None
    return payload.get("data") or {}


def resolve(gpids) -> Dict[str, dict]:
    """Map GPIDs to catalog detail, using a TTL cache in front of the API.

    Unknown GPIDs are omitted. Returns {} when CATALOG_API_URL is unset, so the
    feature stays dormant until configured.
    """
    wanted = {str(g) for g in gpids if g}
    if not wanted or not CATALOG_API_URL:
        return {}

    now = time.time()
    resolved: Dict[str, dict] = {}
    missing: List[str] = []
    for gpid in wanted:
        cached = _cache.get(gpid)
        if cached and cached[1] > now:
            if cached[0] is not None:
                resolved[gpid] = cached[0]
        else:
            missing.append(gpid)

    for start in range(0, len(missing), MAX_GPIDS_PER_REQUEST):
        batch = missing[start:start + MAX_GPIDS_PER_REQUEST]
        data = _request(
            "/api/catalog/resolve",
            {"marketplace": CATALOG_MARKETPLACE, "gpids": ",".join(batch)},
        )
        if data is None:
            # Request failed outright — do not cache the batch as unknown.
            continue
        expires_at = time.time() + CATALOG_CACHE_TTL
        for gpid in batch:
            entry = data.get(gpid)
            _cache[gpid] = (entry, expires_at)
            if entry is not None:
                resolved[gpid] = entry

    return resolved


def image_manifest(limit: int = 2000) -> List[dict]:
    """Every catalog image paired with its product's GPID.

    Used by index_images.py so the index records identity explicitly instead of
    inferring it from a filename. Returns [] when unreachable or unconfigured.
    """
    images: List[dict] = []
    offset = 0
    while True:
        data = _request(
            "/api/catalog/images",
            {"marketplace": CATALOG_MARKETPLACE, "limit": limit, "offset": offset},
            timeout=CATALOG_MANIFEST_TIMEOUT,
        )
        if data is not None and offset == 0:
            print(f"Fetching manifest from {CATALOG_API_URL} ...")
        if not data:
            break
        page = data.get("images") or []
        images.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return images
