"""Scraper for 1stclass-vintage.com using Shopify JSON API.

Fetches all products from the /collections/shop-all/products.json endpoint
with pagination, then fetches full product details from /products/{handle}.json.
"""

import json
from typing import Any

import httpx
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    COLLECTION_URL,
    MAX_RETRIES,
    PRODUCTS_PER_PAGE,
    REQUEST_TIMEOUT,
    CATEGORY_MAP,
    SOURCE,
    BRAND,
    SECOND_HAND,
)


def _build_url(path: str) -> str:
    """Build absolute URL, handling protocol-relative URLs (//...)."""
    if path.startswith("//"):
        return f"https:{path}"
    if path.startswith("/"):
        return f"{BASE_URL}{path}"
    return path


def _fetch_json(url: str, client: httpx.Client) -> dict[str, Any] | None:
    """Fetch a JSON endpoint with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  [ERROR] Failed to fetch {url}: {e}")
                return None
            import time

            time.sleep(1)


def fetch_product_urls(client: httpx.Client) -> list[str]:
    """Fetch all product handles from the collection with pagination.

    Iterates through paginated pages of /collections/shop-all/products.json
    until an empty page is encountered.
    """
    handles: list[str] = []
    page = 1

    while True:
        url = f"{COLLECTION_URL}?page={page}&limit={PRODUCTS_PER_PAGE}"
        data = _fetch_json(url, client)
        if not data:
            break

        products = data.get("products", [])
        if not products:
            break  # No more products

        for p in products:
            handle = p.get("handle")
            if handle:
                handles.append(handle)

        print(f"  Page {page}: found {len(products)} products (total: {len(handles)})")
        # If less than the requested limit, this was the last page
        if len(products) < PRODUCTS_PER_PAGE:
            break
        page += 1

    return handles


def fetch_product_detail(handle: str, client: httpx.Client) -> dict[str, Any] | None:
    """Fetch full product detail from /products/{handle}.json."""
    url = f"{BASE_URL}/products/{handle}.json"
    data = _fetch_json(url, client)
    if data:
        return data.get("product")
    return None


def infer_category(title: str) -> str | None:
    """Infer product category from the title using keyword matching."""
    title_lower = title.lower()
    matches = []
    for keyword, category in CATEGORY_MAP.items():
        if keyword in title_lower:
            matches.append(category)
    return ", ".join(sorted(set(matches))) if matches else None


def extract_size(title: str) -> str | None:
    """Extract size from the product title.

    Titles follow patterns like:
      "Vintage Grey Hoodie - L" -> "L"
      "Vintage Camo Cargo Shorts - 34" -> "34"
      "Vintage White Long Sleeve - M" -> "M"
    """
    if " - " in title:
        parts = title.split(" - ")
        size_candidate = parts[-1].strip()
        if size_candidate and len(size_candidate) <= 10:
            return size_candidate
    return None


def infer_gender(title: str) -> str | None:
    """Infer gender from the title."""
    title_lower = title.lower()
    # Most vintage items are unisex
    if any(w in title_lower for w in ["women", "womens", "female", "girl"]):
        return "women"
    if any(w in title_lower for w in ["men", "mens", "male", "boy"]):
        return "men"
    return None


def build_product_record(product: dict[str, Any]) -> dict[str, Any] | None:
    """Build a Supabase-ready record from a Shopify product JSON.

    Returns None if the product has no images (can't embed).
    """
    title = product.get("title", "")
    images = product.get("images", [])
    if not images:
        return None

    main_image = images[0].get("src", "")
    main_image = _build_url(main_image)

    # Additional images (skip first since it's the primary)
    additional_images_list = []
    for img in images[1:]:
        src = img.get("src", "")
        if src:
            additional_images_list.append(_build_url(src))
    additional_images = " , ".join(additional_images_list) if additional_images_list else None

    # Variant info
    variants = product.get("variants", [{}])
    variant = variants[0] if variants else {}
    price = variant.get("price", "0.00")
    compare_at_price = variant.get("compare_at_price")

    # Determine sale vs original price
    sale_price = None
    original_price = price
    if compare_at_price and float(compare_at_price) > float(price):
        # There's a sale
        original_price = compare_at_price
        sale_price = price

    # Format price with currency (default USD as seen in OG tags)
    price_str = f"{original_price}USD"
    sale_str = f"{sale_price}USD" if sale_price else None

    # Description
    body_html = product.get("body_html", "")
    description = BeautifulSoup(body_html, "html.parser").get_text(strip=True) if body_html else ""

    # Category inference
    category = infer_category(title)

    # Size extraction
    size = extract_size(title)

    # Gender
    gender = infer_gender(title)

    # Tags
    tags_raw = product.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None

    # Build metadata
    metadata = {
        "title": title,
        "description": description,
        "price": price_str,
        "sale": sale_str,
        "size": size,
        "category": category,
        "gender": gender,
        "tags": tags,
        "images_count": len(images),
        "variants": [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "price": v.get("price"),
                "compare_at_price": v.get("compare_at_price"),
                "sku": v.get("sku"),
                "available": v.get("available"),
            }
            for v in variants
        ],
        "product_type": product.get("product_type"),
        "vendor": product.get("vendor"),
        "shopify_created_at": product.get("created_at"),
        "shopify_updated_at": product.get("updated_at"),
    }

    record = {
        "id": str(product["id"]),
        "source": SOURCE,
        "product_url": f"{BASE_URL}/products/{product['handle']}",
        "image_url": main_image,
        "brand": BRAND,
        "title": title,
        "description": description or None,
        "category": category,
        "gender": gender,
        "size": size,
        "price": price_str,
        "sale": sale_str,
        "additional_images": additional_images,
        "second_hand": SECOND_HAND,
        "tags": tags,
        "metadata": json.dumps(metadata),
        "other": None,
        "affiliate_url": None,
        "country": None,
        "compressed_image_url": None,
    }

    return record
