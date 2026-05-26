#!/usr/bin/env python3
"""
1stclass-vintage Scraper

Scrapes all products from 1stclass-vintage.com using the Shopify JSON API,
generates image and text embeddings using google/siglip-base-patch16-384,
and imports everything into Supabase.

Features:
  - Smart upsert: only updates if data actually changed
  - Batch inserts (50/batch) with 3x retry & failure logging
  - Stale product cleanup after 2 consecutive unseen runs
  - Selective embedding: only regenerate when image URL changes
  - Staggered embedding generation (0.5s delay)
  - Detailed run summary

Usage:
    python main.py                    # Run full scrape
    python main.py --resume           # Smart diff against existing products
    python main.py --limit 50         # Only process N products (for testing)
    python main.py --skip-embeddings  # Skip embedding generation (for testing)
"""

import argparse
import logging
import time

import httpx

from config import (
    BASE_URL,
    COLLECTION_URL,
    SOURCE,
    BRAND,
    PRODUCTS_PER_PAGE,
    EMBEDDING_DELAY,
)
from scraper import fetch_product_urls, fetch_product_detail, build_product_record
from database import SupabaseDB, product_changed
from embeddings import SiglipEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape 1stclass-vintage.com products to Supabase"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip unchanged products (smart diff)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max products to process (0 = all)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------
    logger.info("Initializing Supabase connection...")
    db = SupabaseDB()

    existing_products: dict[str, dict] = {}
    if args.resume:
        logger.info("Fetching existing products for comparison...")
        existing_products = db.get_existing_products()

    with httpx.Client(follow_redirects=True) as http_client:
        # ------------------------------------------------------------------
        # STEP 1 – Fetch all product handles from collection pages
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching all product URLs from collection pages...")
        logger.info("=" * 60)
        handles = fetch_product_urls(http_client)
        logger.info(f"Found {len(handles)} total products in the collection")

        if args.limit > 0:
            handles = handles[:args.limit]
            logger.info(f"Limited to {len(handles)} products for testing")

        if not handles:
            logger.info("No products found. Exiting.")
            return

        # Track all product URLs from this run for stale cleanup.
        # Pre-populate from handles so transient fetch failures don't
        # cause false stale deletion.
        seen_urls: set[str] = {f"{BASE_URL}/products/{h}" for h in handles}

        # ------------------------------------------------------------------
        # STEP 2 – Fetch full product details
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STEP 2: Fetching full product details...")
        logger.info("=" * 60)

        products_raw = []
        failed_fetches = 0
        for i, handle in enumerate(handles, 1):
            if i % 50 == 0 or i == 1:
                logger.info(f"  Fetching product {i}/{len(handles)}...")
            product = fetch_product_detail(handle, http_client)
            if product:
                products_raw.append(product)
            else:
                failed_fetches += 1

        logger.info(f"Fetched {len(products_raw)} products ({failed_fetches} failed)")

        # ------------------------------------------------------------------
        # STEP 3 – Build product records
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STEP 3: Building product records...")
        logger.info("=" * 60)

        records = []
        for product in products_raw:
            record = build_product_record(product)
            if record:
                records.append(record)

        logger.info(f"Built {len(records)} product records")

        # ------------------------------------------------------------------
        # STEP 4 – Classify products (new / changed / unchanged)
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STEP 4: Classifying products...")
        logger.info("=" * 60)

        new_records: list[dict] = []
        changed_records: list[dict] = []
        unchanged_count = 0

        for record in records:
            url = record["product_url"]
            seen_urls.add(url)  # add to seen set now that we have a valid record

            if url not in existing_products:
                new_records.append(record)
            elif product_changed(record, existing_products[url]):
                changed_records.append(record)
            else:
                unchanged_count += 1

        # Pre-populate embeddings from existing data for changed products
        # (will be overwritten if regeneration is needed)
        for record in changed_records:
            existing = existing_products.get(record["product_url"])
            if existing:
                record["image_embedding"] = existing.get("image_embedding")
                record["info_embedding"] = existing.get("info_embedding")

        logger.info(f"  New:      {len(new_records)}")
        logger.info(f"  Changed:  {len(changed_records)}")
        logger.info(f"  Skipped:  {unchanged_count}")

        if not new_records and not changed_records:
            logger.info("No new or changed products to process.")

        # ------------------------------------------------------------------
        # STEP 5 – Generate embeddings (only when needed)
        # ------------------------------------------------------------------
        to_embed = new_records + changed_records
        if to_embed and not args.skip_embeddings:
            logger.info("=" * 60)
            logger.info("STEP 5: Generating embeddings (SigLIP)...")
            logger.info("=" * 60)

            embedder = SiglipEmbedder()

            for i, record in enumerate(to_embed, 1):
                existing = existing_products.get(record["product_url"])
                is_new = existing is None

                # Image embedding: regenerate only if new OR image_url changed
                needs_image = is_new or (
                    record.get("image_url") != existing.get("image_url")  # type: ignore[union-attr]
                )

                # Text embedding: regenerate if new OR any text field changed
                # (we already know the product is new/changed at this point)
                needs_text = is_new or True  # always regenerate text for changed products

                if needs_image:
                    emb_list = embedder.embed_images([record.get("image_url")], http_client)
                    record["image_embedding"] = emb_list[0] if emb_list else None
                    logger.debug(f"  Image embedding regenerated for {record.get('title', '')[:40]}")
                else:
                    logger.debug(f"  Image embedding reused for {record.get('title', '')[:40]}")

                if needs_text:
                    info_text = embedder.build_info_text(record)
                    if info_text:
                        text_emb_list = embedder.embed_texts([info_text])
                        record["info_embedding"] = text_emb_list[0] if text_emb_list else None
                    else:
                        record["info_embedding"] = None
                    logger.debug(f"  Text embedding regenerated for {record.get('title', '')[:40]}")
                else:
                    logger.debug(f"  Text embedding reused for {record.get('title', '')[:40]}")

                # Stagger: delay between products to avoid overwhelming the API
                if (needs_image or needs_text) and i < len(to_embed):
                    time.sleep(EMBEDDING_DELAY)

                if i % 10 == 0 or i == 1:
                    logger.info(f"  Embedding {i}/{len(to_embed)}...")

        elif to_embed and args.skip_embeddings:
            logger.info("Skipping embeddings (--skip-embeddings)")
            for record in to_embed:
                record["image_embedding"] = None
                record["info_embedding"] = None

        # ------------------------------------------------------------------
        # STEP 6 – Upsert to Supabase with retry
        # ------------------------------------------------------------------
        if new_records or changed_records:
            logger.info("=" * 60)
            logger.info("STEP 6: Upserting to Supabase...")
            logger.info("=" * 60)

            upsert_result = db.upsert_products(new_records + changed_records)

            if upsert_result["errors"]:
                logger.warning(f"  {upsert_result['errors']} batch(es) had errors")
            logger.info(f"  {len(new_records) + len(changed_records)} records upserted")
        else:
            logger.info("No records to upsert.")

        # ------------------------------------------------------------------
        # STEP 7 – Stale product cleanup
        # ------------------------------------------------------------------
        if args.resume:
            logger.info("=" * 60)
            logger.info("STEP 7: Cleaning up stale products...")
            logger.info("=" * 60)

            stale_result = db.update_stale_tracking(seen_urls)
        else:
            stale_result = {"deleted": 0}

        # ------------------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("RUN SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  New products added:    {len(new_records)}")
        logger.info(f"  Products updated:      {len(changed_records)}")
        logger.info(f"  Products unchanged:    {unchanged_count}")
        logger.info(f"  Stale products deleted:{stale_result['deleted']}")
        if failed_fetches:
            logger.info(f"  API fetch failures:    {failed_fetches}")
        logger.info("Done!")


if __name__ == "__main__":
    main()
