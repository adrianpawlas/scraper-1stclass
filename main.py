#!/usr/bin/env python3
"""
1stclass-vintage Scraper

Scrapes all products from 1stclass-vintage.com using the Shopify JSON API,
generates image and text embeddings using google/siglip-base-patch16-384,
and imports everything into Supabase.

Usage:
    python main.py                    # Run full scrape
    python main.py --resume           # Skip already-imported products
    python main.py --limit 50         # Only process N products (for testing)
    python main.py --skip-embeddings  # Skip embedding generation (for testing)
"""

import argparse
import json
import logging
import sys
import time

import httpx

from config import (
    BASE_URL,
    COLLECTION_URL,
    SOURCE,
    BRAND,
    PRODUCTS_PER_PAGE,
)
from scraper import fetch_product_urls, fetch_product_detail, build_product_record
from database import SupabaseDB
from embeddings import SiglipEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape 1stclass-vintage.com products to Supabase")
    parser.add_argument("--resume", action="store_true", help="Skip already-imported products")
    parser.add_argument("--limit", type=int, default=0, help="Max products to process (0 = all)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize database
    logger.info("Initializing Supabase connection...")
    db = SupabaseDB()

    # Get existing products if resuming
    existing_urls: set[str] = set()
    if args.resume:
        logger.info("Fetching existing product URLs to skip...")
        existing_urls = db.get_existing_product_urls()
        logger.info(f"  Found {len(existing_urls)} existing products in database")

    # Initialize HTTP client
    with httpx.Client(follow_redirects=True) as http_client:
        # Step 1: Fetch all product handles from collection pages
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching all product URLs from collection pages...")
        logger.info("=" * 60)
        handles = fetch_product_urls(http_client)
        logger.info(f"Found {len(handles)} total products in the collection")

        if args.limit > 0:
            handles = handles[: args.limit]
            logger.info(f"Limited to {len(handles)} products for testing")

        # Build product URLs
        product_urls = {f"{BASE_URL}/products/{h}" for h in handles}

        # Filter out already-processed if resuming
        if existing_urls:
            new_handles = [h for h in handles if f"{BASE_URL}/products/{h}" not in existing_urls]
            skipped = len(handles) - len(new_handles)
            logger.info(f"Skipping {skipped} already-imported products")
            handles = new_handles

        if not handles:
            logger.info("No new products to process. Exiting.")
            return

        logger.info(f"Will process {len(handles)} products")

        # Step 2: Fetch full product details
        logger.info("=" * 60)
        logger.info("STEP 2: Fetching full product details...")
        logger.info("=" * 60)

        products_raw = []
        failed = 0
        for i, handle in enumerate(handles, 1):
            if i % 50 == 0 or i == 1:
                logger.info(f"  Fetching product {i}/{len(handles)}...")
            product = fetch_product_detail(handle, http_client)
            if product:
                products_raw.append(product)
            else:
                failed += 1

        logger.info(f"Fetched {len(products_raw)} products ({failed} failed)")

        # Step 3: Build records
        logger.info("=" * 60)
        logger.info("STEP 3: Building product records...")
        logger.info("=" * 60)

        records = []
        for product in products_raw:
            record = build_product_record(product)
            if record:
                records.append(record)

        logger.info(f"Built {len(records)} product records")

        # Step 4: Generate embeddings (if not skipped)
        if not args.skip_embeddings:
            logger.info("=" * 60)
            logger.info("STEP 4: Generating image & text embeddings with SigLIP...")
            logger.info("=" * 60)

            embedder = SiglipEmbedder()

            for i, record in enumerate(records, 1):
                if i % 10 == 0 or i == 1:
                    logger.info(f"  Embedding product {i}/{len(records)}: {record.get('title', '')[:50]}...")
                try:
                    embedder.embed_product_record(record, http_client)
                except Exception as e:
                    logger.error(f"  Failed to embed product {record.get('id')}: {e}")
                    record["image_embedding"] = None
                    record["info_embedding"] = None
        else:
            logger.info("Skipping embeddings (--skip-embeddings)")
            for record in records:
                record["image_embedding"] = None
                record["info_embedding"] = None

        # Step 5: Insert into Supabase
        logger.info("=" * 60)
        logger.info("STEP 5: Inserting into Supabase...")
        logger.info("=" * 60)

        result = db.upsert_products(records)
        logger.info(f"Inserted: {result['inserted']}, Errors: {result['errors']}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total products in collection: {len(handles) + (len(existing_urls) if args.resume else 0)}")
    if args.resume:
        logger.info(f"Already in database: {len(existing_urls)}")
    logger.info(f"Newly processed: {len(records)}")
    logger.info(f"Newly inserted: {result['inserted']}")
    logger.info(f"Errors: {result['errors']}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
