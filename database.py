"""Supabase database operations for the scraper.

Handles smart upsert of product records with change detection,
batch retry logic, and stale product cleanup.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from supabase import create_client, Client

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SOURCE,
    DB_BATCH_SIZE,
    MAX_BATCH_RETRIES,
    FAILED_LOG_FILE,
    STALE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _normalize(val: Any) -> str:
    """Normalize a value for comparison."""
    if val is None:
        return ""
    if isinstance(val, list):
        return json.dumps(sorted(str(v) for v in val), sort_keys=True)
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return str(val)


def product_changed(scraped: dict[str, Any], existing: dict[str, Any]) -> bool:
    """Compare scraped data against an existing database record.

    Returns True if any meaningful field has changed.
    """
    fields = [
        "title", "description", "price", "sale", "image_url",
        "additional_images", "category", "size", "gender",
    ]
    for field in fields:
        if _normalize(scraped.get(field)) != _normalize(existing.get(field)):
            return True

    # Compare tags (ordered lists)
    scraped_tags = scraped.get("tags") or []
    existing_tags = existing.get("tags") or []
    if sorted(scraped_tags) != sorted(existing_tags):
        return True

    # Compare availability
    scraped_avail = scraped.get("is_available")
    existing_avail = existing.get("is_available")
    if scraped_avail is not None and scraped_avail != existing_avail:
        return True

    return False


class SupabaseDB:
    """Manages Supabase database operations for product records."""

    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def get_existing_products(self) -> dict[str, dict[str, Any]]:
        """Fetch ALL existing products for this source, keyed by product_url."""
        try:
            resp = (
                self.client.table(SUPABASE_TABLE)
                .select("*")
                .eq("source", SOURCE)
                .execute()
            )
            products: dict[str, dict[str, Any]] = {}
            for row in resp.data:
                url = row.get("product_url", "")
                if url:
                    products[url] = row
            logger.info(f"  Fetched {len(products)} existing products from database")
            return products
        except Exception as e:
            logger.warning(f"  Failed to fetch existing products: {e}")
            return {}

    def get_existing_product_urls(self) -> set[str]:
        """Get all existing product_urls for this source (lightweight)."""
        try:
            resp = (
                self.client.table(SUPABASE_TABLE)
                .select("product_url")
                .eq("source", SOURCE)
                .execute()
            )
            return {row["product_url"] for row in resp.data}
        except Exception as e:
            logger.warning(f"  Failed to fetch existing product URLs: {e}")
            return set()

    # ------------------------------------------------------------------
    # Upsert with retry
    # ------------------------------------------------------------------

    def upsert_products(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Upsert product records in batches with retry logic.

        Logs persistently-failed products to FAILED_LOG_FILE.

        Returns:
            dict with keys 'errors' and 'failed_count'.
        """
        if not records:
            return {"errors": 0, "failed_count": 0}

        now = datetime.now(timezone.utc).isoformat()
        for record in records:
            record["created_at"] = now
            record["updated_at"] = now

        total = len(records)
        errors = 0
        failed_products: list[dict] = []

        for batch_start in range(0, total, DB_BATCH_SIZE):
            batch = records[batch_start: batch_start + DB_BATCH_SIZE]

            clean_batch = []
            last_error: str = ""
            for record in batch:
                clean = {}
                for key, value in record.items():
                    if key in ("search_tsv", "search_vector", "title_tsv", "brand_tsv", "description_tsv"):
                        continue
                    if isinstance(value, (list, dict)):
                        clean[key] = value
                    elif value is None:
                        clean[key] = None
                    else:
                        clean[key] = value
                clean_batch.append(clean)

            # Retry loop
            success = False
            for attempt in range(1, MAX_BATCH_RETRIES + 1):
                try:
                    self.client.table(SUPABASE_TABLE).upsert(
                        clean_batch,
                        on_conflict="source,product_url",
                        ignore_duplicates=False,
                    ).execute()
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"  Batch upsert attempt {attempt}/{MAX_BATCH_RETRIES} failed: {last_error}"
                    )
                    if attempt < MAX_BATCH_RETRIES:
                        time.sleep(1)

            if not success:
                errors += 1
                batch_ids = [r.get("product_url", r.get("id", "unknown")) for r in clean_batch]
                failed_entry = {
                    "timestamp": now,
                    "error": last_error,
                    "product_urls": batch_ids,
                }
                failed_products.append(failed_entry)
                logger.error(f"  Batch of {len(clean_batch)} products failed after {MAX_BATCH_RETRIES} retries")

        # Log failed products to file
        if failed_products:
            try:
                with open(FAILED_LOG_FILE, "a") as f:
                    for entry in failed_products:
                        f.write(json.dumps(entry) + "\n")
                logger.warning(f"  Logged {len(failed_products)} failed batch(es) to {FAILED_LOG_FILE}")
            except OSError as e:
                logger.error(f"  Could not write to {FAILED_LOG_FILE}: {e}")

        return {"errors": errors, "failed_count": len(failed_products)}

    # ------------------------------------------------------------------
    # Stale product tracking
    # ------------------------------------------------------------------

    def update_stale_tracking(self, seen_urls: set[str]) -> dict[str, Any]:
        """Update stale_count for all products from this source.

        - Products that were **seen** in this run get stale_count = 0.
        - Products that were **not seen** get stale_count incremented by 1.
        - Products with stale_count >= STALE_THRESHOLD are deleted.

        Returns:
            dict with key 'deleted' (count of removed products).
        """
        try:
            resp = (
                self.client.table(SUPABASE_TABLE)
                .select("id, product_url, stale_count")
                .eq("source", SOURCE)
                .execute()
            )
        except Exception as e:
            err_str = str(e)
            if "stale_count" in err_str and "does not exist" in err_str:
                logger.warning(
                    "  Stale tracking requires a 'stale_count' column in the "
                    "products table. Run: ALTER TABLE products ADD COLUMN "
                    "stale_count INTEGER DEFAULT 0;"
                )
            else:
                logger.warning(f"  Could not query products for stale tracking: {err_str}")
            return {"deleted": 0}

        rows = resp.data
        if not rows:
            return {"deleted": 0}

        # Build updates: each row gets a new stale_count
        updates: list[dict] = []
        to_delete_ids: list[str] = []

        for row in rows:
            pid = row["id"]
            url = row.get("product_url", "")
            current_stale = row.get("stale_count", 0) or 0

            if url in seen_urls:
                # Seen this run – reset
                updates.append({"id": pid, "stale_count": 0})
            else:
                new_count = current_stale + 1
                if new_count >= STALE_THRESHOLD:
                    # Stale for too long – mark for deletion
                    to_delete_ids.append(pid)
                else:
                    updates.append({"id": pid, "stale_count": new_count})

        # Apply stale_count updates in batches
        for batch_start in range(0, len(updates), DB_BATCH_SIZE):
            batch = updates[batch_start: batch_start + DB_BATCH_SIZE]
            try:
                self.client.table(SUPABASE_TABLE).upsert(
                    batch,
                    on_conflict="id",
                    ignore_duplicates=False,
                ).execute()
            except Exception as e:
                logger.warning(f"  Failed to update stale counts for a batch: {e}")

        # Delete permanently stale products
        deleted = 0
        for batch_start in range(0, len(to_delete_ids), DB_BATCH_SIZE):
            batch = to_delete_ids[batch_start: batch_start + DB_BATCH_SIZE]
            try:
                self.client.table(SUPABASE_TABLE).delete().in_("id", batch).execute()
                deleted += len(batch)
            except Exception as e:
                logger.warning(f"  Failed to delete stale products batch: {e}")

        if deleted:
            logger.info(f"  Deleted {deleted} stale products (not seen for {STALE_THRESHOLD}+ runs)")

        return {"deleted": deleted}
