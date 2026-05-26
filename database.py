"""Supabase database operations for the scraper.

Handles insertion of product records with vector embeddings
into the products table.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, DB_BATCH_SIZE

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Manages Supabase database operations for product records."""

    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")

    def upsert_products(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Insert or update product records in batches.

        Uses upsert with the unique constraint on (source, product_url).
        Returns summary stats.
        """
        if not records:
            return {"inserted": 0, "updated": 0, "errors": 0}

        # Add created_at timestamp
        now = datetime.now(timezone.utc).isoformat()
        for record in records:
            record["created_at"] = now

        total = len(records)
        inserted = 0
        errors = 0

        for batch_start in range(0, total, DB_BATCH_SIZE):
            batch = records[batch_start : batch_start + DB_BATCH_SIZE]

            # Prepare batch for upsert - convert lists/vectors to proper format
            clean_batch = []
            for record in batch:
                clean = {}
                for key, value in record.items():
                    if key == "search_tsv" or key == "search_vector" or key == "title_tsv" or key == "brand_tsv" or key == "description_tsv":
                        # These are DB-generated, skip them
                        continue
                    if isinstance(value, list):
                        # Convert list to JSON string or keep as-is for vector type
                        # The supabase client will handle vector columns as lists
                        clean[key] = value
                    elif isinstance(value, dict):
                        clean[key] = json.dumps(value)
                    elif value is None:
                        clean[key] = None
                    else:
                        clean[key] = value
                clean_batch.append(clean)

            try:
                # Use upsert with the unique constraint
                self.client.table(SUPABASE_TABLE).upsert(
                    clean_batch,
                    on_conflict="source,product_url",
                    ignore_duplicates=False,
                ).execute()
                inserted += len(clean_batch)
                logger.debug(f"  Inserted batch of {len(clean_batch)} products")
            except Exception as e:
                logger.error(f"  Batch insert error: {e}")
                # Try inserting one by one to isolate failures
                for rec in clean_batch:
                    try:
                        self.client.table(SUPABASE_TABLE).upsert(
                            [rec],
                            on_conflict="source,product_url",
                            ignore_duplicates=False,
                        ).execute()
                        inserted += 1
                    except Exception as e2:
                        logger.error(f"  Failed to insert product {rec.get('id')}: {e2}")
                        errors += 1

        return {"inserted": inserted, "errors": errors}

    def get_existing_product_urls(self) -> set[str]:
        """Get all existing product_urls for this source to skip re-processing."""
        try:
            resp = (
                self.client.table(SUPABASE_TABLE)
                .select("product_url")
                .eq("source", "scraper-1stclass")
                .execute()
            )
            return {row["product_url"] for row in resp.data}
        except Exception as e:
            logger.warning(f"Failed to fetch existing products: {e}")
            return set()
