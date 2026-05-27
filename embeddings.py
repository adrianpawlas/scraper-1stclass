"""Image and text embedding generation using google/siglip-base-patch16-384.

Produces 768-dimensional embeddings for both images and text.
Uses the SigLIP model from HuggingFace transformers.
"""

import io
import logging
from typing import Any

import httpx
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from config import BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM, BATCH_SIZE_EMBEDDINGS

logger = logging.getLogger(__name__)


class SiglipEmbedder:
    """Generates 768-dim image and text embeddings using SigLIP."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading SigLIP model on {device}...")
        self.model = SiglipModel.from_pretrained(EMBEDDING_MODEL).to(device)
        self.processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL)
        self.model.eval()
        logger.info("SigLIP model loaded successfully")

    def _download_image(self, url: str, http_client: httpx.Client) -> Image.Image | None:
        """Download an image from a URL."""
        full_url = url
        if url.startswith("//"):
            full_url = f"https:{url}"
        elif url.startswith("/"):
            full_url = f"{BASE_URL}{url}"

        try:
            resp = http_client.get(full_url, timeout=30.0)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to download image {full_url}: {e}")
            return None

    @torch.no_grad()
    def embed_images(self, image_urls: list[str], http_client: httpx.Client) -> list[list[float] | None]:
        """Generate embeddings for a list of image URLs.

        Returns a list of 768-dim embeddings (or None for failed downloads).
        """
        embeddings: list[list[float] | None] = []
        processed_images: list[Image.Image] = []
        valid_indices: list[int] = []

        for idx, url in enumerate(image_urls):
            img = self._download_image(url, http_client)
            if img is not None:
                processed_images.append(img)
                valid_indices.append(idx)
            else:
                embeddings.append(None)

        if not processed_images:
            return [None] * len(image_urls)

        # Process in batches
        for batch_start in range(0, len(processed_images), BATCH_SIZE_EMBEDDINGS):
            batch_imgs = processed_images[batch_start : batch_start + BATCH_SIZE_EMBEDDINGS]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            image_embeds = outputs.pooler_output / outputs.pooler_output.norm(p=2, dim=-1, keepdim=True)

            for emb in image_embeds.cpu().tolist():
                embeddings.append(emb)

        # Reconstruct in original order
        result: list[list[float] | None] = [None] * len(image_urls)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb

        return result

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate text embeddings for a list of text strings.

        Returns 768-dim embeddings for each text.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), BATCH_SIZE_EMBEDDINGS):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE_EMBEDDINGS]
            inputs = self.processor(
                text=batch_texts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model.get_text_features(**inputs)
            text_embeds = outputs.pooler_output / outputs.pooler_output.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.extend(text_embeds.cpu().tolist())

        return all_embeddings

    def build_info_text(self, record: dict[str, Any]) -> str:
        """Build a comprehensive text string from a product record for text embedding."""
        parts = []
        if record.get("title"):
            parts.append(f"Title: {record['title']}")
        if record.get("description"):
            parts.append(f"Description: {record['description']}")
        if record.get("category"):
            parts.append(f"Category: {record['category']}")
        if record.get("price"):
            parts.append(f"Price: {record['price']}")
        if record.get("sale"):
            parts.append(f"Sale: {record['sale']}")
        if record.get("size"):
            parts.append(f"Size: {record['size']}")
        if record.get("gender"):
            parts.append(f"Gender: {record['gender']}")
        if record.get("brand"):
            parts.append(f"Brand: {record['brand']}")
        if record.get("tags"):
            parts.append(f"Tags: {', '.join(record['tags'])}")

        metadata_raw = record.get("metadata")
        if isinstance(metadata_raw, str):
            import json

            try:
                meta = json.loads(metadata_raw)
                if meta.get("vendor"):
                    parts.append(f"Vendor: {meta['vendor']}")
                if meta.get("product_type"):
                    parts.append(f"Type: {meta['product_type']}")
            except json.JSONDecodeError:
                pass

        return " | ".join(parts)

    def embed_product_record(self, record: dict[str, Any], http_client: httpx.Client) -> dict[str, Any]:
        """Generate image and text embeddings for a product record and return the updated record.

        Also builds the info_embedding from all available product info.
        """
        # Image embedding
        image_url = record.get("image_url")
        if image_url:
            emb_list = self.embed_images([image_url], http_client)
            record["image_embedding"] = emb_list[0]
        else:
            record["image_embedding"] = None

        # Text embedding
        info_text = self.build_info_text(record)
        if info_text:
            text_emb_list = self.embed_texts([info_text])
            record["info_embedding"] = text_emb_list[0] if text_emb_list else None
        else:
            record["info_embedding"] = None

        return record
