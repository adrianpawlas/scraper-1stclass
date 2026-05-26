"""Configuration for the 1stclass-vintage scraper."""

import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SUPABASE_TABLE = "products"

# Scraper settings
BASE_URL = "https://1stclass-vintage.com"
COLLECTION_URL = f"{BASE_URL}/collections/shop-all/products.json"
PRODUCTS_PER_PAGE = 250
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30.0

# Embeddings
EMBEDDING_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
BATCH_SIZE_EMBEDDINGS = 8  # How many images/texts to embed at once

# Database batch insert size
DB_BATCH_SIZE = 50

# Source / brand
SOURCE = "scraper-1stclass"
BRAND = "1stclass"
SECOND_HAND = True

# Category inference mapping
CATEGORY_MAP = {
    "hoodie": "Sweaters, Hoodies",
    "crewneck": "Sweaters",
    "sweater": "Sweaters",
    "fleece": "Sweaters",
    "quarter zip": "Tops",
    "long sleeve": "Tops",
    "tee": "T-Shirts",
    "t-shirt": "T-Shirts",
    "tank": "Tanks, Stringers",
    "stringer": "Tanks, Stringers",
    "shorts": "Bottoms, Shorts",
    "pant": "Bottoms, Pants",
    "jean": "Bottoms, Jeans",
    "cargo": "Bottoms",
    "jacket": "Jackets, Outerwear",
    "zip up": "Jackets, Outerwear",
    "vest": "Vests, Outerwear",
    "button up": "Shirts, Button-Ups",
    "button-down": "Shirts, Button-Ups",
    "shirt": "Shirts",
    "polo": "Polo Shirts",
    "jersey": "Jerseys",
    "hat": "Accessories, Hats",
    "cap": "Accessories, Hats",
    "beanie": "Accessories, Beanies",
    "bag": "Accessories, Bags",
    "belt": "Accessories, Belts",
    "sock": "Accessories, Socks",
    "shoe": "Footwear, Shoes",
    "sneaker": "Footwear, Sneakers",
    "boot": "Footwear, Boots",
}
