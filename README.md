# 1stclass-vintage Scraper

Scrapes all products from [1stclass-vintage.com](https://1stclass-vintage.com) using the Shopify JSON API, generates image and text embeddings using [google/siglip-base-patch16-384](https://huggingface.co/google/siglip-base-patch16-384), and imports everything into Supabase.

## Features

- Full product catalog scraping via Shopify's JSON API with pagination
- 768-dim image embeddings (SigLIP)
- 768-dim text embeddings from product title, description, category, etc.
- Automatic category inference from product titles
- Size, gender extraction from titles
- Sale price detection
- Upsert into Supabase with deduplication by `(source, product_url)`
- Resume support to skip already-imported products
- Configurable limits for testing

## Requirements

- Python 3.10+
- PyTorch (>=2.1.0)
- A Supabase project with a `products` table

## Setup

```bash
# Clone the repository
git clone https://github.com/adrianpawlas/scraper-1stclass.git
cd scraper-1stclass

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The scraper is pre-configured with default values in `config.py`. You can override any setting via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | (pre-set) | Your Supabase project URL |
| `SUPABASE_KEY` | (pre-set) | Your Supabase service role key |

For local development, create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
```

## Usage

```bash
# Full scrape (all products, with embeddings)
python main.py

# Resume: skip products already in the database
python main.py --resume

# Test mode: only process 10 products
python main.py --limit 10

# Test mode: skip embedding generation (faster)
python main.py --skip-embeddings

# Combine options
python main.py --resume --limit 50 --skip-embeddings
```

## Automation (GitHub Actions)

This repository includes a GitHub Actions workflow (`.github/workflows/scrape.yml`) that:

- **Runs automatically** every Monday at 4:00 AM UTC via a scheduled trigger
- **Can be triggered manually** from the GitHub Actions tab with optional parameters:
  - `resume` – Skip already-imported products
  - `limit` – Max products to process (0 = all)
  - `skip_embeddings` – Skip embedding generation

### Setting up GitHub Secrets

For the workflow to authenticate with Supabase, add these secrets in your GitHub repository:

1. Go to **Settings > Secrets and variables > Actions**
2. Add the following secrets:
   - `SUPABASE_URL` – Your Supabase project URL
   - `SUPABASE_KEY` – Your Supabase service role key

The workflow uses these secrets as environment variables, which override the defaults in `config.py`.

### Running Manually

To trigger the workflow manually:
1. Go to the **Actions** tab in your GitHub repository
2. Select the **Scrape 1stclass-vintage** workflow
3. Click **Run workflow**
4. Optionally set `resume`, `limit`, or `skip_embeddings` inputs
5. Click **Run workflow** to start

Scheduled runs (every Monday at 4:00 AM UTC) automatically use `--resume` mode to only process new products since the last run.

## Project Structure

```
.
├── main.py          # Entry point – coordinates the full pipeline
├── scraper.py       # Shopify API fetcher & product record builder
├── database.py      # Supabase client for upserting products
├── embeddings.py    # SigLIP image & text embedding generator
├── config.py        # Configuration & environment variables
├── requirements.txt # Python dependencies
└── .github/
    └── workflows/
        └── scrape.yml  # GitHub Actions automation
```

## Data Flow

1. **Fetch product handles** – Paginates through `/collections/shop-all/products.json`
2. **Fetch product details** – Gets full JSON for each product from `/products/{handle}.json`
3. **Build records** – Extracts title, price, images, description, infers category/size/gender
4. **Generate embeddings** – Downloads product image → SigLIP image embedding → builds text description → SigLIP text embedding
5. **Upsert to Supabase** – Batch inserts with deduplication on `(source, product_url)`
