-- Add updated_at column to products table
-- The scraper sets updated_at on every upsert, but the column was missing.
ALTER TABLE products ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
