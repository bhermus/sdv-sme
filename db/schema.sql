-- db/schema.sql
-- SDV_SME database schema
-- Run once against an empty sdv_sme database with pgvector already installed.
-- Safe to re-run due to IF NOT EXISTS guards.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- pages
-- Source of truth. One row per wiki page, stores raw wikitext.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pages (
    id              SERIAL PRIMARY KEY,
    mediawiki_id    INTEGER UNIQUE NOT NULL,
    title           TEXT UNIQUE NOT NULL,
    url             TEXT,
    categories      TEXT[],
    last_modified   TIMESTAMPTZ,
    raw_wikitext    TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- chunks
-- Retrieval layer. One row per chunk, stores cleaned text and embedding.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    page_id         INTEGER REFERENCES pages(id) ON DELETE CASCADE,
    title           TEXT,
    categories      TEXT[],
    chunk_index     INTEGER,
    start_char      INTEGER,
    end_char        INTEGER,
    content         TEXT,
    token_count     INTEGER,
    embedding       vector(1536),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- pages: fast lookup by mediawiki ID and title
CREATE INDEX IF NOT EXISTS idx_pages_mediawiki_id
    ON pages(mediawiki_id);

CREATE INDEX IF NOT EXISTS idx_pages_title
    ON pages(title);

-- chunks: fast join back to parent page
CREATE INDEX IF NOT EXISTS idx_chunks_page_id
    ON chunks(page_id);

-- chunks: HNSW vector similarity index
-- cosine distance is standard for text embeddings
-- m and ef_construction control index quality vs build time
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);