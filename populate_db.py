import json
import os
import tiktoken
import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # loads .env file in root directory

RAW_FILE = "sdv_wiki_raw.jsonl"
CLEAN_FILE = "sdv_wiki_clean.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_CHUNK = 800

DB_DSN = os.getenv("DB_DSN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DB_DSN or not OPENAI_API_KEY:
    raise ValueError("Missing critical environment configuration. Please check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def get_embedding(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def chunk_long_text(text, max_tokens):
    """Fallback chunker if a single section exceeds token limits."""
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks


def main():
    conn = psycopg2.connect(DB_DSN)
    cursor = conn.cursor()

    print("Loading raw wikitext into memory...")
    raw_lookup = {}
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            raw_data = json.loads(line)
            raw_lookup[raw_data["page_id"]] = raw_data.get("wikitext", "")

    print("Processing clean data, generating embeddings, and inserting...")
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            m_id = page["mediawiki_id"]
            title = page["title"]
            raw_text = raw_lookup.get(m_id, "")
            full_page_text = page.get("full_text", "")

            # populate pages table
            cursor.execute("""
                INSERT INTO pages (mediawiki_id, title, url, categories, last_modified, raw_wikitext, cleaned_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (mediawiki_id) DO UPDATE SET
                    raw_wikitext = EXCLUDED.raw_wikitext,
                    cleaned_text = EXCLUDED.cleaned_text,
                    categories = EXCLUDED.categories,
                    last_modified = EXCLUDED.last_modified,
                    updated_at = NOW()
                RETURNING id;
            """, (
                m_id, title, page.get("url"), page.get("categories", []),
                page.get("last_modified"), raw_text, full_page_text
            ))

            page_db_id = cursor.fetchone()[0]

            # processing chunks
            chunks_to_insert = []
            chunk_index = 0

            for section in page.get("sections", []):
                sec_header = section["header"] if section["header"] else "Overview"
                sec_text = section["text"]

                # find exactly where this section's text begins and ends in full_text
                start_char = full_page_text.find(sec_text)
                if start_char == -1:
                    start_char = 0
                end_char = start_char + len(sec_text)

                # inject Title and Header into the chunk text
                enriched_text = f"Page: {title}\nSection: {sec_header}\n\n{sec_text}"
                token_count = len(tokenizer.encode(enriched_text))

                # check bounds and handle oversized sections
                if token_count > MAX_TOKENS_PER_CHUNK:
                    split_texts = chunk_long_text(enriched_text, MAX_TOKENS_PER_CHUNK)

                    current_start = start_char
                    for split_text in split_texts:
                        approx_len = int(len(sec_text) / len(split_texts))
                        current_end = min(current_start + approx_len, end_char)

                        chunks_to_insert.append((
                            split_text,
                            len(tokenizer.encode(split_text)),
                            current_start,
                            current_end
                        ))
                        current_start = current_end
                else:
                    chunks_to_insert.append((enriched_text, token_count, start_char, end_char))

            # embedding chunks
            if chunks_to_insert:
                # clear out old chunksn for this page to prevent duplicates on re-runs
                cursor.execute("DELETE FROM chunks WHERE page_id = %s", (page_db_id,))

                db_chunk_records = []
                for content, tokens, s_char, e_char in chunks_to_insert:
                    print(f"  Embedding chunk {chunk_index} for '{title}'...")
                    embedding_vector = get_embedding(content)

                    db_chunk_records.append((
                        page_db_id, title, page.get("categories", []), chunk_index,
                        s_char, e_char, content, tokens, embedding_vector
                    ))
                    chunk_index += 1

                # batch insert chunks using execute_values for performance
                execute_values(cursor, """
                    INSERT INTO chunks (page_id, title, categories, chunk_index, start_char, end_char, content, token_count, embedding)
                    VALUES %s
                """, db_chunk_records)

            conn.commit()
            print(f"Successfully processed and stored '{title}'.")

    cursor.close()
    conn.close()
    print("Database population complete!")


if __name__ == "__main__":
    main()
