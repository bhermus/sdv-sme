"""
fetch_sdv_wiki.py
-----------------
Downloads every content page from the Stardew Valley wiki via the MediaWiki API
and saves them to a local JSONL file (one JSON object per line).

Save raw wikitext here — no cleaning, no chunking yet. The goal of this
script is just to get a reliable local cache so there's never a need to hit the
API again. All the messy transformation work happens in the next step.

Dependencies:
    pip install requests

Usage:
    python fetch_sdv_wiki.py

Output:
    sdv_wiki_raw.jsonl  (created in the same directory as the script)
"""

import json
import time
import requests

USER_EMAIL = "put_your@email.here"

API_URL = "https://stardewvalleywiki.com/mediawiki/api.php"
BATCH_SIZE = 50   # pages to request per API call
REQUEST_DELAY = 0.5
OUTPUT_FILE = "sdv_wiki_raw.jsonl"  # each line is a self-contained JSON object
HEADERS = {
    "User-Agent": "SDV-SME-Bot/1.0 (portfolio project; your@email.com)"
}


def get_all_page_titles():
    """
    Uses the MediaWiki `allpages` list to retrieve every title in the
    main namespace (namespace 0 = actual content, not talk pages, user
    pages, etc.).

    The API returns results in batches. When there are more results, it
    includes an `apcontinue` token in the response — we use that token in
    the next request to pick up where we left off. This is called pagination.

    Returns a list of page title strings.
    """
    titles = []

    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,  # main content namespace
        "aplimit": BATCH_SIZE,
        "format": "json",
    }

    print("Fetching all page titles...")

    # loop until API stops giving a continuation token
    while True:
        response = requests.get(API_URL, params=params, headers=HEADERS)

        response.raise_for_status()  # crash early, don't want partial data

        data = response.json()
        pages = data["query"]["allpages"]
        for page in pages:
            titles.append(page["title"])

        print(f"  Retrieved {len(titles)} titles so far...")

        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
        else:
            # done
            break

        # for API politeness
        time.sleep(REQUEST_DELAY)

    print(f"Done. Total titles found: {len(titles)}\n")
    return titles


def fetch_pages_batch(titles_batch):
    """
    Given a list of page titles (up to BATCH_SIZE), fetches the full wikitext content
    and category membership for each page in a single API call.

    MediaWiki lets you query multiple pages at once using a pipe-separated
    title list. This is much more efficient than one request per page.

    Returns a dict keyed by page title, each value being a dict with:
        - page_id      (int)
        - title        (str)
        - wikitext     (str)   raw wikitext markup
        - categories   (list)  list of category name strings
        - last_modified (str)  ISO 8601 timestamp
    """

    titles_str = "|".join(titles_batch)

    params = {
        "action": "query",
        "titles": titles_str,

        # prop=revisions asks for revision data; by default it gets the latest revision, which is the current page
        # prop=categories asks for the categories the page belongs to
        "prop": "revisions|categories",

        # rvprop=content means: within revision data, give us the actual page text
        # rvprop=timestamp gives us when it was last edited.
        "rvprop": "content|timestamp",

        # rvslots=main means we want the main content slot (as opposed to talk pages etc.)
        "rvslots": "main",

        # cllimit=max asks for all categories a page belongs to (up to 500)
        "cllimit": "max",

        "format": "json",
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    # API returns pages in a dict keyed by page ID; rekeyed by title for convenience
    raw_pages = data["query"]["pages"]

    results = {}

    for page_id_str, page_data in raw_pages.items():

        # page_id of -1 means the title doesnt exist
        if page_id_str == "-1":
            continue

        title = page_data["title"]

        # content lives inside revisions -> slots -> main -> content.
        try:
            wikitext = (
                page_data["revisions"][0]["slots"]["main"]["*"]
            )
        except (KeyError, IndexError):
            wikitext = ""

        try:
            last_modified = page_data["revisions"][0]["timestamp"]
        except (KeyError, IndexError):
            last_modified = ""

        raw_cats = page_data.get("categories", [])
        categories = [
            cat["title"].replace("Category:", "")
            for cat in raw_cats
        ]

        results[title] = {
            "page_id": int(page_id_str),
            "title": title,
            "wikitext": wikitext,
            "categories": categories,
            "last_modified": last_modified,
        }

    return results


def chunk_list(lst, size):
    """
    Splits a flat list into consecutive sublists of length `size`.
    The last sublist may be shorter if len(lst) isn't evenly divisible.

    Example: chunk_list([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]

    This is a generator — it yields one batch at a time rather than
    building the whole list of batches in memory at once.
    """
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def main():
    all_titles = get_all_page_titles()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

        total_written = 0
        batches = list(chunk_list(all_titles, BATCH_SIZE))

        print(f"Fetching content for {len(all_titles)} pages in {len(batches)} batches...\n")

        for batch_num, batch in enumerate(batches, start=1):
            print(f"  Batch {batch_num}/{len(batches)}: fetching {len(batch)} pages...")

            pages = fetch_pages_batch(batch)

            for title, page_data in pages.items():

                # ensure_ascii=False preserves non-ASCII characters as-is
                line = json.dumps(page_data, ensure_ascii=False)

                out_file.write(line + "\n")

                total_written += 1

            print(f"    Written {total_written} pages total so far.")

            # for API politeness
            time.sleep(REQUEST_DELAY)

    print(f"\n{total_written} pages saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
