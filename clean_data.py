"""
clean_data.py
-------------
Reads the raw JSONL file produced by fetch_data.py, cleans the wikitext
for each page, and writes a new JSONL file with the cleaned plain text.

What "cleaning" means here:
    - Skip redirect pages entirely (they have no real content)
    - Parse wikitext with mwparserfromhell to strip MediaWiki markup
    - Remove or normalize leftover artifacts (excess whitespace, etc.)
    - Preserve section headers as structured metadata for chunking later

Dependencies:
    pip install mwparserfromhell tiktoken

Usage:
    python clean_data.py

Input:
    sdv_wiki_raw.jsonl      (produced by fetch_data.py)

Output:
    sdv_wiki_clean.jsonl    (one JSON object per line, cleaned)
"""

import json
import re
import mwparserfromhell

INPUT_FILE = "sdv_wiki_raw.jsonl"
OUTPUT_FILE = "sdv_wiki_clean.jsonl"

WIKI_BASE_URL = "https://stardewvalleywiki.com/"


def is_redirect(wikitext):
    """
    Returns True if the page is a MediaWiki redirect.

    Redirect pages contain no real content — they exist solely to forward
    one title to another (e.g. "SDV" -> "Stardew Valley"). Their wikitext
    always starts with #REDIRECT followed by the target page in brackets.

    Example redirect wikitext:
        #REDIRECT [[Stardew Valley]]
    """
    return wikitext.strip().upper().startswith("#REDIRECT")


def preprocess_templates(parsed):
    """
    Walks template nodes in the parsed wikicode and replaces specific
    templates with plain text equivalents before strip_code() runs.
    strip_code() drops templates entirely, so this is necessary to
    prevent loss of information.

    This approach ended up not working, because of parsing with flat=False.
    Because of this, the templates are nested, and can't be found with
    parsed.replace(). I wanted to keep the nested structure to be stored,
    so I chose to go with the regex approach instead, which happens before
    the parsing. I left this function here just to have a record of it.
    """
    for template in parsed.filter_templates(recursive=True):
        name = template.name.strip().lower()

        # character quotes
        if name == "squote":
            # first param is the quote text, optional second is speaker (which isn't needed)
            if template.params:
                quote_text = template.get(1).value.strip()
                parsed.replace(template, f'"{quote_text}"\n')
            else:
                parsed.replace(template, "")

        # prices
        elif name in ("price", "qualityprice", "tprice"):
            # first param is always the gold value
            if template.params:
                amount = template.get(1).value.strip()
                parsed.replace(template, f"{amount}g")
            else:
                parsed.replace(template, "")

        # NPC names
        elif name in ("name", "npc"):
            # first param is the item or NPC name, second (optional) is a label
            if template.params:
                item_name = template.get(1).value.strip()
                try:
                    label = template.get(2).value.strip()
                    parsed.replace(template, f"{item_name} ({label})")
                except ValueError:
                    parsed.replace(template, item_name)
            else:
                parsed.replace(template, "")

        # seasons
        elif name == "season":
            if template.params:
                parsed.replace(template, template.get(1).value.strip())
            else:
                parsed.replace(template, "")


def convert_wikitables(wikitext):
    """
    Converts MediaWiki table syntax to markdown tables.

    MediaWiki table syntax:
        {| ...attributes...     <- table open
        ! Header                <- header cell (! prefix)
        ! Header2               <- another header (own line)
        ! Header1 !! Header2    <- multiple headers on one line
        |-                      <- row separator
        | Cell                  <- data cell (| prefix)
        | Cell1 || Cell2        <- multiple cells on one line
        |}                      <- table close

    We convert this to:
        | Header1 | Header2 |
        |---------|---------|
        | Cell1   | Cell2   |
    """

    def convert_single_table(match):
        # match.group(0) is the full table text including {| and |}
        table_text = match.group(0)
        lines = table_text.split('\n')

        headers = []
        rows = []
        current_row = []

        for line in lines:
            line = line.strip()

            # skip table open/close and row separators
            if line.startswith('{|') or line == '|}':
                continue
            # when we hit a row separator, save the current row if it has content
            if line.startswith('|-'):
                if current_row:
                    rows.append(current_row)
                    current_row = []
                continue

            # header cells — start with !
            if line.startswith('!'):
                # strip the ! and handle both single-line and multi-header syntax
                # e.g. "!style=\"...\" | Name" -> "Name"
                # e.g. "!Name !! Description" -> ["Name", "Description"]
                header_content = line.lstrip('!').strip()

                # handle !! separator for multiple headers on one line
                parts = [p.strip() for p in header_content.split('!!')]
                for part in parts:
                    # if there's a | in the part, everything after it is the actual text
                    # e.g. 'style="width:48px;" | Image' -> 'Image'
                    if re.match(r'^[^\[\]{}|]+\|', part):
                        part = part.split('|', 1)[-1].strip()
                    part = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', part)
                    if part and not part.lower().startswith('file:'):
                        headers.append(part)
                continue

            # data cells — start with |
            if line.startswith('|'):
                # strip leading |
                cell_content = line.lstrip('|').strip()

                # handle || separator for multiple cells on one line
                parts = [p.strip() for p in cell_content.split('||')]
                for part in parts:
                    # strip cell attributes — if there's a | the content follows it
                    # e.g. 'colspan="3" | All Universal Likes' -> 'All Universal Likes'
                    if re.match(r'^[^\[\]{}|]+\|', part):
                        part = part.split('|', 1)[-1].strip()
                    part = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', part)
                    # skip cells that are just file references (images)
                    if not part.lower().startswith('file:'):
                        current_row.append(part)
                continue

        # save the last row if it has content
        if current_row:
            rows.append(current_row)

        rows = [[cell for cell in row if cell] for row in rows]
        rows = [row for row in rows if row]

        # if we got nothing useful, return empty string to drop the table
        if not headers and not rows:
            return ''

        # use headers if we have them, otherwise skip the header row
        md_lines = []
        if headers:
            # expand headers if rows have more columns (prevents schedule truncation)
            max_cols = max([len(headers)] + [len(r) for r in rows])
            while len(headers) < max_cols:
                headers.append('')

            md_lines.append('| ' + ' | '.join(headers) + ' |')
            md_lines.append('|' + '---|' * len(headers))
            for row in rows:
                while len(row) < len(headers):
                    row.append('')
                md_lines.append('| ' + ' | '.join(row[:len(headers)]) + ' |')
        else:
            for row in rows:
                md_lines.append('| ' + ' | '.join(row) + ' |')

        return '\n\n' + '\n'.join(md_lines) + '\n\n'

    # process tables from innermost to outermost to handle nested layout tables
    while re.search(r'\{\|(?:(?!\{\|).)*?\|\}', wikitext, flags=re.DOTALL):
        wikitext = re.sub(r'\{\|(?:(?!\{\|).)*?\|\}', convert_single_table, wikitext, flags=re.DOTALL)

    return wikitext


def preprocess_wikitext(wikitext):
    """
    Replaces specific templates with plain text equivalents using regex,
    before mwparserfromhell parsing. This avoids issues with replacing
    templates found in nested contexts during tree traversal.
    """
    # {{Price|250}} -> 250g
    wikitext = re.sub(r'\{\{[Pp]rice\|([^}]+)\}\}', lambda m: f"{m.group(1).strip()}g", wikitext)
    wikitext = re.sub(r'\{\{[Qq]ualityprice\|([^}]+)\}\}', lambda m: f"{m.group(1).split('|')[0].strip()}g", wikitext)
    wikitext = re.sub(r'\{\{[Tt]price\|([^}]+)\}\}', lambda m: f"{m.group(1).strip()}g", wikitext)

    # {{Recipe|Pumpkin Soup|48}} -> Pumpkin Soup, {{Recipe|Omelet}} -> Omelet
    wikitext = re.sub(r'\{\{[Rr]ecipe\|([^|}]+)(?:\|[^}]*)?\}\}', lambda m: m.group(1).strip(), wikitext)

    # {{Name|Parsnip}} -> Parsnip, {{Name|Parsnip|5}} -> Parsnip
    wikitext = re.sub(r'\{\{[Nn]ame\|([^|}]+)(?:\|[^}]*)?\}\}', lambda m: m.group(1).strip(), wikitext)

    # {{NPC|Robin|Husband}} -> Robin (Husband), {{NPC|Robin}} -> Robin
    def npc_replace(m):
        parts = m.group(1).split("|")
        return f"{parts[0].strip()} ({parts[1].strip()})" if len(parts) > 1 else parts[0].strip()
    wikitext = re.sub(r'\{\{[Nn][Pp][Cc]\|([^}]+)\}\}', npc_replace, wikitext)

    # {{Season|Fall}} -> Fall
    wikitext = re.sub(r'\{\{[Ss]eason\|([^}]+)\}\}', lambda m: m.group(1).strip(), wikitext)

    # {{Squote|text}} or {{Squote|text|speaker}} -> "text"
    wikitext = re.sub(r'\{\{[Ss]quote\|([^|}]+)(?:\|[^}]*)?\}\}', lambda m: f'"{m.group(1).strip()}"', wikitext)

    # these are structural blocks that get removed entirely
    wikitext = re.sub(r'\{\{[Dd]escription\|[^}]+\}\}', '', wikitext)
    wikitext = re.sub(r'\{\{Infobox villager.*?\}\}', '', wikitext, flags=re.DOTALL)
    wikitext = re.sub(r'\{\{Main article.*?\}\}', '', wikitext, flags=re.IGNORECASE)
    wikitext = re.sub(r'\{\{[A-Za-z]+\}\}', '', wikitext)

    # convert wikitables to markdown
    wikitext = convert_wikitables(wikitext)

    return wikitext


def extract_sections(wikitext):
    """
    Parses the wikitext and returns a list of section dicts, each containing:
        - header: the section title as a plain string (e.g. "Gifts")
                  None for the lead section (text before the first header)
        - level:  the header depth (1 = top, 2 = subsection, etc.)
                  None for the lead section
        - text:   the cleaned plain text content of the section

    mwparserfromhell parses wikitext into a tree of nodes. Its
    get_sections() method does the hard work of identifying where
    each section starts and ends.
    """
    wikitext = preprocess_wikitext(wikitext)
    parsed = mwparserfromhell.parse(wikitext)  # Wikicode object
    # preprocess_templates(parsed) <- Old call, does not work

    raw_sections = parsed.get_sections(
    )  # list of Wikicode objects

    sections = []

    for section in raw_sections:
        headings = section.filter_headings()

        if headings:
            heading = headings[0]
            header_text = heading.title.strip_code().strip()
            level = heading.level
            section.remove(heading)
        else:
            # no heading means this is the lead section
            header_text = None
            level = None

        plain_text = section.strip_code()   # converts all remaining wikitext markup to plain text
        plain_text = clean_plain_text(plain_text)  # remove any artifacts that strip_code() leaves behind.

        # skip sections that are empty after cleaning
        if not plain_text:
            continue

        sections.append({
            "header": header_text,
            "level": level,
            "text": plain_text,
        })

    return sections


def clean_plain_text(text):
    """
    Takes a plain text string (already stripped of wikitext markup by
    strip_code()) and removes remaining artifacts.

    strip_code() does most of the heavy lifting, but it leaves behind:
        - Excess blank lines (sometimes 3-4 in a row)
        - Leading/trailing whitespace within lines
        - Occasional stray punctuation from template boundaries
        - File/image references that weren't fully removed
    """
    # remove explicit asset sizes and link flags (e.g., "24px", "center|link=", "|link=None")
    text = re.sub(r'\b\d+px\b', '', text)
    text = re.sub(r'center\|link=', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\|link=[^\s]*', '', text)
    text = re.sub(r'\blink=\b', '', text)

    # remove stray layout attributes from nested layout tables (e.g., |style="width: 15%;"|)
    text = re.sub(r'\|?style="[^"]*"\|?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\|?class="[^"]*"\|?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\|?colspan="[^"]*"\|?', '', text, flags=re.IGNORECASE)

    # remove file or image references (e.g., "File:Robin.png", "Image:Axe.png")
    text = re.sub(r'\b(File|Image):[^\s]+', '', text, flags=re.IGNORECASE)

    # remove explicit category links (e.g., "[[Category:NPCs]]")
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text, flags=re.IGNORECASE)

    # remove interlanguage wiki links (e.g., "de:Robin", "ja:ロビン")
    text = re.sub(r'^[a-z]{2,3}:[^\n]+', '', text, flags=re.MULTILINE)

    # remove thumb image artifacts (e.g., "thumb|right|120px|Robin's lost axe")
    text = re.sub(r'\bthumb\|[^\n]+', '', text)

    # remove wikitable cell noise — "48px|center", and standalone "center" tags
    text = re.sub(r'\b\d+px\|center\b', '', text)
    text = re.sub(r'center(?=[A-Z])', '', text)

    # remove explicit asset sizes and link flags (even if smushed against words)
    text = re.sub(r'\d+px', '', text, flags=re.IGNORECASE)
    text = re.sub(r'link=', '', text, flags=re.IGNORECASE)

    # remove stranded "center" alignment tags in markdown tables
    text = re.sub(r'\|\s*center\s*\|', '| |', text, flags=re.IGNORECASE)

    # remove stray layout attributes from nested layout tables
    text = re.sub(r'\|?style="[^"]*"\|?', '', text, flags=re.IGNORECASE)

    # remove stray raw HTML formatting elements leaking through templates (e.g., <br>, <p>)
    text = re.sub(r'<[^>]+>', '', text)

    # remove any trailing/leading whitespace and eliminate completely empty lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)

    # turn >2 newlines into 2 (which normalizes the text to standard paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def build_url(title):
    """
    Constructs the canonical wiki URL for a page given its title.
    """
    slug = title.replace(" ", "_")
    return WIKI_BASE_URL + slug


def process_page(raw_page):
    """
    Takes a raw page dict (as loaded from sdv_wiki_raw.jsonl) and returns
    a cleaned page dict, or None if the page should be skipped.

    The returned dict contains everything needed for the chunking step:
        - page_id, mediawiki_id, title, url, categories, last_modified
          (carried over from the raw data)
        - sections: list of {header, level, text} dicts
        - full_text: all section text joined together, useful for
          debugging and for pages to be stored as a single chunk
    """
    wikitext = raw_page.get("wikitext", "")

    if is_redirect(wikitext):
        return None

    if not wikitext.strip():
        return None

    sections = extract_sections(wikitext)

    if not sections:
        return None

    # put all of a section's text into a single string
    full_text = "\n\n".join(
        (f"{s['header']}\n{s['text']}" if s['header'] else s['text'])
        for s in sections
    )

    return {
        "mediawiki_id": raw_page["page_id"],
        "title": raw_page["title"],
        "url": build_url(raw_page["title"]),
        "categories": raw_page.get("categories", []),
        "last_modified": raw_page.get("last_modified", ""),
        "sections": sections,
        "full_text": full_text,
    }


def main():
    total_read = 0
    total_written = 0
    total_skipped = 0

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as in_file,
        open(OUTPUT_FILE, "w", encoding="utf-8") as out_file,
    ):
        for line in in_file:
            total_read += 1

            raw_page = json.loads(line)

            cleaned = process_page(raw_page)

            if cleaned is None:
                total_skipped += 1
                continue

            out_file.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            total_written += 1

            # print progress every 100 pages so it's clear something is happening
            if total_written % 100 == 0:
                print(f"  Cleaned {total_written} pages so far...")

    print(f"\nDone.")
    print(f"  Read:    {total_read}")
    print(f"  Written: {total_written}")
    print(f"  Skipped: {total_skipped} (redirects + empty pages)")
    print(f"\nOutput saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
