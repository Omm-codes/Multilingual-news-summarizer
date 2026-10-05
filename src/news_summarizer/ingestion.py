from urllib.parse import urlparse

import trafilatura

MAX_URL_LENGTH = 2_048
MAX_ARTICLE_CHARACTERS = 100_000


def extract_url_text(url: str) -> tuple[str, str]:
    cleaned_url = url.strip()
    if not cleaned_url:
        raise ValueError("Please enter an article URL.")
    if len(cleaned_url) > MAX_URL_LENGTH:
        raise ValueError("Please enter a shorter article URL.")
    parsed_url = urlparse(cleaned_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Please enter a valid HTTP or HTTPS article URL.")
    downloaded = trafilatura.fetch_url(cleaned_url)
    if not downloaded:
        raise ValueError("The article could not be downloaded. Check the URL or paste the text instead.")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError("No readable article text was found at this URL.")
    if len(text) > MAX_ARTICLE_CHARACTERS:
        raise ValueError("The downloaded article is too large. Paste a shorter article instead.")
    return text, cleaned_url
