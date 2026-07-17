"""Find PDF links on source index pages, then download them politely.

Politeness rules: identify ourselves with a User-Agent, one request per
REQUEST_DELAY seconds per host, never re-download a file whose bytes we
already have (manifest keyed by URL, content deduplicated by SHA-256).
"""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import httpx
from bs4 import BeautifulSoup

from .sources import Source

USER_AGENT = (
    "BhutanLegalAidBot/0.1 (+https://github.com/crackeo/legalaid; "
    "non-commercial legal-aid project; contact: repo issues)"
)
REQUEST_DELAY = 1.5  # seconds between requests to the same host
TIMEOUT = 60.0

_last_request_at: dict[str, float] = {}


def _throttle(url: str) -> None:
    host = urlparse(url).netloc
    elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    _last_request_at[host] = time.monotonic()


def _get(client: httpx.Client, url: str) -> httpx.Response:
    _throttle(url)
    resp = client.get(url, follow_redirects=True, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


def find_pdf_links(html: str, page_url: str, same_host_only: bool = True) -> list[dict]:
    """Extract PDF links from an index page.

    Returns [{"url": ..., "title": ...}] with absolute, deduplicated URLs
    in page order.
    """
    soup = BeautifulSoup(html, "html.parser")
    page_host = urlparse(page_url).netloc
    seen: set[str] = set()
    links: list[dict] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"].strip())
        path = urlparse(href).path
        if not path.lower().endswith(".pdf"):
            continue
        if same_host_only and urlparse(href).netloc != page_host:
            continue
        if href in seen:
            continue
        seen.add(href)
        title = a.get_text(" ", strip=True) or unquote(Path(path).stem)
        links.append({"url": href, "title": title})
    return links


def _safe_filename(url: str) -> str:
    """Stable, filesystem-safe name derived from the URL."""
    name = unquote(Path(urlparse(url).path).name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150]
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    return f"{digest}_{name}"


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def crawl_source(source: Source, raw_dir: Path, client: httpx.Client | None = None) -> dict:
    """Scrape a source's index pages and download every new/changed PDF.

    Files land in raw_dir/<source.key>/; a manifest.json in raw_dir records
    url -> {file, sha256, title, source, doc_type, fetched_at}. Unchanged
    files (same SHA-256) are not rewritten. Returns summary counts.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = raw_dir / source.key
    out_dir.mkdir(exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT})

    downloaded = skipped = failed = 0
    try:
        pdf_links: list[dict] = []
        for index_url in source.index_urls:
            html = _get(client, index_url).text
            pdf_links.extend(find_pdf_links(html, index_url, source.same_host_only))

        for link in pdf_links:
            url = link["url"]
            try:
                resp = _get(client, url)
            except httpx.HTTPError as exc:
                print(f"  FAIL {url}: {exc}")
                failed += 1
                continue
            content = resp.content
            sha = hashlib.sha256(content).hexdigest()
            entry = manifest.get(url)
            if entry and entry.get("sha256") == sha:
                skipped += 1
                continue
            fname = _safe_filename(url)
            (out_dir / fname).write_bytes(content)
            manifest[url] = {
                "file": f"{source.key}/{fname}",
                "sha256": sha,
                "title": link["title"],
                "source": source.key,
                "doc_type": source.doc_type,
                "language": source.language,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            downloaded += 1
            print(f"  saved {link['title'][:60]!r} <- {url}")
    finally:
        if own_client:
            client.close()

    return {"found": len(pdf_links), "downloaded": downloaded,
            "skipped_unchanged": skipped, "failed": failed}
