"""
crawler.py – Web crawler for https://quotes.toscrape.com/

This module provides a polite, breadth-first web crawler that systematically
traverses the target website, extracts textual content and metadata from each
page, and returns structured page data suitable for indexing.

AI Declaration:
    AI tools (GitHub Copilot / ChatGPT) were used to assist in structuring the
    overall module layout, suggesting best practices for politeness delays, and
    reviewing error-handling patterns.  All logic was verified and adapted by
    the author.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BASE_URL: str = "https://quotes.toscrape.com/"
POLITENESS_DELAY: float = 6.0  # seconds between successive HTTP requests
REQUEST_TIMEOUT: int = 30  # seconds before a request is considered timed out
MAX_RETRIES: int = 3  # maximum number of retry attempts per URL
RETRY_BACKOFF: float = 2.0  # exponential back-off multiplier
USER_AGENT: str = (
    "UniversitySearchBot/1.0 "
    "(+https://github.com/DavidFeng-8844/WebServicesAndWebDataCwk2; "
    "educational project)"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuoteData:
    """Structured representation of a single quote scraped from a page.

    Attributes:
        text: The full quote text.
        author: Author name.
        tags: List of tags associated with the quote.
    """

    text: str
    author: str
    tags: List[str] = field(default_factory=list)


@dataclass
class PageData:
    """Structured representation of a single crawled page.

    Attributes:
        url: The canonical URL of the page.
        title: The ``<title>`` tag content.
        quotes: List of :class:`QuoteData` objects extracted from the page.
        author_info: Mapping of author name → bio text (for author pages).
        raw_text: All visible text on the page, concatenated.
    """

    url: str
    title: str = ""
    quotes: List[QuoteData] = field(default_factory=list)
    author_info: Dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class WebCrawler:
    """Polite breadth-first crawler for ``quotes.toscrape.com``.

    The crawler respects a strict *politeness window* of at least
    :pydata:`POLITENESS_DELAY` seconds between successive HTTP requests.  It
    employs exponential back-off on transient network errors and limits itself
    to pages within the same domain.

    Usage::

        crawler = WebCrawler()
        pages = crawler.crawl()

    Args:
        base_url: The seed URL to start crawling from.
        politeness_delay: Minimum seconds between successive requests.
        max_pages: Optional upper limit on the number of pages to crawl.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        politeness_delay: float = POLITENESS_DELAY,
        max_pages: Optional[int] = None,
        proxy_url: Optional[str] = None,
    ) -> None:
        self._base_url: str = base_url
        self._politeness_delay: float = politeness_delay
        self._max_pages: Optional[int] = max_pages

        self._visited: Set[str] = set()
        self._queue: List[str] = [self._base_url]
        self._pages: List[PageData] = []

        # Persistent session for connection pooling and consistent headers.
        self._session: requests.Session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

        # Proxy configuration: explicit > environment variable > none.
        resolved_proxy: Optional[str] = self._resolve_proxy(proxy_url)
        if resolved_proxy:
            self._session.proxies.update(
                {"http": resolved_proxy, "https": resolved_proxy}
            )
            logger.info("Using proxy: %s", resolved_proxy)

        self._last_request_time: float = 0.0

    @staticmethod
    def _resolve_proxy(proxy_url: Optional[str] = None) -> Optional[str]:
        """Determine the proxy URL to use, if any.

        Priority: explicit *proxy_url* > ``HTTP_PROXY`` / ``HTTPS_PROXY``
        environment variables > ``None`` (direct connection).

        Args:
            proxy_url: An explicitly provided proxy URL.

        Returns:
            A proxy URL string, or ``None`` for a direct connection.
        """

        if proxy_url:
            return proxy_url

        # Check common environment variables (case-insensitive on Windows).
        for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env_val: Optional[str] = os.environ.get(var)
            if env_val:
                return env_val

        return None

    # -- public API ---------------------------------------------------------

    def crawl(self) -> List[PageData]:
        """Execute the crawl and return a list of :class:`PageData` objects.

        Returns:
            A list of successfully crawled and parsed pages.
        """

        logger.info("Starting crawl from %s", self._base_url)

        while self._queue:
            if self._max_pages is not None and len(self._pages) >= self._max_pages:
                logger.info("Reached max page limit (%d). Stopping.", self._max_pages)
                break

            url: str = self._queue.pop(0)
            normalised: str = self._normalise_url(url)

            if normalised in self._visited:
                continue

            self._visited.add(normalised)

            html: Optional[str] = self._fetch(normalised)
            if html is None:
                continue

            page_data: Optional[PageData] = self._parse(normalised, html)
            if page_data is not None:
                self._pages.append(page_data)
                logger.info(
                    "Crawled [%d]: %s (%d quotes)",
                    len(self._pages),
                    normalised,
                    len(page_data.quotes),
                )

            # Discover new links and enqueue them.
            self._enqueue_links(normalised, html)

        logger.info("Crawl complete. Total pages: %d", len(self._pages))
        return self._pages

    # -- private helpers ----------------------------------------------------

    def _wait_for_politeness(self) -> None:
        """Block until the politeness window has elapsed since the last request."""

        elapsed: float = time.time() - self._last_request_time
        if elapsed < self._politeness_delay:
            sleep_time: float = self._politeness_delay - elapsed
            logger.debug("Politeness delay: sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)

    def _fetch(self, url: str) -> Optional[str]:
        """Fetch raw HTML from *url* with retries and exponential back-off.

        Args:
            url: The fully-qualified URL to request.

        Returns:
            The response body as a string, or ``None`` on failure.
        """

        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_for_politeness()
            try:
                response: requests.Response = self._session.get(
                    url, timeout=REQUEST_TIMEOUT
                )
                self._last_request_time = time.time()
                response.raise_for_status()
                return response.text
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else "N/A"
                logger.warning(
                    "HTTP %s for %s (attempt %d/%d)",
                    status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                # Do not retry client errors (4xx) other than 429.
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    if exc.response.status_code != 429:
                        return None
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Connection error for %s (attempt %d/%d)",
                    url,
                    attempt,
                    MAX_RETRIES,
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout for %s (attempt %d/%d)", url, attempt, MAX_RETRIES
                )
            except requests.exceptions.RequestException as exc:
                logger.error("Unexpected request error for %s: %s", url, exc)
                return None

            # Exponential back-off before retry.
            backoff: float = RETRY_BACKOFF ** attempt
            logger.debug("Backing off %.1fs before retry", backoff)
            time.sleep(backoff)

        logger.error("Failed to fetch %s after %d attempts", url, MAX_RETRIES)
        return None

    def _parse(self, url: str, html: str) -> Optional[PageData]:
        """Parse *html* and build a :class:`PageData` object.

        Delegates to specialised parsers depending on the URL pattern:
        - Quote listing pages (``/``, ``/page/N/``)
        - Author biography pages (``/author/...``)
        - Tag listing pages (``/tag/...``)

        Args:
            url: The URL that produced *html*.
            html: The raw HTML string.

        Returns:
            A populated :class:`PageData` instance, or ``None`` on parse failure.
        """

        try:
            soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        except Exception as exc:  # pragma: no cover – extremely unlikely
            logger.error("Failed to parse HTML from %s: %s", url, exc)
            return None

        title: str = soup.title.string.strip() if soup.title and soup.title.string else ""
        raw_text: str = self._extract_visible_text(soup)

        page = PageData(url=url, title=title, raw_text=raw_text)

        # Detect page type and extract structured data accordingly.
        if "/author/" in url:
            page.author_info = self._parse_author_page(soup)
        else:
            page.quotes = self._parse_quotes(soup)

        return page

    @staticmethod
    def _parse_quotes(soup: BeautifulSoup) -> List[QuoteData]:
        """Extract quote data from a quote-listing page.

        Args:
            soup: Parsed BeautifulSoup tree of the page.

        Returns:
            A list of :class:`QuoteData` instances.
        """

        quotes: List[QuoteData] = []
        for quote_div in soup.select("div.quote"):
            text_span: Optional[Tag] = quote_div.select_one("span.text")
            author_span: Optional[Tag] = quote_div.select_one("small.author")
            tag_anchors = quote_div.select("a.tag")

            text: str = text_span.get_text(strip=True) if text_span else ""
            author: str = author_span.get_text(strip=True) if author_span else ""
            tags: List[str] = [a.get_text(strip=True) for a in tag_anchors]

            # Strip surrounding quotation marks (unicode left/right).
            text = text.strip("\u201c\u201d\"")

            if text:
                quotes.append(QuoteData(text=text, author=author, tags=tags))

        return quotes

    @staticmethod
    def _parse_author_page(soup: BeautifulSoup) -> Dict[str, str]:
        """Extract author biography information from an author detail page.

        Args:
            soup: Parsed BeautifulSoup tree of the page.

        Returns:
            A dict mapping the author name to their biography text.
        """

        info: Dict[str, str] = {}
        name_tag: Optional[Tag] = soup.select_one("h3.author-title")
        bio_tag: Optional[Tag] = soup.select_one("div.author-description")

        name: str = name_tag.get_text(strip=True) if name_tag else "Unknown"
        bio: str = bio_tag.get_text(strip=True) if bio_tag else ""

        if name:
            info[name] = bio

        return info

    @staticmethod
    def _extract_visible_text(soup: BeautifulSoup) -> str:
        """Return all visible text from the page, collapsed to single spaces.

        Non-content elements (``<script>``, ``<style>``, ``<nav>``) are removed
        before extraction.

        Args:
            soup: Parsed BeautifulSoup tree.

        Returns:
            A single string of visible text.
        """

        # Remove non-content elements.
        for tag in soup(["script", "style", "noscript", "nav"]):
            tag.decompose()

        text: str = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace.
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _enqueue_links(self, source_url: str, html: str) -> None:
        """Discover same-domain links in *html* and add unseen ones to the queue.

        Only links within the ``quotes.toscrape.com`` domain are followed.
        Login, tag-pagination, and external links are excluded.

        Args:
            source_url: The page the links were discovered on.
            html: The raw HTML of that page.
        """

        try:
            soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        except Exception:
            return

        base_domain: str = urlparse(self._base_url).netloc

        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            full_url: str = urljoin(source_url, href)
            normalised: str = self._normalise_url(full_url)
            parsed = urlparse(normalised)

            # Stay within the target domain.
            if parsed.netloc != base_domain:
                continue

            # Skip login and other non-content paths.
            if "/login" in parsed.path:
                continue

            if normalised not in self._visited:
                self._queue.append(normalised)

    @staticmethod
    def _normalise_url(url: str) -> str:
        """Normalise a URL by stripping fragments and trailing slashes.

        Args:
            url: The URL to normalise.

        Returns:
            A canonical form of the URL.
        """

        parsed = urlparse(url)
        # Rebuild without fragment.
        normalised: str = parsed._replace(fragment="").geturl()
        # Remove trailing slash for consistency (but keep root "/").
        if normalised.endswith("/") and parsed.path != "/":
            normalised = normalised.rstrip("/")
        return normalised
