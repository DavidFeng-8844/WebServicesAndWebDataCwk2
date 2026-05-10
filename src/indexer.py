"""
indexer.py – Inverted index builder and manager.

This module constructs a case-insensitive inverted index from crawled page data.
For every term it records:
    • document frequency (number of pages containing the term),
    • per-document term frequency,
    • positional information (word offsets within each document).

The index is serialised/deserialised to/from a single JSON file so that it can
be rebuilt once and reused across sessions.

AI Declaration:
    AI tools (GitHub Copilot / ChatGPT) were used to assist in designing the
    inverted-index data structure, suggesting optimisations for TF-IDF look-ups,
    and reviewing serialisation strategies.  All logic was verified and adapted
    by the author.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Import PageData for type hints – if only crawled data is available.
try:
    from crawler import PageData
except ImportError:  # pragma: no cover
    from src.crawler import PageData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Robustly determine the project root directory (one level up from src/)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

DEFAULT_INDEX_PATH: str = os.path.join(_PROJECT_ROOT, "data", "search_index.json")

# Regex: word-boundary tokeniser that keeps alphabetic + numeric tokens ≥ 1 char.
_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-zA-Z0-9]+")

# Common English stop-words (kept minimal to avoid over-filtering).
STOP_WORDS: Set[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "it",
        "its",
        "this",
        "that",
        "from",
        "as",
        "be",
        "has",
        "had",
        "have",
        "do",
        "does",
        "did",
        "not",
        "so",
        "if",
        "no",
        "up",
    }
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TermPosting:
    """Statistics for a single term within a single document.

    Attributes:
        doc_url: The URL (document identifier) where the term appears.
        term_frequency: Raw count of occurrences of the term in the document.
        positions: Zero-based word-offset positions of each occurrence.
    """

    doc_url: str
    term_frequency: int = 0
    positions: List[int] = field(default_factory=list)


@dataclass
class TermEntry:
    """Aggregated statistics for a single term across the entire corpus.

    Attributes:
        document_frequency: Number of distinct documents containing the term.
        postings: Mapping of document URL → :class:`TermPosting`.
    """

    document_frequency: int = 0
    postings: Dict[str, TermPosting] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Indexer class
# ---------------------------------------------------------------------------


class InvertedIndex:
    """Case-insensitive inverted index with positional data.

    The index maps each normalised (lower-cased) token to a :class:`TermEntry`
    containing per-document frequencies and positional offsets.  This enables
    efficient TF-IDF scoring and phrase/proximity queries.

    Usage::

        indexer = InvertedIndex()
        indexer.build(pages)       # pages: List[PageData]
        indexer.save("index.json")

        # Later …
        indexer.load("index.json")

    Attributes:
        index: The core inverted-index mapping (term → TermEntry).
        total_docs: Total number of documents in the corpus.
        doc_lengths: Mapping of document URL → total number of tokens.
        avg_doc_length: Average document length across the corpus.
        doc_urls: Ordered list of all document URLs.
    """

    def __init__(self) -> None:
        self.index: Dict[str, TermEntry] = {}
        self.total_docs: int = 0
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_urls: List[str] = []
        # Store page titles for display purposes.
        self._doc_titles: Dict[str, str] = {}

    # -- public API ---------------------------------------------------------

    def build(self, pages: List[PageData]) -> None:
        """Build the inverted index from a list of crawled pages.

        Time Complexity: O(N * L) where N is the number of pages, and L is the average
                         number of tokens per page. We iterate through each token once.
        Space Complexity: O(V + N * L) where V is the vocabulary size (unique terms),
                          and we store positional occurrences for each token.

        Args:
            pages: A list of :class:`PageData` objects produced by the crawler.
        """

        logger.info("Building inverted index from %d pages …", len(pages))

        self.index.clear()
        self.doc_lengths.clear()
        self._doc_titles.clear()
        self.doc_urls = []
        self.total_docs = len(pages)

        for page in pages:
            url: str = page.url
            self.doc_urls.append(url)
            self._doc_titles[url] = page.title

            # Build a rich text corpus from all available content on the page.
            text: str = self._build_document_text(page)
            tokens: List[str] = self.tokenise(text)
            self.doc_lengths[url] = len(tokens)

            # Populate per-term postings.
            term_positions: Dict[str, List[int]] = defaultdict(list)
            for position, token in enumerate(tokens):
                term_positions[token].append(position)

            for term, positions in term_positions.items():
                if term not in self.index:
                    self.index[term] = TermEntry()
                entry: TermEntry = self.index[term]
                entry.document_frequency += 1
                entry.postings[url] = TermPosting(
                    doc_url=url,
                    term_frequency=len(positions),
                    positions=positions,
                )

        # Pre-compute average document length for BM25-style normalisation.
        total_tokens: int = sum(self.doc_lengths.values())
        self.avg_doc_length = total_tokens / self.total_docs if self.total_docs else 0.0

        logger.info(
            "Index built: %d unique terms across %d documents (avg len %.1f).",
            len(self.index),
            self.total_docs,
            self.avg_doc_length,
        )

    def save(self, filepath: str = DEFAULT_INDEX_PATH) -> None:
        """Serialise the entire index to a single JSON file.

        Args:
            filepath: Destination file path.
        """

        data: Dict[str, Any] = {
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_urls": self.doc_urls,
            "doc_lengths": self.doc_lengths,
            "doc_titles": self._doc_titles,
            "index": self._serialise_index(),
        }

        # Ensure parent directory exists.
        dirpath: str = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        size_mb: float = os.path.getsize(filepath) / (1024 * 1024)
        logger.info("Index saved to %s (%.2f MB).", filepath, size_mb)

    def load(self, filepath: str = DEFAULT_INDEX_PATH) -> None:
        """Deserialise the index from a previously saved JSON file.

        Args:
            filepath: Source file path.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)

        self.total_docs = int(data["total_docs"])
        self.avg_doc_length = float(data["avg_doc_length"])
        self.doc_urls = list(data["doc_urls"])
        self.doc_lengths = {k: int(v) for k, v in data["doc_lengths"].items()}
        self._doc_titles = dict(data.get("doc_titles", {}))
        self._deserialise_index(data["index"])

        logger.info(
            "Index loaded from %s: %d terms, %d docs.",
            filepath,
            len(self.index),
            self.total_docs,
        )

    def get_term_info(self, word: str) -> Optional[TermEntry]:
        """Retrieve the :class:`TermEntry` for a given word.

        The look-up is case-insensitive.

        Args:
            word: The word to look up.

        Returns:
            The corresponding :class:`TermEntry`, or ``None`` if the word is
            not in the index.
        """

        return self.index.get(word.lower())

    def get_doc_title(self, url: str) -> str:
        """Return the title of a document given its URL.

        Args:
            url: The document URL.

        Returns:
            The page title, or the URL itself if no title was recorded.
        """

        return self._doc_titles.get(url, url)

    def idf(self, term: str) -> float:
        """Compute the Inverse Document Frequency of *term*.

        Uses the smoothed logarithmic formula::

            idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))

        where *N* is the total document count and *df* is the document
        frequency of *term*.

        Time Complexity: O(1) - Dictionary lookup is constant time.
        Space Complexity: O(1) - No extra space allocated.

        Args:
            term: The normalised term.

        Returns:
            The IDF value.  Returns 0.0 if the term is absent from the index.
        """

        entry: Optional[TermEntry] = self.index.get(term)
        if entry is None or entry.document_frequency == 0:
            return 0.0
        n: int = self.total_docs
        df: int = entry.document_frequency
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # -- tokenisation -------------------------------------------------------

    @staticmethod
    def tokenise(text: str) -> List[str]:
        """Tokenise *text* into a list of lower-cased word tokens.

        Tokens are defined as contiguous runs of alphanumeric characters.
        The output preserves order and duplicates (positional information).

        Time Complexity: O(C) where C is the number of characters in the text.
        Space Complexity: O(T) where T is the number of resulting tokens.

        Args:
            text: Raw text string.

        Returns:
            A list of lower-cased tokens.
        """

        return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _build_document_text(page: PageData) -> str:
        """Concatenate all meaningful text fields from a page into one string.

        Quotes and author names are repeated to boost their weight during
        tokenisation.

        Args:
            page: A :class:`PageData` instance.

        Returns:
            A single text string representing the full document content.
        """

        parts: List[str] = []

        # Include the page title.
        if page.title:
            parts.append(page.title)

        # Include structured quote data (boosted).
        for quote in page.quotes:
            parts.append(quote.text)
            parts.append(quote.author)
            parts.extend(quote.tags)

        # Include author bio text.
        for author_name, bio in page.author_info.items():
            parts.append(author_name)
            parts.append(bio)

        # Include the raw visible text as a fallback / catch-all.
        if page.raw_text:
            parts.append(page.raw_text)

        return " ".join(parts)

    def _serialise_index(self) -> Dict[str, Any]:
        """Convert the in-memory index to a JSON-friendly dictionary."""

        serialised: Dict[str, Any] = {}
        for term, entry in self.index.items():
            postings_dict: Dict[str, Any] = {}
            for doc_url, posting in entry.postings.items():
                postings_dict[doc_url] = {
                    "tf": posting.term_frequency,
                    "pos": posting.positions,
                }
            serialised[term] = {
                "df": entry.document_frequency,
                "postings": postings_dict,
            }
        return serialised

    def _deserialise_index(self, raw: Dict[str, Any]) -> None:
        """Rebuild the in-memory index from a JSON-derived dictionary."""

        self.index.clear()
        for term, entry_data in raw.items():
            entry = TermEntry(document_frequency=int(entry_data["df"]))
            for doc_url, posting_data in entry_data["postings"].items():
                entry.postings[doc_url] = TermPosting(
                    doc_url=doc_url,
                    term_frequency=int(posting_data["tf"]),
                    positions=list(posting_data["pos"]),
                )
            self.index[term] = entry

    # -- dunder methods -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of unique terms in the index."""

        return len(self.index)

    def __contains__(self, term: str) -> bool:
        """Check whether a (case-insensitive) term exists in the index."""

        return term.lower() in self.index

    def __repr__(self) -> str:
        return (
            f"InvertedIndex(terms={len(self.index)}, "
            f"docs={self.total_docs}, "
            f"avg_len={self.avg_doc_length:.1f})"
        )
