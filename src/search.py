"""
search.py – TF-IDF / BM25 search engine over the inverted index.

This module implements an advanced ranking algorithm based on the Okapi BM25
variant of TF-IDF.  It supports single-word and multi-word queries and returns
results ranked by relevance.

AI Declaration:
    AI tools (GitHub Copilot / ChatGPT) were used to assist in implementing the
    BM25 scoring formula, handling edge cases for empty / special-character
    queries, and structuring result aggregation.  All logic was verified and
    adapted by the author.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from indexer import InvertedIndex, TermEntry, TermPosting, STOP_WORDS
except ImportError:  # pragma: no cover
    from src.indexer import InvertedIndex, TermEntry, TermPosting, STOP_WORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 tuning constants
# ---------------------------------------------------------------------------

BM25_K1: float = 1.5   # Term-frequency saturation parameter.
BM25_B: float = 0.75   # Length-normalisation parameter.

# Bonus multiplier when all query terms appear in a document.
_ALL_TERMS_BONUS: float = 1.5

# Bonus for adjacent (phrase-like) terms in the document.
_ADJACENCY_BONUS: float = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result with its relevance score and metadata.

    Attributes:
        url: The URL of the matching document.
        title: The page title.
        score: The computed relevance score (higher is better).
        matched_terms: Set of query terms that matched in this document.
        snippet: A short excerpt from the page highlighting the match context.
    """

    url: str
    title: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    snippet: str = ""


# ---------------------------------------------------------------------------
# Search engine class
# ---------------------------------------------------------------------------


class SearchEngine:
    """BM25-based search engine operating on an :class:`InvertedIndex`.

    The engine tokenises the query, computes BM25 scores for each term in each
    document, then aggregates per-document scores.  Additional bonuses are
    awarded for:

    - Documents that contain **all** query terms.
    - Documents where query terms appear **adjacent** (phrase match).

    Usage::

        engine = SearchEngine(index)
        results = engine.search("life love")
        for r in results:
            print(r.score, r.url)

    Args:
        index: A populated :class:`InvertedIndex`.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index: InvertedIndex = index

    # -- public API ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        include_stop_words: bool = False,
    ) -> List[SearchResult]:
        """Execute a search query and return ranked results.

        Args:
            query: The user's search query (single or multi-word).
            top_k: Maximum number of results to return.
            include_stop_words: If ``True``, stop-words are not removed from
                the query.  By default, common stop-words are stripped.

        Returns:
            A list of :class:`SearchResult` objects sorted by descending score.
            Returns an empty list for empty or unparseable queries.
        """

        # --- Input validation / sanitisation --------------------------------
        if not query or not query.strip():
            logger.info("Empty query received.")
            return []

        tokens: List[str] = self._index.tokenise(query)

        if not include_stop_words:
            tokens = [t for t in tokens if t not in STOP_WORDS]

        if not tokens:
            logger.info("Query '%s' reduced to empty after stop-word removal.", query)
            # Fall back to raw tokens so the user still gets *something*.
            tokens = self._index.tokenise(query)
            if not tokens:
                return []

        # Deduplicate while preserving order (for adjacency checking).
        seen = set()
        unique_tokens: List[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        logger.debug("Search tokens: %s", unique_tokens)

        # --- BM25 scoring --------------------------------------------------
        doc_scores: Dict[str, float] = {}
        doc_matched: Dict[str, set] = {}

        for term in unique_tokens:
            entry: Optional[TermEntry] = self._index.get_term_info(term)
            if entry is None:
                continue

            idf_val: float = self._index.idf(term)

            for doc_url, posting in entry.postings.items():
                tf: int = posting.term_frequency
                doc_len: int = self._index.doc_lengths.get(doc_url, 1)
                avg_dl: float = self._index.avg_doc_length or 1.0

                # BM25 term score.
                numerator: float = tf * (BM25_K1 + 1.0)
                denominator: float = tf + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * (doc_len / avg_dl)
                )
                bm25_score: float = idf_val * (numerator / denominator)

                doc_scores[doc_url] = doc_scores.get(doc_url, 0.0) + bm25_score

                if doc_url not in doc_matched:
                    doc_matched[doc_url] = set()
                doc_matched[doc_url].add(term)

        if not doc_scores:
            logger.info("No documents matched query '%s'.", query)
            return []

        # --- Bonus: all-terms coverage ------------------------------------
        num_query_terms: int = len(unique_tokens)
        if num_query_terms > 1:
            for doc_url, matched in doc_matched.items():
                if len(matched) == num_query_terms:
                    doc_scores[doc_url] *= _ALL_TERMS_BONUS

        # --- Bonus: adjacency / phrase matching ----------------------------
        if num_query_terms > 1:
            self._apply_adjacency_bonus(unique_tokens, doc_scores, doc_matched)

        # --- Sort & build results ------------------------------------------
        ranked: List[Tuple[str, float]] = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: List[SearchResult] = []
        for doc_url, score in ranked:
            title: str = self._index.get_doc_title(doc_url)
            matched_list: List[str] = sorted(doc_matched.get(doc_url, set()))
            results.append(
                SearchResult(
                    url=doc_url,
                    title=title,
                    score=round(score, 4),
                    matched_terms=matched_list,
                )
            )

        return results

    # -- private helpers ----------------------------------------------------

    def _apply_adjacency_bonus(
        self,
        tokens: List[str],
        doc_scores: Dict[str, float],
        doc_matched: Dict[str, set],
    ) -> None:
        """Apply a score bonus for documents where query terms appear adjacent.

        Two terms are considered *adjacent* if one's positional offset is
        exactly one more than the other's, respecting query order.

        Args:
            tokens: Ordered list of unique query tokens.
            doc_scores: Mutable mapping of doc URL → accumulated score.
            doc_matched: Mapping of doc URL → set of matched terms.
        """

        for i in range(len(tokens) - 1):
            term_a: str = tokens[i]
            term_b: str = tokens[i + 1]

            entry_a: Optional[TermEntry] = self._index.get_term_info(term_a)
            entry_b: Optional[TermEntry] = self._index.get_term_info(term_b)

            if entry_a is None or entry_b is None:
                continue

            # Find documents that contain both terms.
            common_docs: set = set(entry_a.postings.keys()) & set(
                entry_b.postings.keys()
            )

            for doc_url in common_docs:
                positions_a: List[int] = entry_a.postings[doc_url].positions
                positions_b: List[int] = entry_b.postings[doc_url].positions

                if self._has_adjacent_positions(positions_a, positions_b):
                    doc_scores[doc_url] = doc_scores.get(doc_url, 0.0) * _ADJACENCY_BONUS
                    logger.debug(
                        "Adjacency bonus for '%s %s' in %s",
                        term_a,
                        term_b,
                        doc_url,
                    )

    @staticmethod
    def _has_adjacent_positions(
        positions_a: List[int], positions_b: List[int]
    ) -> bool:
        """Check if any position in *positions_a* is immediately before a
        position in *positions_b*.

        Uses a set-based O(n + m) approach.

        Args:
            positions_a: Sorted positions of the first term.
            positions_b: Sorted positions of the second term.

        Returns:
            ``True`` if an adjacent pair exists, ``False`` otherwise.
        """

        set_b = set(positions_b)
        return any((pos + 1) in set_b for pos in positions_a)

    # -- dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        return f"SearchEngine(index={self._index!r})"
