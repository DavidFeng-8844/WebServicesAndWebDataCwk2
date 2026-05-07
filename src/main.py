"""
main.py – Command-Line Interface for the Search Engine Tool.

Provides four commands:
    build   – Crawl the target website, build the inverted index, and save it.
    load    – Load a previously saved index from the file system.
    print   – Display inverted-index statistics for a specific word.
    find    – Search the index for a query phrase and display ranked results.

Usage::

    python -m src.main build
    python -m src.main load
    python -m src.main print life
    python -m src.main find "life love"

AI Declaration:
    AI tools (GitHub Copilot / ChatGPT) were used to assist in structuring the
    CLI argument parsing, formatting output tables, and reviewing the overall
    user-interaction flow.  All logic was verified and adapted by the author.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import List, Optional

# Reconfigure stdout/stderr to UTF-8 on Windows to handle box-drawing
# characters and emoji without UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

try:
    from crawler import WebCrawler, PageData
    from indexer import InvertedIndex, TermEntry, DEFAULT_INDEX_PATH
    from search import SearchEngine, SearchResult
except ImportError:  # pragma: no cover
    from src.crawler import WebCrawler, PageData
    from src.indexer import InvertedIndex, TermEntry, DEFAULT_INDEX_PATH
    from src.search import SearchEngine, SearchResult

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USAGE: str = """
╔══════════════════════════════════════════════════════════════════╗
║                  Search Engine Tool – CLI                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands:                                                      ║
║    build              Crawl, index, and save to disk.           ║
║    load               Load a previously saved index.            ║
║    print <word>       Show index statistics for <word>.         ║
║    find  <query>      Search the index for <query>.             ║
╚══════════════════════════════════════════════════════════════════╝
"""

SEPARATOR: str = "─" * 64


# ---------------------------------------------------------------------------
# CLI handler functions
# ---------------------------------------------------------------------------


def handle_build(index: InvertedIndex) -> InvertedIndex:
    """Crawl the target website, build the index, and persist it to disk.

    Args:
        index: An :class:`InvertedIndex` instance (will be populated in-place).

    Returns:
        The populated :class:`InvertedIndex`.
    """

    print(f"\n{SEPARATOR}")
    print("  ⏳  Phase 1 – Crawling https://quotes.toscrape.com/ …")
    print(f"{SEPARATOR}\n")

    crawler = WebCrawler()
    pages: List[PageData] = crawler.crawl()

    if not pages:
        print("  ⚠  No pages were crawled. Aborting build.")
        return index

    print(f"\n  ✓  Crawled {len(pages)} pages.\n")
    print(f"{SEPARATOR}")
    print("  ⏳  Phase 2 – Building inverted index …")
    print(f"{SEPARATOR}\n")

    index.build(pages)

    print(f"  ✓  Index built: {len(index)} unique terms.\n")
    print(f"{SEPARATOR}")
    print(f"  ⏳  Phase 3 – Saving index to '{DEFAULT_INDEX_PATH}' …")
    print(f"{SEPARATOR}\n")

    index.save(DEFAULT_INDEX_PATH)

    print(f"  ✓  Index saved successfully.\n")
    return index


def handle_load(index: InvertedIndex) -> InvertedIndex:
    """Load a previously saved index from the file system.

    Args:
        index: An :class:`InvertedIndex` instance (will be loaded in-place).

    Returns:
        The loaded :class:`InvertedIndex`.
    """

    print(f"\n{SEPARATOR}")
    print(f"  ⏳  Loading index from '{DEFAULT_INDEX_PATH}' …")
    print(f"{SEPARATOR}\n")

    try:
        index.load(DEFAULT_INDEX_PATH)
        print(f"  ✓  Index loaded: {len(index)} unique terms, "
              f"{index.total_docs} documents.\n")
    except FileNotFoundError:
        print(f"  ✗  Index file '{DEFAULT_INDEX_PATH}' not found.")
        print("     Run 'build' first to create the index.\n")
    except Exception as exc:
        print(f"  ✗  Failed to load index: {exc}\n")
        logger.exception("Index load failure")

    return index


def handle_print(index: InvertedIndex, word: str) -> None:
    """Print the inverted-index statistics for a specific word.

    Args:
        index: A populated :class:`InvertedIndex`.
        word: The word to look up.
    """

    if index.total_docs == 0:
        print("\n  ⚠  No index loaded. Run 'build' or 'load' first.\n")
        return

    print(f"\n{SEPARATOR}")
    print(f"  Index entry for: '{word}'")
    print(f"{SEPARATOR}\n")

    entry: Optional[TermEntry] = index.get_term_info(word)

    if entry is None:
        print(f"  ✗  '{word}' not found in the index.\n")
        return

    print(f"  Term (normalised)     : {word.lower()}")
    print(f"  Document frequency    : {entry.document_frequency}")
    print(f"  IDF score             : {index.idf(word.lower()):.4f}")
    print(f"  Total corpus docs     : {index.total_docs}")
    print()

    # Per-document breakdown.
    print(f"  {'Document URL':<50}  {'TF':>4}  Positions")
    print(f"  {'─' * 50}  {'─' * 4}  {'─' * 20}")

    for doc_url, posting in sorted(
        entry.postings.items(), key=lambda x: x[1].term_frequency, reverse=True
    ):
        positions_preview: str = str(posting.positions[:10])
        if len(posting.positions) > 10:
            positions_preview = positions_preview[:-1] + ", …]"
        print(f"  {doc_url:<50}  {posting.term_frequency:>4}  {positions_preview}")

    print()


def handle_find(index: InvertedIndex, query: str) -> None:
    """Search the index for a query phrase and display ranked results.

    Args:
        index: A populated :class:`InvertedIndex`.
        query: The search query (single or multi-word).
    """

    if index.total_docs == 0:
        print("\n  ⚠  No index loaded. Run 'build' or 'load' first.\n")
        return

    print(f"\n{SEPARATOR}")
    print(f"  Searching for: '{query}'")
    print(f"{SEPARATOR}\n")

    engine = SearchEngine(index)
    results: List[SearchResult] = engine.search(query)

    if not results:
        print(f"  ✗  No results found for '{query}'.\n")
        return

    print(f"  Found {len(results)} result(s):\n")
    print(f"  {'#':<4} {'Score':>8}  {'Matched Terms':<30}  URL")
    print(f"  {'─' * 4} {'─' * 8}  {'─' * 30}  {'─' * 40}")

    for rank, result in enumerate(results, start=1):
        terms_str: str = ", ".join(result.matched_terms)
        print(f"  {rank:<4} {result.score:>8.4f}  {terms_str:<30}  {result.url}")

    print()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def run_interactive(index: InvertedIndex) -> None:
    """Run an interactive REPL loop for the search engine CLI.

    Args:
        index: An :class:`InvertedIndex` (may be empty initially).
    """

    print(USAGE)

    while True:
        try:
            raw_input: str = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye! 👋\n")
            break

        if not raw_input:
            continue

        parts: List[str] = raw_input.split(maxsplit=1)
        command: str = parts[0].lower()

        if command in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break

        elif command == "build":
            index = handle_build(index)

        elif command == "load":
            index = handle_load(index)

        elif command == "print":
            if len(parts) < 2:
                print("  Usage: print <word>\n")
            else:
                handle_print(index, parts[1])

        elif command == "find":
            if len(parts) < 2:
                print("  Usage: find <query>\n")
            else:
                handle_find(index, parts[1])

        elif command == "help":
            print(USAGE)

        else:
            print(f"  Unknown command: '{command}'. Type 'help' for usage.\n")


# ---------------------------------------------------------------------------
# CLI entry point (non-interactive mode)
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point supporting both single-command and interactive modes.

    If command-line arguments are provided, the corresponding command is
    executed directly and the program exits.  Otherwise, the interactive
    REPL is started.
    """

    index = InvertedIndex()

    args: List[str] = sys.argv[1:]

    if not args:
        # No arguments → launch interactive mode.
        run_interactive(index)
        return

    command: str = args[0].lower()

    if command == "build":
        handle_build(index)

    elif command == "load":
        index = handle_load(index)
        # After loading, drop into interactive mode so the user can query.
        run_interactive(index)

    elif command == "print":
        if len(args) < 2:
            print("  Usage: python main.py print <word>")
            sys.exit(1)
        index = handle_load(index)
        handle_print(index, args[1])

    elif command == "find":
        if len(args) < 2:
            print("  Usage: python main.py find <query>")
            sys.exit(1)
        index = handle_load(index)
        handle_find(index, " ".join(args[1:]))

    elif command in ("help", "--help", "-h"):
        print(USAGE)

    else:
        print(f"  Unknown command: '{command}'.")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
