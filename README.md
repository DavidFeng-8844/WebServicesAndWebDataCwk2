# Search Engine Tool

A robust, modular, and publication-quality web crawler and search engine designed to index and query quotes from `quotes.toscrape.com`.

## Overview

**Demo Video** (click the image)
[![Watch the video](https://img.youtube.com/vi/ktcNG5CKBsA/maxresdefault.jpg) ](https://youtu.be/ktcNG5CKBsA)

This project implements a complete, publication-quality search engine pipeline comprising three core modules:
1. **Crawler**: A polite, BFS-based web scraper that respects rate limits (6-second delay), supports proxy configurations, and uses robust custom exceptions (`NetworkFetchError`, `PolitenessWindowViolation`) for error recovery.
2. **Indexer**: An inverted index builder that processes raw text into a normalized, case-insensitive index storing term frequency (TF), document frequency (DF), and positional data. Core operations are fully documented with Time/Space Big-O complexity analysis.
3. **Search Engine**: A retrieval system utilizing Okapi BM25 and TF-IDF mathematical models for document ranking, enhanced with adjacency bonuses for phrase matching and Levenshtein-based query suggestions for typos.

## Advanced Features & Outstanding Polish
- **Query Suggestions**: Implements Levenshtein distance typo detection to recommend "Did you mean: X?" if a search yields no exact matches.
- **Complexity Analysis**: Includes comprehensive Big-O Time and Space complexity comments for indexing and search operations.
- **Defensive Programming**: Employs robust error handling, structured exceptions, and network backoffs.
- **CI/CD Pipeline**: Automated GitHub Actions testing pipeline ensures that `pytest` runs successfully on every push or pull request.

## Architecture

```text
src/
├── crawler.py   # Handles HTTP requests, HTML parsing, and polite crawling
├── indexer.py   # Tokenises text, builds the inverted index, handles serialization
├── search.py    # Implements the BM25 / TF-IDF ranking algorithms
└── main.py      # Provides an interactive CLI REPL
```

## Setup and Installation

**1. Clone the repository and navigate to the project root:**
```bash
git clone <repository_url>
cd CWK2
```

**2. Create a virtual environment (Optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

The application provides an interactive Command Line Interface (CLI). **Always run the application from the project root.**

Start the CLI:
```bash
python -m src.main
```

### CLI Commands

* **`build`**: Crawls the target website, builds the inverted index, and saves it to `data/search_index.json`.
  > *Note: If you require a proxy (e.g., Clash), set `HTTP_PROXY` and `HTTPS_PROXY` environment variables before running this command.*
* **`load`**: Loads the previously saved index from the disk into memory.
* **`print <word>`**: Displays detailed index statistics (DF, IDF, and Term Positions) for a specific word.
* **`find <query>`**: Searches the index for the given query using TF-IDF ranking and returns the top matching URLs.

### Example Session
```text
search> build
... [Crawls 213 pages and saves index] ...
search> find "life love"
  1. 2.2643 https://quotes.toscrape.com/tag/love/page/1
  2. 2.2643 https://quotes.toscrape.com/tag/love
  ...
```

## Running the Tests

The project includes a professional-grade test suite with high coverage using `pytest` and `unittest.mock` to simulate network and file I/O operations.

To run the tests from the project root:
```bash
pytest tests/
```

*(Note: Do not run `python tests/test_crawler.py` directly, as it will cause a `ModuleNotFoundError` due to Python's module resolution. Always use `pytest` from the root directory).*

## Algorithm Details
- **Tokenisation**: Case-insensitive, removes punctuation, keeps alphanumeric terms.
- **Ranking**: Uses an adapted **Okapi BM25** variant of TF-IDF. It normalises for document length to prevent long documents from unfairly outranking shorter ones, and incorporates term-adjacency bonuses for multi-word phrase matches.
