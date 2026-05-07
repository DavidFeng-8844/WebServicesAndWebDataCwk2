"""
tests/test_search.py

Unit tests for the SearchEngine module.
"""

import pytest

from src.search import SearchEngine
from src.indexer import InvertedIndex
from src.crawler import PageData, QuoteData


@pytest.fixture
def populated_index():
    pages = [
        PageData(
            url="http://test.com/1",
            title="First Page",
            quotes=[QuoteData(text="apple banana cherry", author="Author A")]
        ),
        PageData(
            url="http://test.com/2",
            title="Second Page",
            quotes=[QuoteData(text="banana cherry date date", author="Author B")]
        ),
        PageData(
            url="http://test.com/3",
            title="Third Page",
            quotes=[QuoteData(text="apple date elderberry", author="Author C")]
        )
    ]
    index = InvertedIndex()
    index.build(pages)
    return index


def test_single_word_query(populated_index):
    """Test ranking logic for a single-word query."""
    engine = SearchEngine(populated_index)
    results = engine.search("date")
    
    # 'date' appears twice in doc 2, once in doc 3. 
    # TF-IDF / BM25 should rank doc 2 higher than doc 3.
    assert len(results) == 2
    assert results[0].url == "http://test.com/2"
    assert results[1].url == "http://test.com/3"


def test_multi_word_query(populated_index):
    """Test ranking logic for a multi-word query (including adjacency/bonus logic)."""
    engine = SearchEngine(populated_index)
    results = engine.search("apple banana")
    
    # doc 1 has both 'apple' and 'banana' adjacent! -> Should get adjacency and all-terms bonuses.
    # doc 2 has 'banana'
    # doc 3 has 'apple'
    assert len(results) == 3
    assert results[0].url == "http://test.com/1"
    assert "apple" in results[0].matched_terms
    assert "banana" in results[0].matched_terms


def test_tfidf_logic_mathematical(populated_index):
    """Test the mathematical/logical foundations of the TF-IDF ranking logic."""
    # Check that rare words get a higher IDF score than common words.
    # 'elderberry' is in 1 doc. 'apple' is in 2 docs.
    idf_elderberry = populated_index.idf("elderberry")
    idf_apple = populated_index.idf("apple")
    
    assert idf_elderberry > idf_apple

    engine = SearchEngine(populated_index)
    results_elderberry = engine.search("elderberry")
    results_apple = engine.search("apple")
    
    # Since elderberry is much rarer, the top doc matching it should have a higher score
    # than the top doc matching apple (assuming comparable lengths/frequencies).
    assert results_elderberry[0].score > results_apple[0].score


def test_edge_cases(populated_index):
    """Test edge cases: empty queries, words not in the index, and special characters."""
    engine = SearchEngine(populated_index)
    
    # Empty query
    assert engine.search("") == []
    assert engine.search("   ") == []
    
    # Word not in index
    assert engine.search("xylophone") == []
    
    # Special characters (should be tokenised to nothing or valid parts)
    assert engine.search("!@#") == []
    
    # "apple!" should be tokenised to "apple"
    results = engine.search("apple!")
    assert len(results) == 2
    assert "apple" in results[0].matched_terms
