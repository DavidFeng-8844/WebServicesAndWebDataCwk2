"""
tests/test_indexer.py

Unit tests for the InvertedIndex module.
"""

import json
import pytest
from unittest.mock import patch, mock_open

from src.indexer import InvertedIndex
from src.crawler import PageData, QuoteData


@pytest.fixture
def sample_pages():
    return [
        PageData(
            url="http://test.com/1",
            title="Test Page",
            quotes=[QuoteData(text="Hello world! HELLO", author="Author 1", tags=["tag1"])]
        ),
        PageData(
            url="http://test.com/2",
            title="Another Page",
            quotes=[QuoteData(text="world peace", author="Author 2", tags=["tag2"])]
        )
    ]


def test_index_creation(sample_pages):
    """Test the creation of the inverted index from sample pages."""
    indexer = InvertedIndex()
    indexer.build(sample_pages)

    assert indexer.total_docs == 2
    assert "hello" in indexer.index
    assert "world" in indexer.index
    
    hello_entry = indexer.get_term_info("hello")
    assert hello_entry.document_frequency == 1
    assert "http://test.com/1" in hello_entry.postings
    
    # "Hello" appears twice in the quote text, but is also case-normalized
    assert hello_entry.postings["http://test.com/1"].term_frequency == 2


def test_text_normalization():
    """Test text normalization (case-insensitivity and punctuation removal)."""
    indexer = InvertedIndex()
    text = "Hello, World! It's a test-case."
    tokens = indexer.tokenise(text)
    
    assert tokens == ["hello", "world", "it", "s", "a", "test", "case"]


def test_save_and_load_index(sample_pages):
    """Test saving to and loading from search_index.json (mocking the file system)."""
    indexer = InvertedIndex()
    indexer.build(sample_pages)

    mock_file_content = ""

    def mock_write(content):
        nonlocal mock_file_content
        mock_file_content += content

    # Test saving
    with patch("builtins.open", mock_open()) as mocked_file, patch("os.makedirs"), patch("os.path.getsize", return_value=1024):
        mocked_file.return_value.write.side_effect = mock_write
        indexer.save("mock_index.json")

    assert len(mock_file_content) > 0
    saved_data = json.loads(mock_file_content)
    assert saved_data["total_docs"] == 2
    assert "index" in saved_data

    # Test loading
    new_indexer = InvertedIndex()
    with patch("builtins.open", mock_open(read_data=mock_file_content)), patch("os.path.isfile", return_value=True):
        new_indexer.load("mock_index.json")

    assert new_indexer.total_docs == 2
    assert "world" in new_indexer.index
