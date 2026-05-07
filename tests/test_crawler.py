"""
tests/test_crawler.py

Unit tests for the WebCrawler module.
"""

import pytest
from unittest.mock import MagicMock, patch
import requests

from src.crawler import WebCrawler


@pytest.fixture
def mock_html():
    return """
    <html>
        <head><title>Test Title</title></head>
        <body>
            <div class="quote">
                <span class="text">“Test quote 1”</span>
                <small class="author">Author 1</small>
                <a class="tag">tag1</a>
            </div>
            <a href="http://test.com/page/2/">Next</a>
        </body>
    </html>
    """


def test_successful_page_fetch(mock_html):
    """Test that the crawler successfully fetches and parses a page."""
    with patch("requests.Session.get") as mock_get, patch("time.sleep"):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html
        mock_get.return_value = mock_response

        crawler = WebCrawler(base_url="http://test.com", politeness_delay=0)
        crawler._queue = ["http://test.com"]  # Force a single crawl to avoid infinite loop if links parsed
        pages = crawler.crawl()

        assert len(pages) > 0
        assert pages[0].title == "Test Title"
        assert len(pages[0].quotes) == 1
        assert pages[0].quotes[0].text == "Test quote 1"
        assert pages[0].quotes[0].author == "Author 1"
        assert pages[0].quotes[0].tags == ["tag1"]


def test_network_error_handling_and_retries():
    """Test that the crawler handles network errors and retries appropriately."""
    with patch("requests.Session.get") as mock_get, patch("time.sleep") as mock_sleep:
        # First 2 attempts fail, 3rd succeeds
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.text = "<html><title>Success</title></html>"
        
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection lost"),
            requests.exceptions.Timeout("Timeout"),
            mock_response_success
        ]

        crawler = WebCrawler(base_url="http://test.com", politeness_delay=0)
        html = crawler._fetch("http://test.com")

        assert html is not None
        assert "Success" in html
        assert mock_get.call_count == 3
        # Should have backed off twice
        assert mock_sleep.call_count >= 2


def test_politeness_delay():
    """Test that the 6-second politeness delay logic is triggered between requests."""
    with patch("requests.Session.get") as mock_get, patch("time.sleep") as mock_sleep:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html></html>"
        mock_get.return_value = mock_response

        with patch("time.time") as mock_time:
            # First fetch happens at time 100.1, second fetch attempt at 102.1
            # Elapsed time is 2.0s, so it should sleep for 4.0s.
            mock_time.side_effect = [102.1, 102.2]
            
            crawler = WebCrawler(base_url="http://test.com", politeness_delay=6.0)
            crawler._last_request_time = 100.1  
            
            crawler._fetch("http://test.com/2")
            
            mock_sleep.assert_called_once_with(pytest.approx(4.0))
