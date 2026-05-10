# AI Declaration and Evaluation

*Student Name: [Yujie Feng]*  
*Student ID: [201691089]*  
*Coursework: Search Engine Tool (CWK2)*

## 1. Overview of GenAI Usage
Throughout the development of this project, Generative AI (e.g., ChatGPT, Claude) was utilized as a pair-programming assistant. Its primary roles included scaffolding boilerplate code, suggesting algorithmic optimizations for the search ranking, and generating unit test frameworks.

## 2. Specific Examples of AI Assistance

### 2.1 Where AI Helped
**Example 1: Politeness Window Implementation**
* **Context**: I needed to implement a strict 6-second politeness delay for the crawler.
* **AI Contribution**: The AI suggested using a timestamp-based approach (`time.time() - last_request_time`) rather than a hardcoded `time.sleep(6)` after every request.
* **Evaluation**: This was a superior approach. A hard sleep always waits 6 seconds, whereas the timestamp approach calculates the remaining time needed to hit exactly 6 seconds. This factors in network latency and HTML parsing time, leading to a much more efficient overall crawl without violating the server's politeness constraint.

**Example 2: BM25 / TF-IDF Mathematical Modeling**
* **Context**: Implementing the document ranking logic in `search.py`.
* **AI Contribution**: The AI provided the standard formula for TF-IDF and suggested an Okapi BM25 approach to penalize excessively long documents.
* **Evaluation**: While the core formula was correct, the AI's initial code did not handle the zero-division edge case well if a term was present in every single document (resulting in an IDF of 0). I had to manually implement smoothing (`log(N / (df + 1)) + 1`) to ensure stable ranking mathematics and prevent division-by-zero crashes.

### 2.2 Where AI Hindered (Challenges & Refactoring)
**Example 1: Inefficient Data Structures**
* **Context**: Building the Inverted Index in `indexer.py`.
* **AI Suggestion**: The AI initially suggested using a `list` of dictionaries to store postings.
* **My Refactoring**: I realized this would require $O(N)$ lookup time for every term search. I rejected the AI's code and refactored the index to use a nested Python `dict` structure (`self.index[term][url]`), achieving $O(1)$ constant time lookups. This critical thinking step significantly improved the application's performance.

**Example 2: Testing and Mocking**
* **Context**: Writing unit tests for the crawler.
* **AI Suggestion**: The AI wrote tests that actually made live HTTP requests to `quotes.toscrape.com`.
* **My Refactoring**: This is bad practice for unit tests as it slows down the testing pipeline and violates the politeness rule during testing. I had to manually research and implement `unittest.mock.patch` to intercept `requests.Session.get` and inject mocked HTML strings instead.

## 3. Impact on the Learning Process
Using GenAI shifted my learning focus from "syntax memorization" to "architectural design and code review." Instead of spending hours debugging missing parentheses or standard library imports, I spent my time evaluating the *time complexity* and *robustness* of the code.

However, I noticed a risk of "automation bias" — blindly trusting the AI's first output. I had to actively force myself to read the AI's generated code line-by-line. The exercise of finding flaws in the AI's logic (like the inefficient list lookup mentioned above) actually deepened my understanding of Python data structures more than writing it from scratch might have.

## 4. Conclusion
GenAI is a powerful productivity multiplier, but it cannot replace software engineering fundamentals. It acts as an eager, junior developer: it can write code quickly, but the code often lacks context, efficiency, and robust error handling. The responsibility of designing the system architecture and ensuring publication-quality code ultimately remained on my shoulders.
