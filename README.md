# Inverted Index Search Engine (Python)

## Overview

A Python-based search engine that implements core **Information Retrieval** concepts. The system builds an **inverted index** over a text corpus and supports ranked search using **TF–IDF**. The focus is on algorithmic correctness, efficiency, and clear system design.

---

## Key Features

* Text preprocessing (lowercasing, punctuation removal, stop-word filtering)
* Inverted index construction
* Multi-word query support
* TF–IDF–based relevance ranking
* Interactive command-line interface

---

## Project Structure

```text
project-root/
├── data/            # .txt document corpus
├── search_engine.py # Core implementation
└── README.md
```

---

## Core Design

### Inverted Index

Maps each term to the documents it appears in along with term frequencies:

```text
term → { document : frequency }
```

Enables efficient lookup without scanning all documents.

### Ranking (TF–IDF)

Results are ranked using:

```text
TF–IDF(term, doc) = TF(term, doc) × log(N / DF(term))
```

Scores are summed across query terms at query time.

### Index-Time vs Query-Time

* **Index-time:** preprocessing, index construction, IDF computation
* **Query-time:** lookup, score aggregation, sorting

This separation improves performance and scalability.

---

## Usage

```bash
python search_engine.py
```

Example:

```text
Search (or 'exit'): machine learning
doc1.txt: 1.7321
doc3.txt: 0.8452
```

---

## Limitations

* No phrase or positional search
* No persistent on-disk index
* No web interface

These are intentional to keep the project focused on core IR principles.

---

## Possible Extensions

* Boolean AND/OR queries
* BM25 ranking
* Positional indexes
* Index compression

---

## Author

Tushaar Sood

---

## License

Educational use only.
