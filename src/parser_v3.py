"""
search_engine.py

Current capabilities (foundation for next phases):
- Text preprocessing (lowercase, punctuation removal, stop-word filtering)
- Positional inverted index: index[term][doc] = [positions]
- IDF computation: idf[term] = log(N / DF(term))
- Ranked keyword search (TF–IDF) using positional TF = len(positions)
- Exact phrase search using adjacency checks when query is wrapped in quotes

Next things you can build on top of this:
- Combine AND + TF–IDF (require all terms, then rank)
- BM25 ranking
- Fielded search (title vs body)
- Persistence (save/load index)
- Top-K optimization
"""

import os
import math
import string
from typing import Dict, List, Tuple, Literal


# -------------------------
# Config
# -------------------------
STOP_WORDS = {
    "the", "is", "at", "on", "and", "a", "an", "of", "to", "in"
}


# -------------------------
# Preprocessing
# -------------------------
def tokenize(text: str) -> List[str]:
    """Normalize + tokenize document text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in STOP_WORDS]


def tokenize_query(query: str) -> List[str]:
    """Normalize + tokenize query text (same pipeline as docs)."""
    query = query.lower()
    query = query.translate(str.maketrans("", "", string.punctuation))
    return [w for w in query.split() if w and w not in STOP_WORDS]


def parse_query(raw: str) -> Tuple[Literal["PHRASE", "KEYWORD"], str]:
    """
    Detect phrase queries by quotes:
      "machine learning" -> ("PHRASE", machine learning)
      machine learning   -> ("KEYWORD", machine learning)
    """
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return "PHRASE", raw[1:-1]
    return "KEYWORD", raw


# -------------------------
# Indexing (Positional Index)
# -------------------------
PositionalIndex = Dict[str, Dict[str, List[int]]]
IDFMap = Dict[str, float]


def list_text_docs(data_dir: str) -> List[str]:
    """Return .txt filenames (not full paths) in a directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])


def build_positional_index(data_dir: str) -> Tuple[PositionalIndex, int]:
    """
    Build positional index:
      index[term][doc] = [pos1, pos2, ...]
    Returns (index, N) where N is number of documents indexed.
    """
    index: PositionalIndex = {}
    docs = list_text_docs(data_dir)

    for filename in docs:
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenize(text)

        for pos, term in enumerate(tokens):
            if term not in index:
                index[term] = {}
            if filename not in index[term]:
                index[term][filename] = []
            index[term][filename].append(pos)

    return index, len(docs)


def compute_idf(index: PositionalIndex, total_docs: int) -> IDFMap:
    """
    Compute IDF:
      DF(term) = number of docs containing term = len(index[term])
      IDF(term) = log(N / DF(term))
    """
    if total_docs <= 0:
        return {}

    idf: IDFMap = {}
    for term, postings in index.items():
        df = len(postings)
        if df > 0:
            idf[term] = math.log(total_docs / df)
    return idf


# -------------------------
# Retrieval
# -------------------------
def ranked_keyword_query_tfidf(index: PositionalIndex, idf: IDFMap, query: str) -> List[Tuple[str, float]]:
    """
    Ranked retrieval using TF–IDF.
    TF(term, doc) = len(index[term][doc]) because index stores positions.
    Score(doc) = sum_{term in query} TF(term, doc) * IDF(term)
    """
    terms = tokenize_query(query)
    if not terms:
        return []

    scores: Dict[str, float] = {}

    for term in terms:
        if term not in index:
            continue

        postings = index[term]  # {doc: [positions]}
        w_idf = idf.get(term, 0.0)

        for doc, positions in postings.items():
            tf = len(positions)
            scores[doc] = scores.get(doc, 0.0) + tf * w_idf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def phrase_query(index: PositionalIndex, phrase: str) -> List[str]:
    """
    Exact phrase matching using positional index.

    For phrase terms t1 t2 ... tk, in a given doc we need:
      p in positions(t1) and (p+1) in positions(t2) and ... (p+k-1) in positions(tk)

    Returns list of matching docs (unranked).
    """
    terms = tokenize_query(phrase)
    if not terms:
        return []

    if terms[0] not in index:
        return []

    # candidate_docs maps doc -> positions where the phrase is still "alive"
    candidate_docs: Dict[str, List[int]] = {doc: positions[:] for doc, positions in index[terms[0]].items()}

    for i in range(1, len(terms)):
        term = terms[i]
        if term not in index:
            return []

        next_docs = index[term]
        new_candidates: Dict[str, List[int]] = {}

        for doc, prev_positions in candidate_docs.items():
            if doc not in next_docs:
                continue

            next_positions = next_docs[doc]
            next_set = set(next_positions)

            # Keep only positions where this term appears at prev_position + 1
            matches = [p + 1 for p in prev_positions if (p + 1) in next_set]

            if matches:
                new_candidates[doc] = matches

        candidate_docs = new_candidates
        if not candidate_docs:
            return []

    return sorted(candidate_docs.keys())


# -------------------------
# CLI
# -------------------------
def main():
    data_dir = "data"  # change if your folder is different

    print("Building positional index...")
    index, N = build_positional_index(data_dir)
    idf = compute_idf(index, N)

    print(f"Indexed {N} documents. Unique terms: {len(index)}")
    print('Type a query. Use quotes for phrases, e.g. "machine learning". Type exit to quit.\n')

    while True:
        raw = input("Search (or 'exit'): ").strip()
        if raw.lower() == "exit":
            break
        if not raw:
            continue

        mode, q = parse_query(raw)

        if mode == "PHRASE":
            docs = phrase_query(index, q)
            if not docs:
                print("No results found.\n")
            else:
                print("Phrase matches:")
                for doc in docs:
                    print(f"- {doc}")
                print()
        else:
            results = ranked_keyword_query_tfidf(index, idf, q)
            if not results:
                print("No results found.\n")
            else:
                print("Ranked results (TF–IDF):")
                for doc, score in results[:10]:
                    print(f"- {doc}: {score:.4f}")
                print()


if __name__ == "__main__":
    main()