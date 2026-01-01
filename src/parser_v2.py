import re
import os
import string
import math

STOP_WORDS = {
    "the", "is", "at", "on", "and", "a", "an", "of", "to", "in"
}

def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', '' + string.punctuation))
    tokens = [w for w in text.split() if w and w not in STOP_WORDS]
    return tokens


def compute_idf(index, total_docs):
    idf = {}

    for term, postings in index.items():
        df = len(postings)
        idf[term] = math.log(total_docs / df)

    return idf


def build_inverted_index(docs_path):

    index = {}

    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_path, filename), "r", encoding = "utf-8") as f:
                text = f.read()
                tokens = tokenize(text)

                for token in tokens:
                    if token not in index:
                        index[token] = {}
                    if filename not in index[token]:
                        index[token][filename] = 0
                    index[token][filename] += 1
    return index


def query_index(index, query):
    query = query.lower().strip()
    if query in STOP_WORDS:
        return []
    if query not in index:
        return []
    results = index[query]

    ranked = sorted(
        results.items(), 
        key=lambda x: x[1], 
        reverse=True
    )

    return ranked
    
def query_tokenizer(query):
    query = query.lower()
    query = query.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in query.split() if w and w not in STOP_WORDS]
    return tokens

def multi_word_query(query, index):
    terms = query_tokenizer(query)

    if not terms:
        return []
    
    if terms[0] not in index:
        return []
    

    common_docs = index[terms[0]].copy()

    for term in terms[1:]:
        if term not in index:
            return []
        
        term_docs = index[term]

        common_docs = {
            doc: common_docs[doc] + term_docs[doc]
            for doc in common_docs
            if doc in term_docs
        }

        if not common_docs:
            return []
        
        ranked = sorted(
        common_docs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked

def tfidf_query(index, idf, query):
    terms = query_tokenizer(query)

    if not terms:
        return []

    scores = {}

    for term in terms:
        if term not in index:
            continue

        for doc, tf in index[term].items():
            scores[doc] = scores.get(doc, 0) + tf * idf[term]

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked
    

if __name__ == "__main__":
    docs_path = "data"
    index = build_inverted_index(docs_path)
    total_docs = len(os.listdir(docs_path))

    idf = compute_idf(index, total_docs)

    while True:
        q = input("Search (or 'exit'): ")
        if q == "exit":
            break

        results = tfidf_query(index, idf, q)

        if not results:
            print("No results found.")
        else:
            for doc, score in results:
                print(f"{doc}: {score:.4f}")