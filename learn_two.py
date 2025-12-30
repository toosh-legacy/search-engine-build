import os
import re
import math
from collections import Counter

# ----------------------------
# Tokenization
# ----------------------------
def tokenize(text):
    text = text.lower()
    return re.findall(r'\b[a-z]+\b', text)

# ----------------------------
# Load Documents
# ----------------------------
def load_documents(path):
    docs = {}
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                docs[file] = tokenize(f.read())
    return docs

# ----------------------------
# Build Document Frequencies
# ----------------------------
def build_doc_freq(docs):
    df = Counter()
    for words in docs.values():
        df.update(set(words))  # count once per document
    return df

# ----------------------------
# Build TF-IDF Vectors
# ----------------------------
def build_tfidf(docs, df):
    N = len(docs)
    tfidf = {}

    for doc, words in docs.items():
        tf = Counter(words)
        total_terms = sum(tf.values())

        vec = {}
        for term, count in tf.items():
            tf_val = count / total_terms
            idf_val = math.log(N / (1 + df[term]))
            vec[term] = tf_val * idf_val

        tfidf[doc] = vec

    return tfidf

# ----------------------------
# Cosine Similarity
# ----------------------------
def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vec2)
    mag1 = math.sqrt(sum(v*v for v in vec1.values()))
    mag2 = math.sqrt(sum(v*v for v in vec2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

# ----------------------------
# Search
# ----------------------------
def search(query, tfidf):
    query_terms = tokenize(query)
    q_tf = Counter(query_terms)
    total = sum(q_tf.values())

    query_vec = {
        term: (count / total)
        for term, count in q_tf.items()
    }

    scores = []
    for doc, doc_vec in tfidf.items():
        score = cosine_similarity(doc_vec, query_vec)
        if score > 0:
            scores.append((doc, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # 1. Define folder and check if it exists
    folder = "docs"
    if not os.path.exists(folder):
        print(f"FAILED: The folder '{folder}' was not found in: {os.getcwd()}")
    else:
        # 2. Try to load documents
        docs = load_documents(folder)
        
        if not docs:
            print(f"FAILED: Found folder '{folder}', but it contains no .txt files.")
        else:
            print(f"SUCCESS: Loaded {len(docs)} files: {list(docs.keys())}")
            
            # 3. Build the engine
            df = build_doc_freq(docs)
            tfidf = build_tfidf(docs, df)
            print("Engine is ready. Type a word to search.")

            while True:
                q = input("\nSearch (or 'exit'): ").strip()
                if q.lower() == "exit":
                    break
                
                # 4. Show results or a message if nothing matches
                results = search(q, tfidf)
                if not results:
                    print(f"No matches found for '{q}'. Try words you know are in the files.")
                else:
                    for doc, score in results:
                        print(f"Result: {doc} | Score: {score:.4f}")
