import os
import re
from collections import defaultdict

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b[a-z]+\b', text)

def build_index(docs_path):
    index = defaultdict(set)

    for filename in os.listdir(docs_path):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(docs_path, filename), "r", encoding = "utf-8") as f:
            words = tokenize(f.read())
            for word in words:
                index[word].add(filename)
    return index

def search(query, index):
    query_words = tokenize(query)
    scores = defaultdict(int)

    for word in query_words:
        for doc in index.get(word, []):
            scores[doc] += 1
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)



if __name__ == "__main__":
    docs_path = "docs"
    index = build_index(docs_path)

    while True:
        query = input("\nSearch (or 'exit'): ")
        if query.lower() == 'exit':
            break
        results = search(query, index)
        print("Results:")
        for doc, score in results:
            print(f"{doc}: {score}")

