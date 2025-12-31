import re
import os
import string


STOP_WORDS = {
    "the", "is", "at", "on", "and", "a", "an", "of", "to", "in"
}

def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', '' + string.punctuation))
    tokens = [w for w in text.split() if w and w not in STOP_WORDS]
    return tokens


def parse_file(file_path):

    all_tokens = []


    for filename in os.listdir(file_path):
        if filename.endswith('.txt'):
            with open(os.path.join(file_path, filename), "r", encoding = "utf-8") as f:
                text = f.read()
                all_tokens.extend(tokenize(text))
    return all_tokens

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

if __name__ == "__main__":
    docs_path = "data"
    index = build_inverted_index(docs_path)

    for word, postings in index.items():
        print(word, postings)