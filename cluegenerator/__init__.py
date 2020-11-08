import numpy as np
import pandas as pd

from scipy import spatial
from sklearn.decomposition import PCA

embeddings = {}
with open("glove.42B.300d/top_100000.txt", 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings[word] = vector


def distance(word, reference):
    return spatial.distance.cosine(embeddings[word], embeddings[reference])


def closest_words(reference):
    return sorted(embeddings.keys(), key=lambda w: distance(w, reference))


x = np.zeros((100000, 300))
index = 0
for key in embeddings:
    x[index] = embeddings[key]
    index = index + 1

pca = PCA()
pca.fit(x)

# [(w, ", ".join(closest_words(w)[1:10]) + "...") for w in ["magic", "sport", "scuba", "sock"]]
