# classifying a body of text to a label
# spam or not spam classification problem
# machine learning?

import nltk;
import random;
from nltk.corpus import movie_reviews
# words are tuples which are also "features"

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append(list(movie_reviews.words(fileid)))

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

#print(all_words.most_common(15))

print(all_words["stupid"])
