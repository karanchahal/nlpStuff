# classifying a body of text to a label
# spam or not spam classification problem
# machine learning?

import nltk;
import random;
from nltk.corpus import movie_reviews
import pickle
# words are tuples which are also "features"

documents = [(list(movie_reviews.words(fileid)),category)
            for category in  movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) # ordered from most common to least common word

#print(all_words.most_common(15))
#print(all_words["stupid"])

word_features = list(all_words.keys())[:3000] ## taking top 3000 words
# so that we can train our classifier algo against these words

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive Bayes Algorithm!!
# posterior (liklihood) = (prior occurence x liklihood ) / evidence

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f) # Serialization !!  WOW yeas


print ("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

# Saving a python object with PICKLE (serialization)

#save_classifier = open("naivebayes.pickle","wb") ## wb -write bytes
#pickle.dump(classifier,save_classifier)
#save_classifier.close()
