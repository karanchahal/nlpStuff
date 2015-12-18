# classifying a body of text to a label
# spam or not spam classification problem
# machine learning?

import nltk;
import random;
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
# A wrappper for using scikit learn classifier machine learning algorithms
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from nltk.classify import ClassifierI # inherit from nltk classifier
from statistics import mode # for our confidence score



class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers :
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes)) # how many occurences  of most popular vote were in the list
        conf = float(choice_votes)/len(votes)
        return conf







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


print (" Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)
#classifier.show_most_informative_features(15)


# Saving a python object with PICKLE (serialization)

#save_classifier = open("naivebayes.pickle","wb") ## wb -write bytes
#pickle.dump(classifier,save_classifier)
#save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB()) # making a new classifier
MNB_classifier.train(training_set)
print (" MultinomialNB Classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)


# Gausian and BernouslliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB()) # making a new classifier
BernoulliNB_classifier.train(training_set)
print (" BernoulliNB Classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)



"""
GaussianNB_classifier = SklearnClassifier(GaussianNB()) # making a new classifier
GaussianNB_classifier.train(training_set)
print (" GaussianNB Classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)
"""

LogisticRegression_classifier = SklearnClassifier(LogisticRegression()) # making a new classifier
LogisticRegression_classifier.train(training_set)
print (" LogisticRegression Classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

## LOgistic Regression,SVC,LinearSVC,NuSVC and SGDClassifier

SGDClassifier_classifier = SklearnClassifier(SGDClassifier()) # making a new classifier
SGDClassifier_classifier.train(training_set)
print (" SGDClassifier Classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)


SVC_classifier = SklearnClassifier(SVC()) # making a new classifier
SVC_classifier.train(training_set)
print (" SVC Classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)


LinearSVC_classifier = SklearnClassifier(LinearSVC()) # making a new classifier
LinearSVC_classifier.train(training_set)
print (" LinearSVC Classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)


NuSVC_classifier = SklearnClassifier(NuSVC()) # making a new classifier
NuSVC_classifier.train(training_set)
print (" NuSVC Classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

## Our confidence parameter
# if 3 out of 7 are -ve then our confidence is a little lower
# So we're making a classifier is a compilation of all the above classifiers
# Our Classifier

vote_classifier = VoteClassifier(classifier,
                                MNB_classifier,
                                BernoulliNB_classifier,
                                SVC_classifier,
                                SGDClassifier_classifier,
                                NuSVC_classifier,
                                LinearSVC_classifier)
print (" Voted Classifier accuracy percent:", (nltk.classify.accuracy(vote_classifier,testing_set))*100)

print("Classification: ", vote_classifier.classify(testing_set[0][0]),"Confidence:",vote_classifier.confidence(testing_set[0][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[1][0]),"Confidence:",vote_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[2][0]),"Confidence:",vote_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[3][0]),"Confidence:",vote_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[4][0]),"Confidence:",vote_classifier.confidence(testing_set[4][0])*100)
