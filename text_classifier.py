# classifying a body of text to a label
# spam or not spam classification problem
# machine learning?

#use text classification on any binary data set , one or the other


import nltk;
import random;
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
# A wrappper for using scikit learn classifier machine learning algorithms
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI # inherit from nltk classifier
from statistics import mode # for our confidence score
from unidecode import unidecode



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


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


short_pos = open("datasets/pos.txt","r").read()
short_neg = open("datasets/neg.txt","r").read()
documents = []

for r in short_pos.split('\n'):
    documents.append( (r,"pos") )

for r in short_neg.split('\n'):
    documents.append( (r,"neg") )

all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

featuresets = [(find_features(rev),category) for (rev,category) in documents]
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]


""" Loading Pickled Content !!! """



load_classifier = open("pickle/naivebayes.pickle","rb") ## wb -write bytes
classifier = pickle.load(load_classifier)
print (" Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)


load_classifier = open("pickle/MNB.pickle","rb") ## wb -write bytes
MNB_classifier = pickle.load(load_classifier)
print (" MultinomialNB Classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

#BernoulliNB

load_classifier = open("pickle/BernoulliNB.pickle","rb") ## wb -write bytes
BernoulliNB_classifier = pickle.load(load_classifier)
print (" BernoulliNB Classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)




## LOgistic Regression,SVC,LinearSVC,NuSVC and SGDClassifier

load_classifier = open("pickle/LogisticRegression.pickle","rb") ## wb -write bytes
LogisticRegression_classifier = pickle.load(load_classifier)
print (" LogisticRegression Classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)


load_classifier = open("pickle/SGDClassifier.pickle","rb") ## wb -write bytes
SGDClassifier_classifier = pickle.load(load_classifier)
print (" SGDClassifier Classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)


load_classifier = open("pickle/SVC_classifier.pickle","rb") ## wb -write bytes
SVC_classifier = pickle.load(load_classifier)
print (" SVC Classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)


load_classifier = open("pickle/LinearSVC.pickle","rb") ## wb -write bytes
LinearSVC_classifier = pickle.load(load_classifier)
print (" LinearSVC Classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)


load_classifier = open("pickle/NuSVC_classifier.pickle","rb") ## wb -write bytes
NuSVC_classifier = pickle.load(load_classifier)
print (" NuSVC Classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

## Our confidence parameter
# if 3 out of 7 are -ve then our confidence is a little lower
# So we're making a classifier is a compilation of all the above classifiers
# Our Classifier

vote_classifier = VoteClassifier(
                                MNB_classifier,
                                BernoulliNB_classifier,
                                SVC_classifier,
                                NuSVC_classifier,
                                LinearSVC_classifier)
print (" Voted Classifier accuracy percent:", (nltk.classify.accuracy(vote_classifier,testing_set))*100)

print("Classification: ", vote_classifier.classify(testing_set[0][0]),"Confidence:",vote_classifier.confidence(testing_set[0][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[1][0]),"Confidence:",vote_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[2][0]),"Confidence:",vote_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[3][0]),"Confidence:",vote_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", vote_classifier.classify(testing_set[4][0]),"Confidence:",vote_classifier.confidence(testing_set[4][0])*100)
