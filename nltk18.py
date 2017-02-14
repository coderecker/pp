import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
from nltk.tokenize import word_tokenize

class Votes_Classifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            votes.append(classifier.classify(features))

        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            votes.append(classifier.classify(features))
        conf = votes.count(mode(votes)) / len(votes)
        return conf


documents = []
all_words = []


short_pos = open('short_reviews/positive.txt','r').read()
short_neg = open('short_reviews/negative.txt','r').read()

short_pos = word_tokenize(short_pos)
short_neg = word_tokenize(short_neg)

for words in short_pos:
    documents.append((words,'pos'))
    all_words.append(words.lower())

for words in short_neg:
    documents.append((words,'neg'))
    all_words.append(words.lower())

all_words = nltk.FreqDist(all_words)

random.shuffle(documents)

def find_features(document):
    words = set(document)
    features = {}
    for w in list(all_words.keys())[:3000]:
        features[w] = (w in words)

    return features

featuresets = [(find_features(word), category) for word, category in documents]

Positive Test Data
train_set = featuresets[:10000]
test_set = featuresets[10000:]

###Negative Test Data
##train_set = featuresets[100:]
##test_set = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)
fh = open('moviereviews_naive.pickle','wb')
##fh = open('moviereviews_naive.pickle','rb')
pickle.dump(classifier,fh)
##classifier = pickle.load(fh)
fh.close()

print("NLTK Classifier Accuracy Percent: ",nltk.classify.accuracy(classifier, test_set))


MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(train_set)
print("MultinomialNB Classifier Accuracy Percent: ",nltk.classify.accuracy(MultinomialNB_classifier, test_set))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB Classifier Accuracy Percent: ",nltk.classify.accuracy(BernoulliNB_classifier, test_set))
             
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression Classifier Accuracy Percent: ",nltk.classify.accuracy(LogisticRegression_classifier, test_set))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(train_set)
print("SGDClassifier Classifier Accuracy Percent: ",nltk.classify.accuracy(SGDClassifier_classifier, test_set))

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(train_set)
print("SVC Classifier Accuracy Percent: ",nltk.classify.accuracy(SVC_classifier, test_set))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC Classifier Accuracy Percent: ",nltk.classify.accuracy(LinearSVC_classifier, test_set))

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_set)
print("NuSVC Classifier Accuracy Percent: ",nltk.classify.accuracy(NuSVC_classifier, test_set))


voted_classifier = Votes_Classifier(classifier,
                                    MultinomialNB_classifier,
                                    BernoulliNB_classifier,
                                    LogisticRegression_classifier,
                                    SGDClassifier_classifier,
                                    LinearSVC_classifier,
                                    NuSVC_classifier)

print("voted_classifier Classifier Accuracy Percent: ",nltk.classify.accuracy(voted_classifier, test_set))

##print("Voted Classifier Classification: ",voted_classifier.classify(test_set[0][0]), " Voted Classifier Confidence: ",voted_classifier.confidence(test_set[0][0]))
##print("Voted Classifier Classification: ",voted_classifier.classify(test_set[1][0]), " Voted Classifier Confidence: ",voted_classifier.confidence(test_set[1][0]))
##print("Voted Classifier Classification: ",voted_classifier.classify(test_set[2][0]), " Voted Classifier Confidence: ",voted_classifier.confidence(test_set[2][0]))
##print("Voted Classifier Classification: ",voted_classifier.classify(test_set[3][0]), " Voted Classifier Confidence: ",voted_classifier.confidence(test_set[3][0]))
##print("Voted Classifier Classification: ",voted_classifier.classify(test_set[4][0]), " Voted Classifier Confidence: ",voted_classifier.confidence(test_set[4][0]))


                                    
                                    
                                    
                                    
                                    

    
