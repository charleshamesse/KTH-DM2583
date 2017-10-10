import time
import random
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from helpers import get_data, split_text_rating
from sklearn.ensemble import VotingClassifier

compare_datasets = False

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def main():
    print("Fetching training dataset..")
    data = get_data('data/data_movies_tr.json')#[:100]

    X, Y = split_text_rating(data)

    print("Creating features from training set..")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True)
    vectors = vectorizer.fit_transform(X)    
    
    #Train classifiers
    t0 = time.time()
    params = {'trainX':vectors, 'trainY':Y}
    knnclf = train(KNeighborsClassifier(n_neighbors=40), params, 'knn')
    svmclf = train(svm.SVC(kernel='poly', C=5, degree=4, coef0=2), params, 'svm')
    nnclf = train(MLPClassifier(max_iter=500, alpha=0.01, activation="logistic", learning_rate="invscaling", hidden_layer_sizes=(10, 10, 110)), params, 'neural network')
    nbclf = train(MultinomialNB(), params, 'naïve Bayes')
    dtclf = train(tree.DecisionTreeClassifier(), params, 'decision tree')  

    vclf = train(VotingClassifier(estimators=[('knn', knnclf), ('svm', svmclf), ('nn', nnclf), ('nb', nbclf), ('dt', dtclf)], voting='hard'), params, 'ensemble')
    t1 = time.time()
    dt = t1 - t0
    print("Training all classifiers took %.2fs.\n" % (dt))

    #Test classifiers
    print("Fetching testing dataset..")
    data = get_data('data/data_movies_ts.json')#[:100]
    X, Y = split_text_rating(data)
    print("Creating features from testing set..")
    vectors = vectorizer.transform(X)

    t0 = time.time()
    params = {'testX':vectors, 'testY':Y}
    test(knnclf, params,'knn')
    test(svmclf, params, 'svm')
    test(nnclf, params, 'neural network')
    test(nbclf, params, 'naïve Bayes')
    test(dtclf, params, 'decision tree')
    test(vclf, params, 'voting ensemble classifier')
    t1 = time.time()
    dt = t1 - t0
    print("Testing all classifiers took %.2fs.\n" % (dt))


def train(clf,params,name):
    t0 = time.time()
    print("Creating " + name + " classifier..")
    clf.fit(params['trainX'], params['trainY'])
    t1 = time.time()
    dt = t1 - t0
    print("...in %.2fs.\n" % (dt))
    return clf

def test(clf, params, name):
    t0 = time.time()

    print("Making " + name + " predictions..")
    predictions = clf.predict(params['testX'])
    t1 = time.time()
    dt = t1 - t0
    i = 0
    correct_predictions = 0
    for row in predictions:
        if row == params['testY'][i]:
            correct_predictions = correct_predictions + 1
        i = i + 1

    print("Result for " + name + " classifier: %d/%d correct predictions (%.2f%%), in %.2fs.\n" % (correct_predictions, len(predictions), 100.*correct_predictions/len(predictions), dt))
    print(classification_report(params['testY'], predictions))

def train_and_test(clf, params, name):
    t0 = time.time()
    print("Creating " + name + " classifier..")
    clf.fit(params['trainX'], params['trainY'])

    print("Making " + name + " predictions..")
    predictions = clf.predict(params['testX'])
    t1 = time.time()
    dt = t1 - t0
    i = 0
    correct_predictions = 0
    for row in predictions:
        if row == params['testY'][i]:
            correct_predictions = correct_predictions + 1
        i = i + 1

    print("Result for " + name + " classifier: %d/%d correct predictions (%.2f%%), in %.2fs.\n" % (correct_predictions, len(predictions), 100.*correct_predictions/len(predictions), dt))
    print(classification_report(params['testY'], predictions))
    return clf


if __name__ == '__main__':
    main()
