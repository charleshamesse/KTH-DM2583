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
    print("Fetching training and testing datasets..")
    train_data = get_data('data/data_movies_tr.json')
    test_data = get_data('data/data_movies_ts.json')

    trainX, trainY = split_text_rating(train_data)
    testX, testY = split_text_rating(test_data)

    print("Creating features from training set..")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True)
    tr_vectors = vectorizer.fit_transform(trainX)

    ts_x_featurized = vectorizer.transform(testX)
    
    
    params = {'trainX':tr_vectors, 'trainY':trainY, 'testX':ts_x_featurized, 'testY':testY}
    knnclf = train_and_test(KNeighborsClassifier(n_neighbors=40), params, 'knn')
    svmclf = train_and_test(svm.SVC(kernel='poly', C=5, degree=4, coef0=2), params, 'svm')
    nnclf = train_and_test(MLPClassifier(max_iter=500, alpha=0.01, activation="logistic", learning_rate="invscaling", hidden_layer_sizes=(10, 10, 110)), params, 'neural network')
    nbclf = train_and_test(MultinomialNB(), params, 'naïve Bayes')
    dtclf = train_and_test(tree.DecisionTreeClassifier(), params, 'decision tree')  

    train_and_test(VotingClassifier(estimators=[('knn', knnclf), ('svm', svmclf), ('nn', nnclf), ('nb', nbclf), ('dt', dtclf)], voting='hard'), params, 'ensemble')

    

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
