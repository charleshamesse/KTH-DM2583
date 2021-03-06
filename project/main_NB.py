import time
import random
import itertools
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from helpers import get_data, split_text_rating

compare_datasets = False

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def main():
    t0 = time.time()
    print("Fetching training and testing datasets..")
    data = get_data('data/data_music.json')

    random.shuffle(data)
    train_data = data[0:int(len(data)/2)]
    test_data = data[int(len(data)/2):len(data)]

    trainX, trainY = split_text_rating(train_data)
    testX, testY = split_text_rating(test_data)

    print("Creating features from training set..")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True)
    tr_vectors = vectorizer.fit_transform(trainX)

    print("Creating classifier..")
    parameters = {
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'hidden_layer_sizes': [x for x in itertools.product((10,60,110,160),repeat=3)],
        'alpha': 10.0 ** -np.arange(1, 5),
        'activation': ["logistic", "relu"]
    }
    mlp = MLPClassifier()
    clf = GridSearchCV(mlp, parameters)
    clf.fit(tr_vectors, trainY)
    sorted(clf.cv_results_.keys())
    ts_x_featurized = vectorizer.transform(testX)

    print("Making predictions..")
    predictions = clf.predict(ts_x_featurized)
    t1 = time.time()
    dt = t1 - t0
    i = 0
    correct_predictions = 0
    for row in predictions:
        if row == testY[i]:
            correct_predictions = correct_predictions + 1
        i = i + 1

    print("Result: %d/%d correct predictions (%.2f%%), in %.2fs.\n" % (correct_predictions, len(predictions), 100.*correct_predictions/len(predictions), dt))
    print(classification_report(testY, predictions))


if __name__ == '__main__':
    main()
