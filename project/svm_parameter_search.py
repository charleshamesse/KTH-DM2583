import time
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from helpers import load_txt, load_xlsx, load_csv

compare_datasets = False

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def main():
    t0 = time.time()
    print("Fetching training and testing datasets..")
    data = get_data('data/data.json')

    random.shuffle(data)
    train_data = data[0:int(len(data)/2)]
    test_data = data[int(len(data)/2):len(data)]

    trainX, trainY = split_text_rating(train_data)
    testX, testY = split_text_rating(test_data)

    print("Creating features from training set..")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True)
    tr_vectors = vectorizer.fit_transform(trainX)

    print("Grid searching params for SVM classifier..")
    #clf = MultinomialNB()
    params = {'kernel':('linear', 'poly', 'rbf'), 'C':[1, 10], 'degree':[2,3,4,5], 'coef0':[5, 7, 10, 15, 17, 20]}
    #params = {'kernel':['poly'], 'C':[10], 'degree':[3], 'coef0':[5]}
    bestclf = 0
    bestRes = 0
    bestPredictions = 0
    for classifier in params['kernel']:
        for c in params['C']:
            for d in params['degree']:
                for coef in params['coef0']:
                    clf = svm.SVC(kernel=classifier, C=c, degree=d, coef0=coef)
                    clf.fit(tr_vectors, trainY)
                    ts_x_featurized = vectorizer.transform(testX)
                    predictions = clf.predict(ts_x_featurized)
                    t1 = time.time()
                    i = 0
                    correct_predictions = 0
                    for row in predictions:
                        if row == testY[i]:
                            correct_predictions = correct_predictions + 1
                        i = i + 1
                    if correct_predictions > bestRes:
                        bestRes=correct_predictions
                        bestclf = clf
                        bestPredictions = predictions
                        print('kernel:', classifier,'C:', c, 'degree:', d, 'coef0:', coef)
                        print('Numcorrect', bestRes)
                    

    dt = t1 - t0
    print("Result: %d/%d correct predictions (%.2f%%), in %.2fs.\n" % (bestRes, len(bestPredictions), 100.*bestRes/len(bestPredictions), dt))
    print(classification_report(testY, bestPredictions))


if __name__ == '__main__':
    main()
