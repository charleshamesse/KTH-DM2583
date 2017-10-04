import time
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
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
    tr = load_txt("data/train_set.txt")
    tr_x = tr[0]
    tr_y = tr[1]

    ts_x_raw = load_xlsx("data/test_set.xlsx")
    ts_x = [row["A"] for row in ts_x_raw]

    ts_y_raw = load_txt("data/test_set_y.txt")
    ts_y = ts_y_raw[1]
    ts_y = ts_y[0:len(ts_y)-1] # because there's a new line at the end

    if compare_datasets:
        # Check our test labels against Eysteinn's
        ts_y_alternate = load_csv("data/test_dataset.csv")
        different = []
        for i in range(len(ts_y_alternate)):
            if ts_y[i] is not ts_y_alternate[i]:
                different.append(i)
        print("Number of different entries:")
        print(len(different))
        print(different)

    print("Creating features from training set..")
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True)
    tr_vectors = vectorizer.fit_transform(tr_x)

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
                    clf.fit(tr_vectors, tr_y)
                    ts_x_featurized = vectorizer.transform(ts_x)
                    predictions = clf.predict(ts_x_featurized)
                    t1 = time.time()
                    i = 0
                    correct_predictions = 0
                    for row in predictions:
                        if row == ts_y[i]:
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
    print(classification_report(ts_y, bestPredictions))


if __name__ == '__main__':
    main()
