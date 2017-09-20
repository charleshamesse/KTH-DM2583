import time
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def xlsx(fname):
    import zipfile
    from xml.etree.ElementTree import iterparse
    z = zipfile.ZipFile(fname)
    strings = [el.text for e, el in iterparse(z.open('xl/sharedStrings.xml')) if el.tag.endswith('}t')]
    rows = []
    row = {}
    value = ''
    for e, el in iterparse(z.open('xl/worksheets/sheet1.xml')):
        if el.tag.endswith('}v'): # <v>84</v>
            value = el.text
        if el.tag.endswith('}c'): # <c r="A3" t="s"><v>84</v></c>
            if el.attrib.get('t') == 's':
                value = strings[int(value)]
            letter = el.attrib['r'] # AZ22
            while letter[-1].isdigit():
                letter = letter[:-1]
            row[letter] = value
            value = ''
        if el.tag.endswith('}row'):
            rows.append(row)
            row = {}
    return rows


def load_data(path):
    lines = []
    tr_x = []
    tr_y = []

    with open(path) as f:
        content = f.read()
        lines = content.split('\n')

    for line in lines:
        line_components = line.split('\t')
        tr_x.append(line_components[1])
        tr_y.append(line_components[0])

    return [tr_x, tr_y]

def main():
    # Fetch train and test data
    tr = load_data("data/train_set.txt")
    tr_x = tr[0]
    tr_y = tr[1]

    ts = xlsx("data/test_set.xlsx")
    ts_x = [row["A"] for row in ts]

    # Featurize test data
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=True)
#HashingVectorizer(stop_words='english', alternate_sign=False, n_features=2 ** 16)
    tr_vectors = vectorizer.fit_transform(tr_x)

    # Instanciate classifier
    clf = MultinomialNB()
    clf.fit(tr_vectors, tr_y)
    ts_x_featurized = vectorizer.transform(ts_x)

    # Make predictions
    predictions = clf.predict(ts_x_featurized[1:10])
    i = 0
    for row in predictions:
        print([row, ts_x[i]])
        print()
        i = i + 1

    '''
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(tr_vectors, tr_y)
    t1 = time.time()
    #prediction_rbf = classifier_rbf.predict(test_vectors)
    #t2 = time.time()
    #time_rbf_train = t1-t0
    #time_rbf_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    #print(classification_report(ts_y, prediction_rbf))
    '''

if __name__ == '__main__':
    main()
