import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

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
    tr = load_data("data/train_set.txt")
    tr_x = tr[0]
    tr_y = tr[1]

    ts = xlsx("data/test_set.xlsx")
    ts_x = [row["A"] for row in ts]

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    tr_vectors = vectorizer.fit_transform(tr_x)

    # TODO: use naive bayes
    clf = MultinomialNB()
    clf.fit(tr_vectors, tr_y)
    ts_x_featurized = vectorizer.transform(ts_x)
    predictions = clf.predict(ts_x_featurized[1:10])

    print(predictions)

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
