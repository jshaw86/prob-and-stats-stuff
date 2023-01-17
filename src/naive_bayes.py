import os
import io
import sys
import numpy
import csv
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield path, row


def dataFrameFromDirectory(path):
    rows = []
    index = []
    for filename, line in readFiles(path):
        reason = line['reason']
        body = line['body']
        rows.append({
            'body': body, 
            'reason': reason,
        })
        index.append(filename)
    return DataFrame(rows, index=index)

data = DataFrame({'body':[], 'reason':[]})

data = data.append(dataFrameFromDirectory(sys.argv[1]))

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['body'].values)

classifier = MultinomialNB()
targets = data['reason'].values
classifier.fit(counts, targets)
