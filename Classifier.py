from sklearn.cluster.tests.test_k_means import n_samples, n_features


from sklearn.ensemble import RandomForestClassifier as RF
import math
import csv
import collections
import random
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import sklearn.datasets
import collections


class RandomForest :

    def __init__(self,n_tree = 10,
                 split_function = "entropy",
                 max_depth = None,
                 min_samples_split=2,
                 max_features = "auto",
                 ):
        self.split_function = split_function
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_tree = n_tree




from SystemEvents.Text_Suite import attribute_run
from sklearn.externals import joblib
import mlUtil
import math
import argparse




class RandomForest():
    #We have number of trees. Random features for each one. Compare them to the real random forest classifier.
    def __init__(self, estimators):
        self.estimators = estimators
        self.tree_classifiers = []

    def fit (self, data, label):
        data = numpy.array(data)
        data_range = range(len(data[0]))
        feature_length = int(math.sqrt(len(data[0])))
        dlabels = zip(data,label)
        for _ in range(self.estimators):
            random.shuffle(dlabels)
            data_each = numpy.array([i[0] for i in dlabels])
            label_each = numpy.array([i[1] for i in dlabels])
            data_for_each = data_each[:2*len(data_each)/10]
            random.shuffle(data_range)
            each_data_range = data_range[:feature_length]
            each_data = data_for_each[:,each_data_range]
            #clf = DecisionTreeClassifier()
            clf = DecisionTreeRegressor()
            clf.fit(each_data,label_each[:2*len(dlabels)/10])
            self.tree_classifiers.append((each_data_range,clf))

    def predict(self,data):
        predicts = []
        results = []
        data = numpy.array(data)
        for data_range,clf in self.tree_classifiers:
            predicts.append(clf.predict(data[:,data_range]))
        reformed_predicts = numpy.transpose(numpy.array(predicts))
        for _ in reformed_predicts:

            results.append(collections.Counter(_).most_common()[0][0])

        return results






if __name__ == '__main__':
    rd = RandomForest(100)
    digits = {}
    #temp_data = sklearn.datasets.make_classification(n_samples=40000,n_features=40 ,n_informative = 3, n_classes = 2)
    #digits['data'] = temp_data[0]
    #digits['target'] = temp_data[1]
    digits = sklearn.datasets.load_diabetes()
    data_length = 9 * len(digits['data'])/10
    rd.fit(digits['data'][:data_length],digits['target'][:data_length])
    results = rd.predict(digits['data'][data_length:])
    labels = digits['target'][data_length:]
    rmse = 0
    #print digits['data'][0]
    for res_index,res in enumerate(results):
       rmse += ((res - labels[res_index])*(res - labels[res_index]))
    print math.sqrt(rmse/float((len(digits['data'])/10)))

    #print [res == labels[res_index]   for res_index,res in enumerate(results)].count(True) / float((len(digits['data'])/10))




def read_csv(file , label = 0):
    csv_reader = csv.reader(file)
    header = []
    data = []
    labels =[]
    attribs = []
    for row in csv_reader:
        if not header :
            header.extend(row)
            attribs.extend(row[:label])
            if label+1 < len(row):
                attribs.extend(row[label+1:])
        else :
            temp_row = []
            temp_row.extend(row[:label])
            if label+1 < len(row):
                temp_row.extend(row[label+1:])
            data.append(temp_row)
            labels.append(row[label])
    return header,attribs , data, labels
