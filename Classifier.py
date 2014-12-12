from sklearn.cluster.tests.test_k_means import n_samples, n_features


from sklearn.ensemble import RandomForestClassifier as RF
import math
import csv
import collections
import random
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import sklearn.datasets
import collections
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import sys


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
            label_each = numpy.array([i[1] for i in dlabels])
            data_each = numpy.array([i[0] for i in dlabels])
            data_for_each = data_each[:3*len(data_each)/10]
            random.shuffle(data_range)
            each_data_range = data_range[:feature_length]
            #print each_data_range
            each_data = data_for_each[:,each_data_range]
            #clf = DecisionTreeClassifier()
            #clf = DecisionTreeRegressor()
            clf = linear_model.LinearRegression()
            clf.fit(each_data,label_each[:3*len(dlabels)/10])
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

def create_user_dictionary_mean(data , labels): #index 2 is the user , index 3 is the rating
    user_dictionary = {}
    for _ in zip(data , labels):
        if not _[0][2] in user_dictionary:
            user_dictionary[_[0][2]] = [0,0] #index 0 is the total value, index 1 is the count
        user_dictionary[_[0][2]][0]+=int(_[1])
        user_dictionary[_[0][2]][1]+=1
    result = {}
    for i in user_dictionary: #Calculating the mean
        result[i] = user_dictionary[i][0]/float(user_dictionary[i][1])
    return result

def modify_words(data,labels):
    data = [i[1:] for i in data] # getting rid of index 0 which is user id
    artist_dictionary = {}

    for i in labels:
        artist_dictionary[i] = [] # create the artist dictionary

    for i_index, i in enumerate(data):
        if ( len(artist_dictionary[labels[i_index]]) == 0 ):
            artist_dictionary[labels[i_index]] = [0]*(85)
            artist_dictionary[labels[i_index]][84] = 0 # the last index is the count of how many times we saw this
            #artist
        artist_dictionary[labels[i_index]][84]+=1 # Had to hard code this. Because of one of the training file which is
        #corrupted
        for j_index , j in enumerate(i) :
            if j_index > 83 : break # A bug in the report that one of the artist has more than 83 columns information
            ttt = 0
            if j.startswith("H"): # Heard of
                ttt = 1
            elif j.startswith("N"): # Not heard of
                ttt = 0
            elif j.startswith("E") : # Ever heard of
                ttt = -1
            elif j.startswith("L") : # Listened to recently:
                ttt = 1
            elif j.startswith("O"): # Own
                j = j.split(" ")
                if j[1].startswith("n"): # none of the music
                    ttt = 0
                elif j[1].startswith("a"): # all or most of the music
                    ttt = 5
                else :
                    if j[2].startswith("lo"): # lot of the music
                        ttt = 2
                    elif j[1].startswith("li"): # little of the music
                        ttt = 1
            elif j.startswith("D") or j.startswith ("d"): #Dont know
                ttt = 0
            else:
                if j == '':
                    ttt = 0
                else :
                    #print i
                    if len(j.split(".")) > 1: # Getting rid of float numbers
                        j = j.split(".")[0]
                    ttt = int(j)

            artist_dictionary[labels[i_index]][j_index]+=ttt
    result = {}
    for i in artist_dictionary: # Calculating the mean values
        ttt = []
        for j in artist_dictionary[i][:-1]:
            ttt.append(j / float(artist_dictionary[i][-1]))
        result[i] = ttt
    return result

def create_user_dictionary(data , labels):

    user_dictionary = {}

    for i_index , i in enumerate(data):
        user_dictionary[labels[i_index]] = []
        if ( i[0].startswith("L")): # Less than an hour
            i[0] = 0
        elif ( i[0].startswith("M")) : # More than ....
            i[0] = i[0].split(" ")[2]
        else :
            i[0] = i[0].split(' ')[0].split('+')[0]

        user_dictionary[labels[i_index]].extend([float(j) if j != '' else 0 for j in i])
    return user_dictionary


def create_training_data(user_dictionary, artist_dictionary, data_train):

    result = []

    for i in data_train : #index 0 = artist , index 1 = Track , index 2 = user ,
        ttt = []
        ttt.extend(artist_dictionary[i[0]])
        ttt.extend([int(i[1])])
        if i[2] in user_dictionary:
            ttt.extend(user_dictionary[i[2]])
        else:
            ttt.extend([50]*20)
        result.append(ttt[:105]) # Had to hard code because of some missing information in the training file
    return numpy.array(result)


def modify_labels(data , labels, dictionary):#index 0 = artist , index 1 = Track , index 2 = user ,
    for i_index , i in enumerate(labels):
        user = data[i_index][2]
        labels[i_index] = i - dictionary[user]
    return labels

if __name__ == '__main__':

    header_train , attribs_train, data_train, labels_train = read_csv(open("data/train.csv","r") , label = 3) #Reading the data from csv file
    header_sample , attribs_sample, data_sample, labels_sample = read_csv(open("data/sample.csv","r") , label = 3) #Reading the data from csv file
    header_data, attribs_user , data_user, labels_user = read_csv(open("data/users.csv","r"))
    header_words, attribs_words , data_words, labels_words = read_csv(open("data/words.csv","r"))

    data_train = numpy.array(data_train)

    labels_train = [int(i) for i in labels_train]

    data_sample = numpy.array(data_sample)
    labels_sample = [int(i) for i in labels_sample]



    data_train  = data_train[:,:-1]#Removing time from the data

    data_sample = data_sample[:,:-1]


    data_user = numpy.array(data_user)


    data_user = data_user[:,-20:] # Only using the last 20 columns in the file, answers to questions


    user_dictionary_mean = create_user_dictionary_mean(data_train,labels_train)#Extracting user info

    user_dictionary = create_user_dictionary(data_user,labels_user)

    artist_dictionary = modify_words(data_words , labels_words)



    labels_train = modify_labels(data_train , labels_train , user_dictionary_mean)






    #labels_train = modify_labels(data_sample , labels_sample , user_dictionary_mean)





    data = create_training_data ( user_dictionary , artist_dictionary , data_train)


    #data_train_sample= create_training_data ( user_dictionary , artist_dictionary , data_sample)


    # rd = RandomForest(100)
    # data_length = 9 * len(data)/10
    # rd.fit(data[:data_length],labels_train[:data_length])
    # results = rd.predict(data[data_length:])
    # labels = labels_train[data_length:]

    #rd = linear_model.LinearRegression()


    # n = len(data_train_sample)
    #
    # #n = len(data_train)
    #
    # kf = cross_validation.KFold(n, n_folds=10, indices=True)
    #
    # avv_rmse = 0
    # for train, test in kf:
    #     rd = RandomForest(100)
    #     train_set=[]
    #     test_set = []
    #     labels_train = []
    #     labels_test = []
    #     for i in train :
    #         train_set.append(data_train_sample[i])
    #         labels_train.append(labels_sample[i])
    #     for i in test :
    #         test_set.append(data_train_sample[i])
    #         labels_test.append(labels_sample[i])
    #
    #     rd.fit(train_set,labels_train)
    #
    #     results = rd.predict(test_set)
    #
    #     rmse = 0
    #     for res_index,res in enumerate(results):
    #         rmse += ((res - labels_test[res_index])*(res - labels_test[res_index]))
    #
    #     print "each itteration rmse : " , math.sqrt(rmse/float(len(labels_test)))
    #     avv_rmse += math.sqrt(rmse/float(len(labels_test)))
    # print "avverage rmse : " , avv_rmse/10.




    n = len(data)



    kf = cross_validation.KFold(n, n_folds=10, indices=True)

    avv_rmse = 0
    for train, test in kf:
        #rd = RandomForestClassifier(n_estimators=50)
        rd = RandomForest(100)
        #rd= DecisionTreeRegressor()
        #rd = linear_model.LinearRegression()
        train_set=[]
        test_set = []
        labels_t = []
        labels_test = []
        for i in train :
            train_set.append(data[i])
            labels_t.append(labels_train[i])
        for i in test :
            test_set.append(data[i])
            labels_test.append(labels_train[i])

        rd.fit(train_set,labels_t)

        results = rd.predict(test_set)

        rmse = 0
        for res_index,res in enumerate(results):
            rmse += ((res - labels_test[res_index])*(res - labels_test[res_index]))

        print "each itteration rmse : " , math.sqrt(rmse/float(len(labels_test)))
        avv_rmse += math.sqrt(rmse/float(len(labels_test)))
    print "avverage rmse : " , avv_rmse/10.







