__author__ = 'alighahramani'
from sklearn.ensemble import RandomForestClassifier as RF
import math

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




class DecisionTree():
    '''
        Sklearn-style decision tree classifier, using entropy
        '''
    def __init__(self, attrib_d=None, attribs=None,default_v=None, depth = 1):
        ''' initialize classifier
            '''
        if not attribs:
            attribs = []
        if attrib_d:
            self.attrib_dict = attrib_d
        else:
            self.attrib_dict = {}
        self.attribute_list = attribs
        self.default_value = default_v
        self.depth = depth


    def fit(self, X, y):
        '''X and y are as in sklearn classifier.fit expected arguments
            Creates a decision tree
            '''
        if not self.attrib_dict:
            if X:
                self.attrib_dict = [str(y) for y in range(len(X[0]))]
        if not self.attribute_list:
            self.attribute_list = self.attrib_dict;
        self.clf = self.makeTree(X, y, self.attribute_list, self.attrib_dict, self.default_value)
        #printTree(self.clf)
        #return self.clf

    def predict(self, X):
        ''' Return a class label using the decision tree created by the fit method
            '''
        target = []
        for x in X :
            target.append(self.clf.classify(x , self.attribute_list , self.default_value))
        return target


    def entropy(self, labels):
        '''takes as input a list of class labels. Returns a float
            indicating the entropy in this data.
            Hint: you don't have to implement log_2(x), see math.log()
            '''
        ent = 0.0;
        target_names= list(set(labels))
        for x in target_names:
            case = x;
            countCase = 0;
            for x in labels :
                if x ==case :
                    countCase+=1
            if countCase != 0:
                ent += -1 * (float(countCase)*math.log(float(countCase)/len(labels) , 2)/len(labels))
        return ent



    ### Compute remainder - this is the amount of entropy left in the data after
    ### we split on a particular attribute. Let's assume the input data is of
    ### the form:
    ###    [(value1, class1), (value2, class2), ..., (valuen, classn)]
    def remainder(self, data) :
        possibleValues = set([item[0] for item in data])
        r = 0.0

        for value in possibleValues :
            c = [item[0] for item in data].count(value)
            r += (float(c) / len(data) ) * self.entropy([item[1] for item in
                                                         data if item[0] == value])
        return r

    ###
    def selectAttribute(self, x, y):
        '''
            selectAttribute: choose the index of the attribute in the current
            dataset that minimizes remainder(A).
            '''
        attribNumber = 0;
        min = 1000;
        for i in range(len(x[0])):
            attribList = [];
            for j in range(len( x)):
                temp = (x[j][i] , y[j]);
                attribList.append(temp);
            temp = self.remainder(attribList);
            if ( temp <min):
                min = temp;
                attribNumber = i;

        return attribNumber;

    ### a tree is simply a data structure composed of nodes (of type TreeNode).
    ### The root of the tree
    ### is itself a node, so we don't need a separate 'Tree' class. We
    ### just need a function that takes in a dataset and our attribute dictionary,
    ### builds a tree, and returns the root node.
    ### makeTree is a recursive function. Our base case is that our
    ### dataset has entropy 0 - no further tests have to be made. There
    ### are two other degenerate base cases: when there is no more data to
    ### use, and when we have no data for a particular value. In this case
    ### we use either default value or majority value.
    ### The recursive step is to select the attribute that most increases
    ### the gain and split on that.
    ### assume: input looks like this:
    ### dataset: [[v11, v21, ..., vd1], [v12,v22, ..., vd2] ...[v1n,v2n,...vdn] ],
    ###    remaining training examples with values for only the unused features
    ### labels: [c1, ..., cn], remaining target labels for the dataset
    ### attributes: [a1, a2, ...,ax] the list of remaining attribute names
    ### attrib_dict: {a1: [a1vals], a2: [a2vals],...,ad: [advals]}
    ### the dictionary keys are attribute names and the dictionary values are either the list
    ### of values that attribute takes on or 'real' for real-valued attributes (handle for Extra Credit)
    def makeTree(self, dataset, labels, attributes, attrib_dict, defaultValue , depth = 0):
        ''' Helper recursive function for creating a tree
            '''
        if not self.attribute_list :
            self.attribute_list = attributes;
        if not dataset :
            return TreeNode(None , defaultValue , 0)
        if not attributes:
            return TreeNode(None , self.majority(labels) , len(dataset));
        if self.sameClass(labels):

            return TreeNode(None, labels[0] , len(dataset));
        else:

            attrib = self.selectAttribute(dataset , labels);
            attribValues = attrib_dict.get(self.attribute_list[attrib]);
            newAttrib = [];
            for y in attributes:
                if y!= self.attribute_list[attrib]:
                    newAttrib.append(y);
            head = TreeNode(self.attribute_list[attrib] , None , len(dataset))
            if depth>= self.depth:
                return head
            for x in attribValues:
                tempData = [];
                tempLabel = [];
                for i in range(len(dataset)):
                    if dataset[i][attrib] == x:
                        tempData.append(dataset[i]);
                        tempLabel.append(labels[i]);
                head.children.update({x:self.makeTree(tempData , tempLabel ,newAttrib, attrib_dict , defaultValue , depth = depth+1)});



            return head


    def majority(self , labels):
        '''
            A method to find the majority of the target values
            '''
        value1 = labels[0]
        value2 = labels[0]
        for x in labels:
            if x != value2 :
                value2 = x;
                break;
        if value1 != value2 :
            value1Count = 0;
            value2Count = 0;
            for x in labels:
                if x== value1 :
                    value1Count+=1;
                elif x==value2 :
                    value2Count+=1;
            if value1Count > value2Count:
                return value1;
            else:
                return value2;
        else :
            return value1;



    def sameClass(self , labels):
        '''
            A method to check if we have reached a leaf
            '''
        tar = labels[0];
        for x in labels:
            if x != tar :
                return False;
        return True;
### Helper class for DecisionTree.
### A TreeNode is an object that has either:
### 1. An attribute to be tested and a set of children, one for each possible
### value of the attribute, OR
### 2. A value (if it is a leaf in a tree)
class TreeNode:
    def __init__(self, attribute, value , numberOfData):
        self.attribute = attribute
        self.value = value
        self.children = {}
        self.number = numberOfData;

    def __repr__(self):
        if self.attribute:
            return self.attribute
        else:
            return self.value

    ### a node with no children is a leaf
    def is_leaf(self):
        return self.children == {}

    ###
    def classify(self, x, attributes, default_value):
        '''
            return the value for the given data
            the input will be:
            x - an object to classify - [v1, v2, ..., vn]
            attributes - the names of all the attributes
            '''
        if self.is_leaf():
            return self.value;
        else:
            attId = attributes.index(self.attribute);
            attValue = x[attId];
            temp = self.children.get(attValue)
            if not temp :
                return default_value
            return temp.classify(x , attributes , default_value);


#here's a way to visually check your tree
def printTree(root, val='Tree', indentNum=0):
    """ For printing the decision tree in a nice format
        Usage: printTree(rootNode)
        """
    indent = "\t"*indentNum
    if root.is_leaf():
        print indent+"+-"+str(val)+'-- '+str(root.value)
        print indent + 'Number of Samples:' + str(root.number)

    else:
        print indent+"+-"+str(val)+'-- <'+str(root.attribute)+'>'
       # print len(root.children)
        print indent+"{"
        for k in root.children.keys():
            printTree(root.children[k],k,indentNum+1)
        print indent+"}"


if __name__ == '__main__':

    #will need some args in constructor
    tree = DecisionTree(attrib_d = data['feature_dict'],attribs=data['feature_names'] , default_v = zeroR(data))
    tree.fit(data['data'], data['target'])
    #test on training data
    tt =tree.predict(data['data'])

