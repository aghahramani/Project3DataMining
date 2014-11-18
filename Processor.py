import sys
import cPickle as pickle

### Reads a file and returns the data stored in lists of dictionaries
def ProcessData(filename) :
    data = []
    dataValues = []
    f = open(filename, 'r')
    for line in f :
        if len(dataValues) == 0 :
            dataValues = line.split(",")
        else :
            valueDic = {}
            values = line.split(",")
            for i in range(len(dataValues)) :
                if i >= len(values) :
                    break
                try:
                    valueDic[dataValues[i]] = int(values[i].strip())
                except :
                    valueDic[dataValues[i]] = values[i].strip()
            data.append(valueDic)

    return data

### FOR TRAIN/TEST ONLY
### Returns dictionary of all values for given key1 for each key0.
### Default key1 to Rating and duplication allowed to True
def KeyToKey(data, key0, key1="Rating", dup = True) :
    keyedData = {}
    for part in data :
        value = part[key0]
        if value in keyedData :
            otherKeyValue = part[key1]
            otherKeyValues = keyedData[value]
            if dup or otherKeyValue not in otherKeyValues :
                otherKeyValues.append(otherKeyValue)
            keyedData[value] = otherKeyValues
        else :
            otherKeyValues = [part[key1]]
            keyedData[value] = otherKeyValues

    return keyedData

### FOR TRAIN/TEST ONLY
### Returns multi-layered dictionary of Ratings for list of keys
def RatingByMultiKeys(data, keys) :
    keyedData = {}

    for part in data :
        value = part[keys[0]]
        if value in keyedData :
            keyedData[value] = multiKeysHelper(part, keyedData[value], value, keys, 1)
        else :
            keyedData[value] = multiKeysHelper(part, {}, value, keys, 1)

    return keyedData

### Recursive helper function for RatingByMultiKey
def multiKeysHelper(dataPart, keyedData, oldVal, keys, index) :
    if index == len(keys) or keys[index] == "Rating":
        if oldVal in keyedData :
            keyedData.append(dataPart["Rating"])
            return keyedData
        else :
            return [dataPart["Rating"]]
    else :
        newVal = dataPart[keys[index]]
        index += 1
        if newVal in keyedData :
            keyedData[newVal] = multiKeysHelper(dataPart, keyedData[newVal], newVal, keys, index)
        else :
            keyedData[newVal] = multiKeysHelper(dataPart, {}, newVal, keys, index)
        return keyedData

### Fills in missing values of words.csv with assumed 0
def cleanWords(data) :
    for part in data :
        for key in part :
            if len(str(part[key])) == 0 :
                part[key] = 0

    return data

### FOR WORDS ONLY
### Returns dictionary of all values for given key.
def keyWords(key) :
    keyedData = {}
    for part in data :
        value = part[key]
        if value in keyedData :
            otherData = {}
            for otherKey in part :
                if key != otherKey :
                    otherData[otherKey] = part[otherKey]
            currentData = keyedData[value]
            currentData.append(otherData)
            keyedData[value] = currentData
        else :
            otherData = {}
            for otherKey in part :
                if key != otherKey :
                    otherData[otherKey] = part[otherKey]
            keyedData[value] = [otherData]

    return keyedData

### Normalizes key mapped data (single key only)
def normalize(data) :
    for key in data :
        ratings = data[key]
        a = 0.0
        b = 1.0
        A = 100.0
        B = 0.0
        for rating in ratings :
            if rating < A :
                A = float(rating)
            if rating > B :
                B = float(rating)

        normalizedResults = []
        for rating in ratings :
            if A == B :
                normalizedResults.append(1)
            else :
                normalizedResult = a + ((rating - A) * (b - a)) / (B - A)
                normalizedResults.append(normalizedResult)
        data[key] = normalizedResults
    return data

### Pickle current data structure for future use
def pickleData(filename, data) :
    pickle.dump(data, open(filename, "wb"))

### Unpickle saved data structure
def unpickleData(filename) :
    return pickle.load(open(filename, "rb"))


if __name__ == '__main__':
    args = sys.argv
    filename = args[1]
    try :
        data = unpickleData(filename + ".p")
    except :
        data = ProcessData(filename)
        pickleData(filename + ".p", data)

    data = cleanWords(data)
    for part in data :
        print part
        break
