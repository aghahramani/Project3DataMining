import sys

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
                valueDic[dataValues[i]] = values[i].strip()
            data.append(valueDic)

    return data

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


if __name__ == '__main__':
    args = sys.argv
    filename = args[1]
    data = ProcessData(filename)
    print "Data length:",len(data)
    artistData = RatingByMultiKeys(data, ["Artist","Track"])
    for artist in artistData :
        print "Artist:",artist
        print artistData[artist]
        print len(artistData)
        print len(artistData[artist])
        break
