import sys

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


if __name__ == '__main__':
    args = sys.argv
    filename = args[1]
    data = ProcessData(filename)
    print data[0]
