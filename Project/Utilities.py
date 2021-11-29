import random


def quickselect(indexes, list, k):
    """
    Found at https://rcoh.me/posts/linear-time-median-finding/
    modified to run by giving and taking indexes
    Select the kth element in l (0 based)
    :param l: List of numerics
    :param k: Index
    :param pivot_fn: Function to choose a pivot, defaults to random.choice
    :return: The kth element of l
    """
    if len(list) == 1:
        assert k == 0
        return list[0]

    pivot = list[random.choice(indexes)]

    lows = [el for el in indexes if list[el] < pivot]
    highs = [el for el in indexes if list[el] > pivot]
    pivots = [el for el in indexes if list[el] == pivot]

    if k < len(lows):
        return quickselect(lows, list, k)
    elif k < len(lows) + len(pivots):
        # We got lucky and guessed the median
        return pivots[0]
    else:
        return quickselect(highs, list, k - len(lows) - len(pivots))


def convToBin(num, tup):
    if float(num) < tup[0]:
        return 0
    elif float(num) > tup[1]:
        return 2
    else: return 1


def parseData(File, testD=0):
    data = {}
    attCount = None
    f = open(File, 'r')
    dataCount = 0
    indices = set()
    pList = []
    medList = []
    allList = []
    unknown = set()  # key is index in data, value is term index
    unknownList = []  # unknownList[i] is the mode of that attribute
    for line in f:
        terms = line.strip().split(',')
        if attCount is None:
            for i in range(0, len(terms)):
                pList.append(set())
                pList[i].add(terms[i])
                medList.append([])
                allList.append([])
                allList[i].append(terms[i])
                if terms[i][len(terms[i]) - 1].isdigit():
                    medList[i].append(int(terms[i]))
                else:
                    medList[i] = None
            attCount = len(terms) - 1
            if not testD: medList[len(medList) - 1] = None # don't want to convert our result to discrete even if we can
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        else:
            for i in range(0, len(terms)):
                allList[i].append(terms[i])
                pList[i].add(terms[i])
                if medList[i] is not None and terms[i][len(terms[i]) - 1].isdigit():
                    medList[i].append(int(terms[i]))
                else: medList[i] = None
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)

        data[dataCount] = terms.copy()
        indices.add(dataCount)
        dataCount += 1
        inL = list(indices)
    for i in range(0, len(medList)):
        if medList[i] is not None:
            medList[i] = medList[i][quickselect(inL, medList[i], int(len(medList[i]) / 3))], \
                         medList[i][quickselect(inL, medList[i], int(2*len(medList[i])/3))]
    for key in data.keys():
        for i in range(0, len(medList)):
            if medList[i] is not None:
                pList[i] = {0, 1, 2}
                data[key][i] = convToBin(data[key][i], medList[i])

    dataCount += 1
    f.close()
    return data, dataCount, medList, [data, dataCount, attCount, indices, pList]


def parseTData(File, meds):
    data = {}
    attCount = None
    f = open(File, 'r')
    dataCount = 0
    indices = set()
    #pList = []
    medList = meds
    allList = []
    unknown = set()  # key is index in data, value is term index
    unknownList = []  # unknownList[i] is the mode of that attribute
    for line in f:
        terms = line.strip().split(',')
        ID = terms[0]
        terms = terms[1:]
        if attCount is None:
            #for i in range(1, len(terms)):
                #pList.append(set())
                #pList[i-1].add(terms[i])
            attCount = len(terms) - 1
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        #else:
            #for i in range(1, len(terms)):
                #pList[i-1].add(terms[i])
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)

        data[ID] = terms
        indices.add(dataCount)
        dataCount += 1
        inL = list(indices)


    for key in data.keys():
        for i in range(0, len(medList)):
            if medList[i] is not None:
                #pList[i] = {0, 1, 2}
                data[key][i] = convToBin(data[key][i], medList[i])

    dataCount += 1
    f.close()
    return data, dataCount, medList, [data, dataCount, attCount, indices]

