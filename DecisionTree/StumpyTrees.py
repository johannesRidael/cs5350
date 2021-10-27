import math
import sys
from statistics import mode
from statistics import median

#DEPTH = 1
#TRAINING_FILE = "train.csv"
#TEST_FILE = "test.csv"
#TYPE = "GI" # EE, ME, GI

def findEE(data, indexes, splitInd, tagInd, weights):
    splitDic = {}
    count = 0
    for index in indexes:
        if data[index][splitInd] not in splitDic:
            splitDic[data[index][splitInd]] = {}
            splitDic[data[index][splitInd]][data[index][tagInd]] = weights[index]
            splitDic[data[index][splitInd]]["total"] = weights[index]
            count += 1
        elif data[index][tagInd] not in splitDic[data[index][splitInd]]:
            splitDic[data[index][splitInd]][data[index][tagInd]] = weights[index]
            splitDic[data[index][splitInd]]["total"] += weights[index]
            count += 1
        else:
            splitDic[data[index][splitInd]][data[index][tagInd]] += weights[index]
            splitDic[data[index][splitInd]]["total"] += weights[index]
            count += 1

        # splitDic[x][y] now contains a count of that result y with that data x
    entropy = 0
    tSum = 0
    for splitVal in splitDic:
        hold = 0

        for tag in splitDic[splitVal]:
            if tag != "total":
                tSum += splitDic[splitVal][tag]
                hold -= splitDic[splitVal][tag] * math.log(splitDic[splitVal][tag] / splitDic[splitVal]["total"], 2)
                # divided by the total, but then when it is scaled to its fraction of all data that gets cancelled out
        entropy += hold
    entropy = hold / tSum
    return entropy




# returns a list of sets where the attribute at splitInd is all equal
def splitData(data, pList, indexes, splitInd):
    thing = data
    sets = []
    seen = {}
    nI = 0
    for ind in indexes:
        if data[ind][splitInd] not in seen:
            seen[data[ind][splitInd]] = nI
            sets.append(set())
            sets[nI].add(ind)
            nI += 1
        else:
            sets[seen[data[ind][splitInd]]].add(ind)
    maxInd = -1
    maxi = 0
    j = 0
    for s in sets:
        if len(s) > maxi:
            maxi = len(s)
            maxInd = j
        j += 1
    for poss in pList[splitInd]:
        if poss not in seen:
            seen[poss] = nI
            nI += 1
            sets.append(sets[maxInd].copy())

    #print(sets[2])
    #print(seen)
    return sets, seen


def weightedMode(data, inds, weights, card):
    counts = {}
    for ind in inds:
        if data[ind][card] not in counts.keys():
            counts[data[ind][card]] = weights[ind]
        else:
            counts[data[ind][card]] += weights[ind]
    mKey = data[ind][card]
    for key in counts:
        if counts[key] > counts[mKey]:
            mKey = key
    return mKey


def ID3(data, tree, count, card, indexes, depth, pastSplits, pList, weights):

    mini = card
    mInd = -1
    print("new stump")
    for i in range(0, card):  # this loops finds the minimum Entropy/Error/GI
        if i not in pastSplits:  # no choosing the same one
            hold = findEE(data, indexes, i, card, weights)  # card is also the index of the result
            print("i: ", i, " ", hold)
            if hold < mini:
                mInd = i
                mini = hold
    print(mini)
    tup = splitData(data, pList, indexes, mInd)
    #print("tup: ", tup)
    tree[mInd] = {}
    hold = pastSplits.copy()
    hold.add(mInd)
    for p in tup[1]:
        tree[mInd][p] = {}
        #ID3(data, tree[mInd][p], count, card, tup[0][tup[1][p]], depth + 1, hold, pList, weights)
        tree[mInd][p][card] = weightedMode(data, tup[0][tup[1][p]], weights, card)


def predict(instance, tree):
    #print(instance)
    ins = instance
    currI = list(tree.keys())[0]
    currD = tree
    attCount = len(instance) - 1
    #print(currD)
    while currI < attCount:
        hold = currI
        #print("hold: ",hold)
        currD = currD[currI]
        #print("currD: ", currD)
        currI = list(currD[ins[currI]].keys())[0]
        #print("in: ", instance[currI])
        #print("currI: ", currI)
        currD = currD[ins[hold]]
    #print(currD[currI])
    return currD[currI]


def parseData(File):
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
                if terms[i] == "unknown":
                    unknown.add(dataCount)
        else:
            for i in range(0, len(terms)):
                allList[i].append(terms[i])
                pList[i].add(terms[i])
                if terms[i][len(terms[i]) - 1].isdigit():
                    medList[i].append(int(terms[i]))
                if terms[i] == "unknown":
                    unknown.add(dataCount)

        data[dataCount] = terms.copy()
        indices.add(dataCount)
        dataCount += 1

    for i in range(0, len(medList)):
        if medList[i] is not None:
            medList[i] = median(medList[i])

    for i in range(0, attCount):
        unknownList.append(mode(allList[i]))

    for index in unknown:
        for i in range(0, attCount):
            if data[index][i] == "unknown":
                data[index][i] = unknownList[i]

    for key in data.keys():
        for i in range(0, len(medList)):
            if medList[i] is not None:
                pList[i] = {0, 1}
                if int(data[key][i]) > medList[i]:
                    data[key][i] = 1
                else:
                    data[key][i] = 0

    dataCount += 1
    f.close()
    return data, dataCount, medList, [data, dataCount, attCount, indices, pList]


def createStump(weights, arr):
    """
    :param weights: weights[i] is the weight of the data at index i
    :param arr: the array returned at parseData[3]
    :return: dTree for feeding into predict
    """
    dTree = {}
    ID3(arr[0], dTree, arr[1], arr[2], arr[3], 0, set(), arr[4], weights)
    # produces a deciscion tree in dTree where the first key will be the first index split, the second layer of keys will be
    # possible values of said split, the third will be the second index split, and so on. A leaf node is denoted by a
    # non-dict entry
    return dTree



"""
print("Type: ", TYPE)
for DEPTH in range(1, 17):





    #print("\n\ndata: ", data)
    #print("tree: ", dTree)

    # instance is a list representing a datapoint


    total = 0
    corr = 0
    f = open(TRAINING_FILE, 'r')
    for line in f:
        t = line.strip().split(',')
        #print("t: ", t)
        #print("pred: ", predict(t))
        #print("act: ", t[attCount])
        for i in range(0, len(medList)):
            if medList[i] is not None:
                if int(t[i]) >= medList[i]:
                    t[i] = 1
                else:
                    t[i] = 0
        if predict(t) == t[attCount]:
            corr += 1
        total += 1
    f.close()

    total2 = 0
    corr2 = 0
    f = open(TEST_FILE, 'r')
    for line in f:
        t = line.strip().split(',')
        #print("t: ", t)
        for i in range(0, len(medList)):
            if medList[i] is not None:
                if int(t[i]) >= medList[i]:
                    t[i] = 1
                else:
                    t[i] = 0
        if predict(t) == t[attCount]:
            corr2 += 1
        total2 += 1
    f.close()

    print()
    print()
    print("Depth: ", DEPTH)
    print("TRAINING TEST")
    print("Correct: ", corr)
    print("Total: ", total)
    print("Accuracy: ", corr/total)
    print()
    print("ACTUAL TEST")
    print("Correct: ", corr2)
    print("Total: ", total2)
    print("Accuracy: ", corr2/total2)"""
