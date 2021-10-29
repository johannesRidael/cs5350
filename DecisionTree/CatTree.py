import math
import sys
from statistics import mode

#DEPTH = 1
#TRAINING_FILE = "train.csv"
#TEST_FILE = "test.csv"
#TYPE = "GI" # EE, ME, GI

DEPTH = sys.argv[1]
TRAINING_FILE = sys.argv[2]
TEST_FILE = sys.argv[3]
TYPE = sys.argv[4]

def findEE(indexes, splitInd, tagInd):
    splitDic = {}
    for index in indexes:
        if data[index][splitInd] not in splitDic:
            tempDic = {}
            tempDic[data[index][tagInd]] = 1
            splitDic[data[index][splitInd]] = tempDic
            splitDic[data[index][splitInd]]["total"] = 1
        elif data[index][tagInd] not in splitDic[data[index][splitInd]]:
            splitDic[data[index][splitInd]][data[index][tagInd]] = 1
            splitDic[data[index][splitInd]]["total"] += 1
        else:
            splitDic[data[index][splitInd]][data[index][tagInd]] += 1
            splitDic[data[index][splitInd]]["total"] += 1
        # tagDic[x][y] now contains a count of that result with that data

    entropy = 0
    tSum = 0
    for splitVal in splitDic:
        hold = 0
        for tag in splitDic[splitVal]:
            if tag != "total":
                tSum += splitDic[splitVal][tag]
                hold -= splitDic[splitVal][tag] * math.log(splitDic[splitVal][tag] / splitDic[splitVal]["total"], 2)
        entropy += hold
    entropy = entropy / tSum
    return entropy


def findME(indexes, splitInd, tagInd):
    splitDic = {}
    #print(len(indexes))
    for index in indexes:
        if data[index][splitInd] not in splitDic:
            tempDic = {}
            tempDic[data[index][tagInd]] = 1
            splitDic[data[index][splitInd]] = tempDic
            splitDic[data[index][splitInd]]["total"] = 1
        elif data[index][tagInd] not in splitDic[data[index][splitInd]]:
            splitDic[data[index][splitInd]][data[index][tagInd]] = 1
            splitDic[data[index][splitInd]]["total"] += 1
        else:
            splitDic[data[index][splitInd]][data[index][tagInd]] += 1
            splitDic[data[index][splitInd]]["total"] += 1
        # tagDic[x][y] now contains a count of that result with that data
    #print(splitDic)

    totalWrong = 0
    for splitVal in splitDic:
        maxi = 0
        for tag in splitDic[splitVal]:
            if tag != "total":
                if maxi < splitDic[splitVal][tag]:
                    totalWrong += maxi
                    maxi = splitDic[splitVal][tag]
                else:
                    totalWrong += splitDic[splitVal][tag]
    #print(totalWrong)
    return totalWrong  # normally this would be scaled but the math all cancels out anyway


def findGI(indexes, splitInd, tagInd):
    splitDic = {}
    for index in indexes:
        if data[index][splitInd] not in splitDic:
            tempDic = {}
            tempDic[data[index][tagInd]] = 1
            splitDic[data[index][splitInd]] = tempDic
            splitDic[data[index][splitInd]]["total"] = 1
        elif data[index][tagInd] not in splitDic[data[index][splitInd]]:
            splitDic[data[index][splitInd]][data[index][tagInd]] = 1
            splitDic[data[index][splitInd]]["total"] += 1
        else:
            splitDic[data[index][splitInd]][data[index][tagInd]] += 1
            splitDic[data[index][splitInd]]["total"] += 1
        # tagDic[x][y] now contains a count of that result with that data

    GI = 0
    tSum = 0
    for splitVal in splitDic:
        hold = 0
        for tag in splitDic[splitVal]:
            if tag != "total":
                tSum += splitDic[splitVal][tag]
                hold -= (splitDic[splitVal][tag] / splitDic[splitVal]["total"]) ** 2
        GI += splitDic[splitVal]["total"] * (1 - hold)
    GI = hold / tSum
    return GI


# returns a list of sets where the attribute at splitInd is all equal
def splitData(indexes, splitInd):
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


def ID3(tree, count, card, indexes, depth, pastSplits):

    bf = True
    val = None
    for ind in indexes:
        if val == None:
            val = data[ind][card]
        else:
            if val != data[ind][card]:
                bf = False
                break
    if bf:
        tree[card] = val # base case, all the values match so our tree ends here.
        return

    # if we have capped out our depth, create a leaf of the most occurring element
    if depth == DEPTH:
        #print("indexes: ", indexes)
        tags = []
        for ind in indexes:
            tags.append(data[ind][card])
        #print("mode: ", mode(tags))
        tree[card] = mode(tags)
        #print("tree: ", tree)

        return



    mini = dataCount
    mInd = -1
    for i in range(0, card):  # this loops finds the minimum Entropy/Error/GI
        if i not in pastSplits: #no choosing the same one
            if TYPE == "ME": hold = findME(indexes, i, card)  # card is also the index of the result
            elif TYPE == "GI": hold = findGI(indexes, i, card)  # card is also the index of the result
            else: hold = findEE(indexes, i, card)  # card is also the index of the result
            if hold < mini:
                mInd = i
                mini = hold
    tup = splitData(indexes, mInd)
    #print("tup: ", tup)
    tree[mInd] = {}
    hold = pastSplits.copy()
    hold.add(mInd)
    for p in tup[1]:
        tree[mInd][p] = {}
        ID3(tree[mInd][p], count, card, tup[0][tup[1][p]], depth + 1, hold)
#    for b in tup[0]:  # b is a set of indices
#        #print("b: ", b)
#        h = b.pop()
#        #print("h: ", h)
#        b.add(h)
#        #print("b: ", b)
#        ID3(tree[mInd][data[h][mInd]], count, card, b, depth + 1)


data = {}
attCount = None
f = open(TRAINING_FILE, 'r')
dataCount = 0
indices = set()
pList = []
for line in f:
    terms = line.strip().split(',')
    if attCount == None:
        for i in range(0, len(terms)):
            pList.append(set())
            pList[i].add(terms[i])
    else:
        for i in range(0, len(terms)):
            pList[i].add(terms[i])
    attCount = len(terms) - 1
    data[dataCount] = terms.copy()
    indices.add(dataCount)
    dataCount += 1
dataCount += 1
f.close()

for DEPTH in range(1, 7):
    dTree = {}

    ID3(dTree, dataCount, attCount, indices, 0, set())
    # produces a deciscion tree in dTree where the first key will be the first index split, the second layer of keys will be
    # possible values of said split, the third will be the second index split, and so on. A leaf node is denoted by a
    # non-dict entry

    #print("\n\ndata: ", data)
    #print("tree: ", dTree)

    # instance is a list representing a datapoint
    def predict(instance):
        #print(instance)
        currI = list(dTree.keys())[0]
        currD = dTree
        #print(currD)
        while currI < attCount:
            hold = currI
            #print("hold: ",hold)
            currD = currD[currI]
            #print("currD: ", currD)
            #if instance[currI] not in currD:
            #    return 'unacc'
            currI = list(currD[instance[currI]].keys())[0]
            #print("in: ", instance[currI])
            #print("currI: ", currI)
            currD = currD[instance[hold]]
        #print(currD[currI])
        return currD[currI]

    total = 0
    corr = 0
    f = open(TRAINING_FILE, 'r')
    for line in f:
        t = line.strip().split(',')
        #print("t: ", t)
        #print("pred: ", predict(t))
        #print("act: ", t[attCount])
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
    print("Accuracy: ", corr2/total2)
