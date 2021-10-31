import random
import sys
import numpy as np
import DecisionTree.RanTreesCreator as trees
from statistics import mean

def convBin(s):
    if s == "yes": return 1
    else: return 0

def createRanL(data, T, inds, arr, m = None):
    treeList = []
    if m is None: m = len(data)
    for i in range(1, T+1):
        indices = set()
        for j in range(len(inds)):
            indices.add(random.choice(inds))
        treeList.append(trees.createTree(indices, 6, arr))
    return treeList

def predictRan(t, treeL):
    summ = 0
    for tree in treeL:
        if trees.predict(t, tree)  == "yes":
            summ += 1
        else:
            summ -=1
    if summ >= 0:
        return "yes"
    else:
        return "no"

indices = np.arange(0, 5000)
pInd = set()

data, count, medians, arr = trees.parseData("train.csv")
tData, testC, testM, tArr = trees.parseData("test.csv")
bagList = []
for run in range(100):
    print(run)
    random.shuffle(indices)
    try:
        bagList.append(createRanL(data, 100, indices[:1000], arr))
    except:
        break
    #print(sys.getsizeof(bagList[run]))
    #for bag in bagList[run]:
        #print(sys.getsizeof(bag))

#dmean = 0
#for key in tData.keys():
#    if tData[key][len(tData[key]) - 1] == "yes":
#        dmean += 1

#dmean = dmean/testC

tBiasList = []
bBiasList = []
tVarList = []
bVarList = []
tMSEList = []
bMSEList = []
i = 0
for key in tData.keys():
    avgb = 0
    avgt = 0
    c = 0
    tpredList = []
    bpredList = []
    for bag in bagList:
        tG = convBin(trees.predict(tData[key], bag[0]))
        bG = convBin(predictRan(tData[key], bag))
        avgt += tG
        avgb += bG
        tpredList.append(tG)
        bpredList.append(bG)
        c += 1
    tBiasList.append((convBin(tData[key][len(tData[key]) - 1]) - avgt / c) ** 2)
    bBiasList.append((convBin(tData[key][len(tData[key]) - 1]) - avgb / c) ** 2)
    tVar = 0
    co = 0
    for guess in tpredList:
        tVar += (guess - avgt / c) ** 2
        co += 1
    tVarList.append(tVar / co)
    bVar = 0
    co = 0
    for guess in bpredList:
        bVar += (guess - avgb / c) ** 2
        co += 1
    bVarList.append(bVar / co)

    tMSEList.append(tVarList[i] + tBiasList[i])
    bMSEList.append(bVarList[i] + bBiasList[i])

    i += 1

tBias = mean(tBiasList)
tVar = mean(tVarList)
bBias = mean(bBiasList)
bVar = mean(bVarList)

print("Type\tBias\tVariance\tMSE")
print("1Tree\t", tBias, '\t', tVar, '\t', tBias + tVar)
print("Ran\t", bBias, '\t', bVar, '\t', bBias + bVar)