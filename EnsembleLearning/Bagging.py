import math
import random

import DecisionTree.NumTreesCreator as trees
from random import randrange

CARD = 16
TR_F = "train.csv"
TE_F = "test.csv"


def predictBag(t, treeL):
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

def testBag(trees, data: dict):
    incorrect, total = 0, 0
    for key in data.keys():
        t = data[key]
        if predictBag(t, trees) != t[len(t) - 1]:
            incorrect += 1
        total += 1
    return incorrect, total

def createBag(data, T, arr, m = None):
    treeList = []
    if m is None: m = len(data)
    for i in range(1, T+1):
        indices = set()
        for j in range(len(data)):
            indices.add(randrange(len(data) - 1))
        treeList.append(trees.createTree(indices, arr))
    return treeList

def createBagL(data, T, inds, arr, m = None):
    treeList = []
    if m is None: m = len(data)
    for i in range(1, T+1):
        indices = set()
        for j in range(len(inds)):
            indices.add(random.choice(inds))
        treeList.append(trees.createTree(indices, arr))
    return treeList

#ErTeF = open('bagTe', 'w')
#ErTrF = open('bagTr', 'w')

#testD, testC, testM, testA = trees.parseData(TE_F)
#data, count, medians, arr = trees.parseData(TR_F)


#treeList = []

#for i in range(1, 501):
#    indices = set()
#    for j in range(count):
#        indices.add(randrange(count-1))
#    treeList.append(trees.createTree(indices, arr))
#    inc, tot = testBag(treeList, data)
#    ErTrF.write(str(i) + " Tree Training: Incorrect: " + str(inc) + " Total: " + str(tot) + " Error: " + str(inc/tot) + '\n')
#    inc, tot = testBag(treeList, testD)
#    ErTeF.write(str(i) + " Tree Training: Incorrect: " + str(inc) + " Total: " + str(tot) + " Error: " + str(inc / tot) + '\n')
