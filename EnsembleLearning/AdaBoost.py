import math

import DecisionTree.StumpyTrees as stumps
CARD = 16
TRAININGFILE = "train.csv"
TESTFILE = "test.csv"


def testTree(tree, data: dict):
    incorrect, total = 0, 0
    wrongInd = set()
    for key in data.keys():
        t = data[key]
        prediction = stumps.predict(t, tree)
        if prediction != t[len(data[key]) - 1]:
            incorrect += 1
            wrongInd.add(key)
        total += 1
    return incorrect, total, wrongInd

def wTestTree(tree, data: dict, weights):
    incorrect, total = 0, 0
    for key in data.keys():
        t = data[key]
        prediction = stumps.predict(t, tree)
        if prediction != t[len(data[key]) - 1]:
            incorrect += weights[key]
        total += weights[key]
    return incorrect / total


def predictADA(t, trees, alphas):
    summ = 0
    for j in range(len(alphas)):
        if stumps.predict(t, trees[j]) == "yes":
            summ += 1 * alphas[j]
        else:
            summ -= 1 * alphas[j]
    if summ >= 0:
        return "yes"
    else:
        return "no"

def testADA(trees, alphas, data: dict):
    incorrect, total = 0, 0
    wrongInd = set()
    for key in data.keys():
        t = data[key]
        if predictADA(t, trees, alphas) != t[len(data[key]) - 1]:
            incorrect += 1
            wrongInd.add(key)
        total += 1
    return incorrect, total, wrongInd



stumpsTRstr = open('stumpTr', 'w')
stumpsTEstr = open('stumpTe', 'w')
adaTRstr = open('adaTr', 'w')
adaTEstr = open('adaTe', 'w')
wF  = open('wF', 'w')
aF = open('aF', 'w')
tF = open('tF', 'w')
weights = []
alphas = []
testData, testCount, testMed, testArr = stumps.parseData(TESTFILE)
data, count, medians, arr = stumps.parseData(TRAININGFILE)
#print(medians)
count = len(data)
for i in range(count):
    weights.append(1/count)
wF.write(str(weights) + '\n')
trees = []
trees.append(stumps.createStump(weights, arr).copy())
inc, tot, wrongInds = testTree(trees[0], data)
e = wTestTree(trees[0], data, weights)
stumpsTRstr.write("Tree 1 Training Error: " + str(e) + '\n')
#print(trees[0])
alpha = 0.5 * math.log((1-e)/(e))
aF.write(str(alpha) + '\n')
alphas.append(alpha)
err = wTestTree(trees[0], testData, weights)
stumpsTEstr.write("Tree 1 Test Error: " + str(err) + '\n')
inc, tot, wInds = testADA(trees, alphas, data)
adaTRstr.write("1 Tree Training: Incorrect: " + str(inc) + " Total: " + str(tot) + " Error: " + str(inc/tot) + '\n')
inc, tot, wInds = testADA(trees, alphas, testData)
adaTEstr.write("1 Tree Test: Incorrect: " + str(inc) + " Total: " + str(tot) + " Error: " + str(inc/tot) + '\n')
for it in range(1, 501):
    wsum = 0
    for i in range(count):
        if i in wrongInds:
            weights[i] = weights[i] * math.exp(alpha)
        else:
            weights[i] = weights[i] * math.exp(-alpha)
        wsum += weights[i]
    for i in range(count):
        weights[i] = weights[i] / wsum
    #print("wsum: ", math.fsum(weights))
    wF.write(str(weights) + '\n')
    trees.append(stumps.createStump(weights, arr).copy())
    inc, tot, wrongInds = testTree(trees[it], data)
    #print(trees[it])
    e = wTestTree(trees[it], data, weights)
    stumpsTRstr.write("Tree " + str(it) + " Training Error: " + str(e) + '\n')
    alpha = 0.5 * math.log((1 - e) / (e))
    alphas.append(alpha)
    aF.write(str(alpha) + '\n')
    err = wTestTree(trees[it], testData, weights)
    stumpsTEstr.write("Tree" + str(it) + "Test Error: " + str(err) + '\n')
    inc, tot, wInds = testADA(trees, alphas, data)
    adaTRstr.write(str(it) + " Tree Training: Incorrect: " + str(inc) + " Total: " + str(tot) + " Error: " + str(inc / tot) + '\n')
    inc, tot, wInds = testADA(trees, alphas, testData)
    adaTEstr.write(str(it) + " Tree Test: Incorrect: " + str(inc) + " Total: " + str(tot) + " Error: " + str(inc / tot) + '\n')

print("Finished")

