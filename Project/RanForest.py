import pandas as pd
import EnsembleLearning.RanFor as rf
import DecisionTree.RanTreesCreator as rt



TrainingF = "train_final.csv"
TestF = "test_final.csv"
TestRF = "RFoutput.csv"

def parseTData(File, meds):
    data = {}
    attCount = None
    f = open(File, 'r')
    dataCount = 0
    indices = set()
    pList = []
    medList = meds
    allList = []
    unknown = set()  # key is index in data, value is term index
    unknownList = []  # unknownList[i] is the mode of that attribute
    for line in f:
        terms = line.strip().split(',')
        if attCount is None:
            for i in range(1, len(terms)):
                pList.append(set())
                pList[i].add(terms[i])
            attCount = len(terms) - 1
               #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        else:
            for i in range(1, len(terms)):
                allList[i].append(terms[i])
                pList[i].add(terms[i])
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)

        data[dataCount] = terms[1:].copy()
        indices.add(dataCount)
        dataCount += 1
        inL = list(indices)
    for i in range(0, len(medList)):
        if medList[i] is not None:
            medList[i] = medList[i][rt.quickselect(inL, medList[i], int(len(medList[i]) / 3))], \
                         medList[i][rt.quickselect(inL, medList[i], int(2*len(medList[i])/3))]
    for key in data.keys():
        for i in range(0, len(medList)):
            if medList[i] is not None:
                pList[i] = {0, 1, 2}
                data[key][i] = rt.convToBin(data[key][i], medList[i])

    dataCount += 1
    f.close()
    return data, dataCount, medList, [data, dataCount, attCount, indices, pList]

def fracPred(t, treeL):
    summ = 0
    for tree in treeL:
        summ += rt.predict(t, tree)
    return summ / len(treeL)


data, count, meds, arr = rt.parseData(TrainingF)

treeL = rf.createFor(data, 150, arr)

testD, tc, tm, tar = parseTData(TestF, 1)

f = open(TestRF, 'w')
f.write("ID,Prediction")
for i in range(tc):
    f.write(str(i) + str(fracPred(data[i], treeL)) + '\n')
f.close()
