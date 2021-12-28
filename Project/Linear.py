import math

import Utilities as U
import numpy as np
from SVM import SVM
from scipy.optimize import minimize
from sklearn import svm

C = 0.9
o = 1

TrainingF = "train_final.csv"
TestF = "test_final.csv"

exes, wise, dataCount, indices, arr = U.parseNumData(TrainingF)

#print(wise[wise == 1])

data, count, meds, arr = U.parseData(TrainingF)

card = len(meds) - 1
dicList = []
for i in range(card):
    dicList.append({})

for id in data.keys():
    for i in range(card):
        if data[id][i] not in dicList[i]:
            dicList[i][data[id][i]] = {}
            dicList[i][data[id][i]]["total527"] = 0
            dicList[i][data[id][i]]["0"] = 0
            dicList[i][data[id][i]]["1"] = 0   # dicList[index][variable][value] stores a count of that
        dicList[i][data[id][i]]["total527"] += 1
        dicList[i][data[id][i]][data[id][card]] += 1

def bayePredict(term):
    pos = 1
    neg = 1
    for j in range(card):
        try:
            pos *= (dicList[j][term[j]]["1"] / dicList[j][term[j]]["total527"])
            neg *= (dicList[j][term[j]]["0"] / dicList[j][term[j]]["total527"])
        except KeyError:
            print(term[j], " was not in the training data, did not modify probability\n")
    return pos / (pos + neg)

def baeTerms(catTerms):
    addedT = []
    for j in range(card):
        try:
            pos = (dicList[j][catTerms[j]]["1"] / dicList[j][catTerms[j]]["total527"])
        except:
            print(catTerms[j], " was not in the training data, did not modify probability")
            pos = 0  # try different values here
        addedT.append(pos)
    return addedT

def gaussTerm(ex):
    h = 1 / (math.sqrt(2 * math.pi) * o) ** len(ex)
    return h * math.e ** (-(np.linalg.norm(ex) ** 2) / (2 * o ** 2))


baeXes = []
extendeds = []
for i in range(len(exes)):
    baeXes.append(baeTerms(data[i]))
    extendeds.append(exes[i] + baeXes[i])

for i in range(len(exes)):
    exes[i].append(gaussTerm(exes[i]))
    baeXes[i].append(gaussTerm(baeXes[i]))
    extendeds[i].append(gaussTerm(extendeds[i]))

exes = np.array(exes)
wise = np.array(wise)
#print(wise[wise == 1])
baeXes = np.array(baeXes)
extendeds = np.array(extendeds)



out1 = minimize(SVM.sciLoss, np.zeros(len(exes[0]) + 1), args=(exes, wise, C), method='Nelder-Mead')
out2 = minimize(SVM.sciLoss, np.zeros(len(baeXes[0]) + 1), args=(baeXes, wise, C), method='Nelder-Mead')
out3 = minimize(SVM.sciLoss, np.zeros(len(extendeds[0]) + 1), args=(extendeds, wise, C), method='Nelder-Mead')

xW = out1.x[:len(out1.x) - 1]
xB = out1.x[len(out1.x) - 1]
bW = out2.x[:len(out2.x) - 1]
bB = out2.x[len(out2.x) - 1]
exW = out3.x[:len(out3.x) - 1]
exB = out3.x[len(out3.x) - 1]



texes, tc, tindices = U.parseNumTData(TestF)

tData, tc, tm, arr = U.parseTData(TestF, meds)

bTex = []
exTex = []
for i in range(len(texes)):
    bTex.append(baeTerms(tData[str(i + 1)]))
    exTex.append(texes[i] + bTex[i])



guesses = []
bGuesses = []
exGuesses = []


for idee in tData:
    a = xW @ texes[int(idee)-1] + xB
    b = bW @ bTex[int(idee)-1] + bB
    c = exW @ exTex[int(idee)-1] + exB
    guesses.append((idee, xW @ texes[int(idee)-1] + xB))
    bGuesses.append((idee, bW @ bTex[int(idee)-1] + bB))
    exGuesses.append((idee, exW @ exTex[int(idee)-1] + exB))

bf = open('JustX.csv', 'w')
pf = open('JustP.csv', 'w')
ef = open('extended.csv', 'w')

bf.write("ID,Prediction\n")
pf.write("ID,Prediction\n")
ef.write("ID,Prediction\n")

for i in range(len(guesses)):
    bf.write(guesses[i][0] + ',' + str(guesses[i][1]) + '\n')
    pf.write(bGuesses[i][0] + ',' + str(bGuesses[i][1]) + '\n')
    ef.write(exGuesses[i][0] + ',' + str(exGuesses[i][1]) + '\n')


xcor, bcor, ecor, tot = 0, 0, 0, 0
for i in range(len(exes)):
    a = wise[i] * (xW @ exes[i] + xB)
    b = wise[i] * (bW @ baeXes[i] + bB)
    c = wise[i] * (exW @ extendeds[i] + exB)
    if wise[i] * (xW @ exes[i] + xB) > 0: xcor += 1
    if wise[i] * (bW @ baeXes[i] + bB) > 0: bcor += 1
    if wise[i] * (exW @ extendeds[i] + exB) > 0: ecor += 1
    tot += 1

print("Base Acc: ", xcor/tot)
print("Bays Acc: ", bcor/tot)
print("Ext Acc: ", ecor/tot)
