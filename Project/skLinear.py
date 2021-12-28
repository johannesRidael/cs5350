from sklearn import svm
import math
import numpy as np
import Utilities as U

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



baeXes = []
extendeds = []
for i in range(len(exes)):
    baeXes.append(baeTerms(data[i]))
for i in range(len(baeXes)):
    baeXes[i].append(bayePredict(data[i]))

wise = np.array(wise)
#print(wise[wise == 1])
baeXes = np.array(baeXes)

lin = svm.SVC()
poly = svm.SVC(kernel='poly', degree=2)
poly3 = svm.SVC(kernel='poly', degree=3)
poly4 = svm.SVC(kernel='poly', degree=4)

lin.probability = True
poly.probability = True
poly3.probability = True
poly4.probability = True

lin.fit(baeXes, wise)
poly.fit(baeXes, wise)
poly3.fit(baeXes, wise)
poly4.fit(baeXes, wise)



pcor, p3cor, p4cor, lCor, tot = 0, 0, 0, 0, 0,
p = poly.predict(baeXes)
p3 = poly3.predict(baeXes)
p4 = poly4.predict(baeXes)
L = lin.predict(baeXes)

for i in range(len(exes)):
    if p[i] == wise[i]: pcor += 1
    if p3[i] == wise[i]: p3cor += 1
    if p4[i] == wise[i]: p4cor += 1
    if L[i] == wise[i]: lCor += 1
    tot += 1


print("p2 Acc: ", pcor/tot)
print("p3 Acc: ", p3cor / tot)
print("p4 Acc: ", p4cor / tot)
print("L Acc: ", lCor / tot)


tData, tc, tm, arr = U.parseTData(TestF, meds)

bTex = []
k = list(tData.keys())
for i in k:
    bTex.append(baeTerms(tData[i]))
for i in range(len(bTex)):
    bTex[i].append(np.product(bTex[i]))

g = poly4.predict_proba(bTex)

f = open('p4.csv', 'w')
f.write("ID,Prediction\n")
for i in range(len(g)):
    f.write(str(k[i]) + ',' + str(g[i][1]) + '\n')

