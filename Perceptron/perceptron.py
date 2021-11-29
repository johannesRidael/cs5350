import random
from sklearn.linear_model import Perceptron
import numpy as np

EC = 10
DIM = 10
lr = 0.5

def convToPM(f):
    if f >= 0:
        return 1
    else:
        return -1

def votePred(wcbs, x):
    count = 0
    for wcb in wcbs:
        w = wcb[0]
        c = wcb[1]
        b = wcb[2]
        count += c * convToPM(w @ x + b)
    if count > 0:
        return 1
    else: return 0

def avgPred(wcbs, x):
    count = 0
    for wcb in wcbs:
        w = wcb[0]
        c = wcb[1]
        b = wcb[2]
        count += c * (w @ x + b)
    if count > 0:
        return 1
    else: return 0

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
            #for j in range(0, len(terms)):
                #pList.append(set())
                #pList[i].add(terms[i])
                #medList.append([])
                #allList.append([])
                #allList[i].append(terms[i])
               # if terms[i][len(terms[i]) - 1].isdigit():
                #    medList[i].append(int(terms[i]))
               # else:
               #     medList[i] = None
            attCount = len(terms) - 1
            #if not testD: medList[len(medList) - 1] = None # don't want to convert our result to discrete even if we can
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        #else:
            #for i in range(0, len(terms)):
                #allList[i].append(terms[i])
                #pList[i].add(terms[i])
                #if medList[i] is not None and terms[i][len(terms[i]) - 1].isdigit():
                #    medList[i].append(int(terms[i]))
                #else: medList[i] = None
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        for index in range(len(terms)):
            terms[index] = float(terms[index])
        data[dataCount] = np.array(terms)
        indices.add(dataCount)
        dataCount += 1
        inL = list(indices)
    dataCount += 1
    f.close()
    return data, dataCount, attCount, [data, dataCount, attCount, indices, pList]

data, dc, dim, arr = parseData("./bank-note/train.csv")
inds = list(arr[3])

w = np.zeros(dim)
a = np.zeros(dim)
wcbL = []

b = 0
c = 0
exes = []
wise = []

for i in range(EC):
    print("Epoch: ", i)
    random.shuffle(inds)
    for ind in inds:
        #tlr = lr/ (i+1))
        tlr = lr
        x = data[ind][:dim]
        y = data[ind][dim]
        if i == 0:
            exes.append(x)
            wise.append(y)
        guess = w @ x + b
        if guess <= 0: guess = -1
        else: guess = 1
        if guess == -1 and y == 1:
            wcbL.append((w.copy(), c, b))
            w = w + tlr*x
            b = b + tlr
            c = 1
        elif guess == 1 and y == 0:
            wcbL.append((w.copy(), c, b))
            w = w - tlr*x
            b = b - tlr
            c = 1
        else:
            c += 1
        a = a + w

a = a/dc
cor = 0
tot = 0

vc = 0
ac = 0
a2c = 0

for ind in inds:
    x = data[ind][:dim]
    guess = w @ x + b
    if guess <= 0: guess = 0
    else: guess = 1
    if guess == data[ind][dim]:
        cor += 1
    guess = a @ x + b
    if guess <= 0:
        guess = 0
    else:
        guess = 1
    if guess == data[ind][dim]:
        ac += 1
    if votePred(wcbL, x) == data[ind][dim]:
        vc += 1
    if avgPred(wcbL, x) == data[ind][dim]:
        a2c += 1

    tot +=1

#print(w, b)
for wcb in wcbL:
    print(wcb[0], "&", wcb[2], "&", wcb[1], "\\\\")
print("Normal Train Accuracy: ", cor/tot)
print("Vote Train Accuracy: ", vc/tot)
print("Avg Train Accuracy: ", a2c/tot)
#print("avg2 Train Accuracy: ", a2c/tot)
print()
cor = 0
tot = 0
vc = 0
ac = 0
a2c = 0
f = open("./bank-note/test.csv")
for line in f:
    x = line.split(',')[:dim]
    for i in range(len(x)):
        x[i] = float(x[i])
    x = np.array(x)
    y = float(line.split(',')[dim])
    guess = w @ x + b
    if guess <= 0:
        guess = 0
    else:
        guess = 1
    if guess == y:
        cor += 1
    guess = a @ x + b
    if guess <= 0:
        guess = 0
    else:
        guess = 1
    if guess == y:
        ac += 1
    if votePred(wcbL, x) == y:
        vc += 1
    if avgPred(wcbL, x) == y:
        a2c += 1
    tot += 1


print("Normal Test Accuracy: ", cor/tot)
print("Vote Test Accuracy: ", vc/tot)
print("Avg Test Accuracy: ", a2c/tot)
#print("avg2 Test Accuracy: ", a2c/tot)

#cmp = Perceptron()
#cmp.fit(exes, wise)
#print(cmp.score(exes, wise))
#print()
#print(cmp.get_params())