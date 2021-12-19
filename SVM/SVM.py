import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import random
import math
from scipy import optimize
import sklearn

C = 700 / 873  # 100, 500, 700
o = 100 #0.1, 0.5, 1, 5, 100

def to1(num):
    if num != 1: return -1
    return 1


def sloss(we, bee, x, y):
    loss = (we.T @ we) / 2 + C * max(0, 1 - y * (we.T @ x + bee))  # * dataCount)
    return loss


def tLoss(we, be):
    loss = (we.T @ we) / 2
    for i in range(len(exes)):
        x = exes[i]
        y = wise[i]
        loss = loss + C * max(0, 1 - y * (we.T @ x + be))
    return loss


def sciLoss(dw):
    ow = dw[:4]
    ob = dw[4]
    return tLoss(ow, ob)


def sGrad(we, b, x, y):
    m = max(0, 1 - y * (we.T @ x + b))

    g = []
    for i in range(len(we)):
        if m == 0:
            g.append(we[i])
        else:
            g.append(we[i] - C * y * x[i])

    # g = w.copy()

    # g = w - C * y * x
    g = np.array(g)
    return g


def beeGee(we, be, x, y):
    m = max(0, 1 - y * (we.T @ x + be))
    if m == 0:
        return 0
    else:
        return - C * y  # * dataCount


def gaussKern(x1, x2, c=o):
    dis = LA.norm(x1 - x2)
    dis = dis ** 2
    dis = dis / c
    return math.e ** (-dis)


def dual(alphas):
    ret = 0
    hold = 0

    # print(alphas)
    # print(M.shape)
    # print(alphas.shape)
    hold = 0.5 * alphas @ M @ alphas
    """
    for i in range(len(alphas)):
        if alphas[i] != 0:
            for j in range(len(alphas)):
                if alphas[j] != 0:
                    ret += data[i]["y"] * data[j]["y"] * alphas[i] * alphas[j] * data[i]["x"].T @ data[j]["x"]
            hold += alphas[i]
    ret = (ret/2) - hold
    """
    return -sum(alphas) + hold

def jac(alphas):
    return np.dot(alphas.T, M) - np.ones_like(alphas)


def extDual(alp): # for test
    return -(alp.sum() - 0.5 * np.dot(alp.T, np.dot(M, alp)))  # negative of (7.32)


def kerDual(alphas):
    ret = 0
    hold = 0
    for i in range(len(alphas)):
        for j in range(len(alphas)):
            ret += data[i]["y"] * data[j]["y"] * alphas[i] * alphas[j] * gaussKern(data[i]["x"].T, data[j]["x"])
        hold += alphas[i]
    ret = (ret / 2) - hold
    return ret


def gaussPred(alphas, x):
    s = 0
    for i in infInds:
        s += alphas[i] * wise[i] * gaussKern(exes[i], x)
    return s


def kPred(alphas, kernel=None):
    if kernel == "Gaussian":
        return None
    else:
        return None


def getB(alphas, kernel=None):
    s = 0
    for i in infInds:
        bi = wise[i] - wDual.T @ exes[i]
        s += bi
    be = s / len(infInds)
    return be


def con1(alphas):
    boo = np.all(alphas > 0) and np.all(alphas < C)
    if boo:
        return 1
    else:
        return -1


def con2(alphas):
    s = 0
    for i in range(len(alphas)):
        s += alphas[i] * data[i]["y"]
    if s == 0:
        return 1
    return -1


def aK(x1, x2, kernel=None):
    if kernel == "Gaussian":
        return gaussKern(x1, x2)
    else:
        return x1.dot(x2)


card = 4
data = {}
attCount = None
f = open("./bank-note/train.csv", 'r')
dataCount = 0
indices = set()
exes = []
wise = []
j = 0
for line in f:
    exes.append([])
    a = []
    terms = line.strip().split(',')
    for i in range(len(terms)):
        if i < len(terms) - 1:
            a.append(float(terms[i]))
    exes[j] = np.array(a)
    data[dataCount] = {}
    data[dataCount]["x"] = np.array(a)
    #exes[j] = a
    data[dataCount]["y"] = to1(int(terms[card]))
    wise.append(to1(int(terms[card])))
    indices.add(dataCount)
    dataCount += 1
    j += 1
exes = np.array(exes)
wise = np.array(wise)

K = np.empty((dataCount, dataCount))
for i in range(dataCount):
    for j in range(dataCount):
        K[i, j] = aK(exes[i], exes[j], kernel="Gaussian")
# print(K)
# print(len(wise))
# print("K: ", K.shape)
# h = wise @ K
# print("h:", h.shape)
M = wise * K * wise.T
# print("M: ", M.shape)
# Q = wise @ K @ wise[:, np.newaxis]
# print("Q: ", Q.shape)


bnds = [(0, C)] * dataCount
constraints = (#{'type': 'ineq', 'fun': lambda ays: conMat1 - np.dot(conMat2, ays), 'jac': lambda ays: -conMat2},
               {'type': 'eq', 'fun': lambda x: np.dot(x, wise), 'jac': lambda x: wise})


#a0 = np.random.rand(dataCount)
#print(dual(a0))
#print(extDual(a0))

"""
for d in data.keys():
    if data[d]["y"] == 1:
        plt.scatter(data[d]["x"][0], data[d]["x"][1], marker="+", c="green")
    else:
        plt.scatter(data[d]["x"][0], data[d]["x"][1], marker="_", c="red")
plt.show()

for d in data.keys():
    if data[d]["y"] == 1:
        plt.scatter(data[d]["x"][2], data[d]["x"][3], marker="+", c="green")
    else:
        plt.scatter(data[d]["x"][2], data[d]["x"][3], marker="_", c="red")

plt.show()
"""

out = optimize.minimize(sciLoss, np.zeros(5), method='Nelder-Mead')
print(out)

optW = out.x[:4].copy()
optB = out.x[4].copy()
"""
gam = 1
gam0 = 1
a = 1.1
w0 = np.zeros(card)
last = np.inf
funs = []
ws = [w0]
bees = [0]
# funs.append(tLoss(w0, data))
inds = list(indices)
updates = []

j = 0
wc = 0
while j < 100:
    random.shuffle(inds)
    u = 0
    for i in inds:
        grad = sGrad(ws[wc], bees[wc], exes[i], wise[i])
        bg = beeGee(ws[wc], bees[wc], exes[i], wise[i])
        if wise[i] * (ws[wc] @ exes[i] + bees[wc]) <= 1:
            ws.append(ws[wc] - gam * grad)
            bees.append(bees[wc] - gam * bg)
        else:
            ws.append((1 - gam) * ws[wc])
        bees.append(bees[wc] - gam * bg)
        fun = tLoss(ws[wc + 1], bees[wc + 1])
        funs.append(fun)
        # print(fun)
        wc += 1
    gam = gam0 / (1+ (gam0/a) * j)  # /(1+j) or /(1+ (gam0/a) * j)
    j += 1
plt.plot(funs)
# plt.show()
print(ws[len(ws) - 1])
w = ws[len(ws) - 1]
b = bees[len(bees) - 1]
print(fun)
print("*************************")
print(tLoss(ws[wc], bees[wc]))
"""
a0 = np.random.rand(dataCount)
# print("a0: ", a0)
out = optimize.minimize(dual, a0, bounds=bnds, constraints=constraints, method="SLSQP", jac=jac)
print(out.nit)
aOpt = out.x
# optimal alphas
print(np.dot(wise, aOpt))
for i in range(len(aOpt)):
    if np.isclose(aOpt[i], 0):
        aOpt[i] = 0

infInds = []
mInds = []
for i in range(len(aOpt)):
    if aOpt[i] > 0:
        infInds.append(i)
        if aOpt[i] < C:
            mInds.append(i)

print("Support Vectors: ", len(infInds))
print("Support Inds: ", infInds)

wDual = np.array([0, 0, 0, 0])
for i in infInds:
    wDual = wDual + aOpt[i] * wise[i].copy() * exes[i].copy()
wDual = wDual / LA.norm(wDual)  # just to shrink our numbers a bit, doesn't actually change our line.
bDual = getB(aOpt)

print("wDual: ", wDual)
print("bDual: ", bDual)
print("Dual Loss: ", tLoss(wDual, bDual))

cor, tot = 0, 0
oCor, dcor = 0, 0
for i in indices:
    #if wise[i] * (w @ exes[i] + b) > 0:
    #    cor += 1
    if wise[i] * (optW @ exes[i] + optB) > 0:
        oCor += 1
    if wise[i] * gaussPred(aOpt, exes[i]) > 0: #(wDual.T @ exes[i] + bDual) > 0:
        dcor += 1
    tot += 1

data = {}
attCount = None
f = open("./bank-note/test.csv", 'r')
dataCount = 0
indices = set()
texes = []
twise = []
j = 0
for line in f:
    texes.append([])
    a = []
    terms = line.strip().split(',')
    for i in range(len(terms)):
        if i < len(terms) - 1:
            a.append(float(terms[i]))
    data[dataCount] = {}
    data[dataCount]["x"] = np.array(a)
    texes[j] = a
    data[dataCount]["y"] = to1(int(terms[card]))
    twise.append(to1(float(terms[card])))
    indices.add(dataCount)
    dataCount += 1
    j += 1

tcor, ttot = 0, 0
otCor, dtcor = 0, 0
for i in indices:
    #if twise[i] * (w @ texes[i] + b) > 0:
    #    tcor += 1
    if twise[i] * (optW @ texes[i] + optB) > 0:
        otCor += 1
    if twise[i] * gaussPred(aOpt, texes[i]) > 0: #(wDual.T @ texes[i] + bDual) > 0:
        dtcor += 1
    ttot += 1

#print("P Training Acc:\t", cor / tot)
#print("P Testing Acc:\t", tcor / ttot)
print("Opt train acc:\t", oCor / tot)
print("opt test acc:\t", otCor / ttot)

# print(cor, tot)
print("Dual Training Acc:\t", dcor / tot)
# print(tcor, ttot)
print("Dual Testing Acc:\t", dtcor / ttot)

plt.show()
