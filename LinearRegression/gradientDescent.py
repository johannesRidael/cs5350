import math
import random

import numpy
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

card = 7

def LMS(indexes, w: np.array, data):
    ret = 0
    for index in indexes:
        ret += (data[index]["y"] - w.T @ data[index]["x"]) ** 2
    return ret/2

def gradient(indexes, w: np.array, data):
    ret = np.zeros(w.shape)
    for index in indexes:
        #print(data[index])
        error = data[index]["y"] - (w.T @ data[index]["x"])
        for j in range(len(w)):
            ret[j] -= error * data[index]["x"][j]
    return ret

data = {}
attCount = None
f = open("train.csv", 'r')
dataCount = 0
indices = set()
x = []
y = []
j = 0
for line in f:
    x.append([])
    a = [1]
    terms = line.strip().split(',')
    for i in range(len(terms)):
        if i < len(terms) - 1:
            a.append(float(terms[i]))
    data[dataCount] = {}
    data[dataCount]["x"] = np.array(a)
    x[j] = a
    data[dataCount]["y"] = float(terms[card])
    y.append(float(terms[card]))
    indices.add(dataCount)
    dataCount += 1
    j+=1
x = numpy.array(x)
y = numpy.array(y)

gam = 0.015
w0 = np.zeros(card + 1)
last = np.inf
funs = []
ws = []
ws.append(w0)
funs.append(LMS(indices, w0, data))
i = 0
while True:
    grad = gradient(indices, ws[i], data)
    ws.append(ws[i] - gam*grad)
    fun = LMS(indices, ws[i+1], data)
    funs.append(fun)
    #print(funs[i+1])
    if LA.norm(ws[i] - ws[i+1], 1) < 1e-6:
        break
    last = fun
    i += 1
print(ws[len(ws) - 1])
print(fun)
print('\n************************************************************************')
plt.plot(funs)
plt.show()
gam = 0.0008
w0 = np.zeros(card + 1)
last = np.inf
funs = []
ws = []
ws.append(w0)
funs.append(LMS(indices, w0, data))

j=0
while True:
    i = random.randrange(dataCount)
    grad = gradient([i], ws[j], data)
    ws.append(ws[j] - gam * grad)
    fun = LMS(indices, ws[j+1], data)
    funs.append(fun)
    #print(fun)
    if LA.norm(ws[j] - ws[j+1], 1) < 1e-7:
        break
    j += 1
plt.plot(funs)
plt.show()
print(ws[len(ws) - 1])
print(fun)
w = LA.inv(x.T @ x) @ x.T
hold = w @ y.T
w = hold
print("*************************")
print(w)
print(LMS(indices, w, data))