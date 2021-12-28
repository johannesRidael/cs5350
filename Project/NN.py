import math
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import Utilities as u
import numpy as np

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(f'Using {device} device')
laArr = [250, 550, 850]
tol = 0.00000001

class NeuralNetwork(nn.Module):
    def __init__(self, numIn, layers, p=0.4):
        super(NeuralNetwork, self).__init__()
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(numIn)

        all_layers = []
        input_size = numIn

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.Mish())
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], 2))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return x


exes, outs, ataCount, indices, [dataCount, attCount, indices] = u.parseNumData('train_final.csv')

data, count, meds, arr = u.parseData('train_final.csv')

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
for i in range(len(exes)):
    baeXes.append(baeTerms(data[i]))
for i in range(len(baeXes)):
    baeXes[i].append(bayePredict(data[i]))


inTensors = torch.tensor(exes)  # baexes or exes
inTensors = inTensors.to(device)

ys = []
for y in outs:
    ys.append([0, 0])
    if y == 1: ys[len(ys) - 1][1] = float(1)
    else: ys[len(ys)-1][0] = float(1)

outTen = torch.tensor(ys)
outTen = outTen.to(device)

m = NeuralNetwork(len(inTensors[0]), laArr)
m.to(device)

print(m)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)

epochs = 40000
agLoss = []
i = 0
while i < 2 or math.fabs(agLoss[i-2] - agLoss[i-1]) > tol:
    y_pred = m(inTensors)
    lss = loss(y_pred, outTen)
    agLoss.append(lss)

    optimizer.zero_grad()
    lss.backward()
    optimizer.step()
    if i % 10000 == 0:
        print("Epoch: ", i, "\tLoss: ", lss)
    i+=1

guess = m(inTensors)
guess = guess.cpu()
guess = guess.detach().numpy()

cor, tot = 0, 0
for i in range(len(exes)):
    g = 1
    if guess[i][0] > guess[i][1]: g = -1
    if g == outs[i]: cor += 1
    tot += 1

print("Training Acc: ", cor/tot)


texes, tc, tindices = u.parseNumTData('test_final.csv')

tData, tc, tm, arr = u.parseTData('test_final.csv', meds)

bTex = []
k = list(tData.keys())
for i in k:
    bTex.append(baeTerms(tData[i]))
for i in range(len(bTex)):
    for j in range(len(bTex[i])):
        bTex[i][j] = float(bTex[i][j])
for i in range(len(bTex)):
    bTex[i].append(float(np.product(bTex[i])))


tesTensors = torch.tensor(texes)  # bTex or texes
tesTensors = tesTensors.to(device)

guess = m(tesTensors)
guess = guess.cpu()
guess = guess.detach().numpy()
f = open('NN.csv', 'w')
f.write("ID,Prediction\n")
for i in range(len(guess)):
    f.write(str(i+1) + ',' + str(guess[i][1] - guess[i][0]) + '\n') # guess[i][1] - guess[i][0]

agLoss = torch.tensor(agLoss)
agLoss = agLoss.detach().numpy()
f.close()
plt.plot(agLoss)
plt.show()
