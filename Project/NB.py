import Utilities as u

TrainingF = "train_final.csv"
TestF = "test_final.csv"
TestP = "NBoutput.csv"

data, count, meds, arr = u.parseData(TrainingF)

card = len(meds) - 1
dicList = []
for i in range(card):
    dicList.append({})

totals = {"total527": 0, "1": 0, "0": 0}

for id in data.keys():
    for i in range(card):
        if data[id][i] not in dicList[i]:
            dicList[i][data[id][i]] = {}
            dicList[i][data[id][i]]["total527"] = 0
            dicList[i][data[id][i]]["0"] = 0
            dicList[i][data[id][i]]["1"] = 0   # dicList[index][variable][value] stores a count of that
        dicList[i][data[id][i]]["total527"] += 1
        dicList[i][data[id][i]][data[id][card]] += 1
    totals[data[id][card]] += 1
    totals["total527"] += 1

pYes = totals['1'] / totals["total527"]
pNo = totals['0'] / totals['total527']

def predict(term):
    pos = 1
    neg = 1
    for j in range(card):
        try:
            pos *= (dicList[j][term[j]]["1"] / totals['1'])  # / dicList[j][term[j]]["total527"]
            neg *= (dicList[j][term[j]]["0"] / totals['0'])  # dicList[j][term[j]]["total527"]
        except KeyError:
            print(term[j], " was not in the training data, did not modify probability\n")
    pt = pos * pYes
    nt = neg * pNo
    return pt - nt

tData, tc, tm, arr = u.parseTData(TestF, meds)

f = open(TestP, 'w')
f.write("ID,Prediction\n")
for id in tData:
    p = predict(tData[id])
    f.write(id + ',' + str(p) + '\n')
