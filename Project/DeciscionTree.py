import Utilities as u
from DecisionTree import NumTreesCreator as NT

TrainingF = "train_final.csv"
TestF = "test_final.csv"
TestP = "DToutput.csv"

data, dc, meds, arr = u.parseData(TrainingF)

tree = NT.createTree(arr[3], arr)

td, tc, tm, tarr = u.parseTData(TestF, meds)

f = open(TestP, 'w')
f.write("ID,Prediction\n")
for id in td:
    p = NT.predict(td[id], tree)
    f.write(id + ',' + str(p) + '\n')