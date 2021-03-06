from sklearn.naive_bayes import GaussianNB
from  sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
import Utilities as u

TrainingF = "train_final.csv"
TestF = "test_final.csv"
TestP = "SNBoutput.csv"

convertedIndices = [0,
                    {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5,
                     'Without-pay': 6, 'Never-worked': 7, '?': 8},
                    0,
                    {'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5, 'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12, 'Doctorate': 13, '5th-6th': 14, 'Preschool': 15, '?': 16},
                    0,
                    {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2, 'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6},
                    {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12, 'Armed-Forces': 13, '?': 14},
                    {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5},
                    {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4},
                    {'Female': 0, 'Male': 1, '?': 2},
                    0,
                    0,
                    0,
                    {'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5, 'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11, 'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18, 'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24, 'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31, 'Nicaragua': 31, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36, 'Trinadad&Tobago':  37, 'Peru': 38, 'Hong':39, 'Holand-Netherlands': 40, '?': 41}
                    ]

data, count, meds, arr = u.parseData(TrainingF)

X = []
y = []
for key in data.keys():
    X.append(data[key][:len(data[key]) - 1])
    y.append(data[key][len(data[key]) - 1])

for i in range(len(X)):
    for j in range(len(convertedIndices)):
        if convertedIndices[j] != 0:
            X[i][j] = convertedIndices[j][X[i][j]]
        else:
            X[i][j] = float(X[i][j])


m = CategoricalNB()

m.fit(X, y)

p = m.predict(X)

print("accuracy was: ", accuracy_score(y, p))

tData, tc, tm, arr = u.parseTData(TestF, meds)

tex = []
k = list(tData.keys())
for key in tData.keys():
    tex.append(tData[key])

for i in range(len(tex)):
    for j in range(len(convertedIndices)):
        if convertedIndices[j] != 0:
            tex[i][j] = convertedIndices[j][tex[i][j]]
        else:
            tex[i][j] = float(tex[i][j])

g = m.predict_proba(tex)

f = open('skNB.csv', 'w')
f.write("ID,Prediction\n")
for i in range(len(g)):
    f.write(str(k[i]) + ',' + str(g[i][1]) + '\n')

