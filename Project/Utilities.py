import random
from SVM import SVM

convertedIndices = [0,
                    {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4, 'Local-gov': 5, 'State-gov': 6,
                     'Without-pay': 7, 'Never-worked': 8, '?': -1},
                    9,
                    {'Bachelors': 10, 'Some-college': 11, '11th': 12, 'HS-grad': 13, 'Prof-school': 14, 'Assoc-acdm': 15, 'Assoc-voc': 16, '9th': 17, '7th-8th': 18, '12th': 19, 'Masters': 20, '1st-4th': 21, '10th': 22, 'Doctorate': 23, '5th-6th': 24, 'Preschool': 25, '?': -1},
                    26,
                    {'Married-civ-spouse': 27, 'Divorced': 28, 'Never-married': 29, 'Separated': 30, 'Widowed': 31, 'Married-spouse-absent': 32, 'Married-AF-spouse': 33},
                    {'Tech-support': 34, 'Craft-repair': 35, 'Other-service': 36, 'Sales': 37, 'Exec-managerial': 38, 'Prof-specialty': 39, 'Handlers-cleaners': 40, 'Machine-op-inspct': 41, 'Adm-clerical': 42, 'Farming-fishing': 43, 'Transport-moving': 44, 'Priv-house-serv': 45, 'Protective-serv': 46, 'Armed-Forces': 47, '?': -1},
                    {'Wife': 48, 'Own-child': 49, 'Husband': 50, 'Not-in-family': 51, 'Other-relative': 52, 'Unmarried': 53},
                    {'White': 54, 'Asian-Pac-Islander': 55, 'Amer-Indian-Eskimo': 56, 'Other': 57, 'Black': 58},
                    {'Female': 59, 'Male': 60, '?': -1},
                    61,
                    62,
                    63,
                    {'United-States': 64, 'Cambodia': 65, 'England': 66, 'Puerto-Rico': 67, 'Canada': 68, 'Germany': 69, 'Outlying-US(Guam-USVI-etc)': 70, 'India': 71, 'Japan': 72, 'Greece': 73, 'South': 74, 'China': 75, 'Cuba': 76, 'Iran': 77, 'Honduras': 78, 'Philippines': 79, 'Italy': 80, 'Poland': 81, 'Jamaica': 82, 'Vietnam': 83, 'Mexico': 84, 'Portugal': 85, 'Ireland': 86, 'France': 87, 'Dominican-Republic': 88, 'Laos': 89, 'Ecuador': 90, 'Taiwan': 91, 'Haiti': 92, 'Columbia': 93, 'Hungary': 94, 'Guatemala': 95, 'Nicaragua': 96, 'Scotland': 97, 'Thailand': 98, 'Yugoslavia': 99, 'El-Salvador': 100, 'Trinadad&Tobago':  101, 'Peru': 100, 'Hong':101, 'Holand-Netherlands': 102, '?': -1}
                    ]
cic = 103

def convertToNums(term):
    newT = [0]*cic
    for i in range(len(term)):
        t = term[i]
        if isinstance(convertedIndices[i], int):
            newT[convertedIndices[i]] = float(t)
        elif t != '?':
            newT[convertedIndices[i][t]] = 1
    return newT



def quickselect(indexes, list, k):
    """
    Found at https://rcoh.me/posts/linear-time-median-finding/
    modified to run by giving and taking indexes
    Select the kth element in l (0 based)
    :param l: List of numerics
    :param k: Index
    :param pivot_fn: Function to choose a pivot, defaults to random.choice
    :return: The kth element of l
    """
    if len(list) == 1:
        assert k == 0
        return list[0]

    pivot = list[random.choice(indexes)]

    lows = [el for el in indexes if list[el] < pivot]
    highs = [el for el in indexes if list[el] > pivot]
    pivots = [el for el in indexes if list[el] == pivot]

    if k < len(lows):
        return quickselect(lows, list, k)
    elif k < len(lows) + len(pivots):
        # We got lucky and guessed the median
        return pivots[0]
    else:
        return quickselect(highs, list, k - len(lows) - len(pivots))


def convToBin(num, tup):
    if float(num) < tup[0]:
        return 0
    elif float(num) > tup[1]:
        return 2
    else: return 1


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
            for i in range(0, len(terms)):
                pList.append(set())
                pList[i].add(terms[i])
                medList.append([])
                allList.append([])
                allList[i].append(terms[i])
                if terms[i][len(terms[i]) - 1].isdigit():
                    medList[i].append(int(terms[i]))
                else:
                    medList[i] = None
            attCount = len(terms) - 1
            if not testD: medList[len(medList) - 1] = None # don't want to convert our result to discrete even if we can
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        else:
            for i in range(0, len(terms)):
                allList[i].append(terms[i])
                pList[i].add(terms[i])
                if medList[i] is not None and terms[i][len(terms[i]) - 1].isdigit():
                    medList[i].append(int(terms[i]))
                else: medList[i] = None
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)

        data[dataCount] = terms.copy()
        indices.add(dataCount)
        dataCount += 1
        inL = list(indices)
    for i in range(0, len(medList)):
        if medList[i] is not None:
            medList[i] = medList[i][quickselect(inL, medList[i], int(len(medList[i]) / 3))], \
                         medList[i][quickselect(inL, medList[i], int(2*len(medList[i])/3))]
    for key in data.keys():
        for i in range(0, len(medList)):
            if medList[i] is not None:
                pList[i] = {0, 1, 2}
                data[key][i] = convToBin(data[key][i], medList[i])

    dataCount += 1
    f.close()
    return data, dataCount, medList, [data, dataCount, attCount, indices, pList]

def parseNumData(File, testD=0):
    attCount = None
    f = open(File, 'r')
    dataCount = 0
    exes, wise = [], []
    indices = []
    pList = []
    medList = []
    allList = []
    unknown = set()  # key is index in data, value is term index
    unknownList = []  # unknownList[i] is the mode of that attribute
    for line in f:
        terms = line.strip().split(',')
        if attCount is None:
            attCount = len(terms) - 1
        exes.append(convertToNums(terms[:len(terms) - 1]))
        wise.append(SVM.to1(int(terms[len(terms) - 1])))
        indices.append(dataCount)
        dataCount += 1
        inL = list(indices)
    dataCount += 1
    f.close()
    return exes, wise, dataCount, indices, [dataCount, attCount, indices]


def parseTData(File, meds):
    data = {}
    attCount = None
    f = open(File, 'r')
    dataCount = 0
    indices = set()
    #pList = []
    medList = meds
    allList = []
    unknown = set()  # key is index in data, value is term index
    unknownList = []  # unknownList[i] is the mode of that attribute
    for line in f:
        terms = line.strip().split(',')
        ID = terms[0]
        terms = terms[1:]
        if attCount is None:
            #for i in range(1, len(terms)):
                #pList.append(set())
                #pList[i-1].add(terms[i])
            attCount = len(terms) - 1
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)
        #else:
            #for i in range(1, len(terms)):
                #pList[i-1].add(terms[i])
                #if terms[i] == "unknown":
                #    unknown.add(dataCount)

        data[ID] = terms
        indices.add(dataCount)
        dataCount += 1
        inL = list(indices)


    for key in data.keys():
        for i in range(0, len(medList)):
            if medList[i] is not None:
                #pList[i] = {0, 1, 2}
                data[key][i] = convToBin(data[key][i], medList[i])

    dataCount += 1
    f.close()
    return data, dataCount, medList, [data, dataCount, attCount, indices]

def parseNumTData(File, testD=0):
    attCount = None
    f = open(File, 'r')
    dataCount = 0
    exes = []
    indices = []
    pList = []
    medList = []
    allList = []
    unknown = set()  # key is index in data, value is term index
    unknownList = []  # unknownList[i] is the mode of that attribute
    for line in f:
        terms = line.strip().split(',')
        if attCount is None:
            attCount = len(terms) - 1
        exes.append(convertToNums(terms[1:]))
        indices.append(dataCount)
        dataCount += 1
        inL = list(indices)
    dataCount += 1
    f.close()
    return exes, dataCount, indices
