from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVR,SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from TreeStructure import RAE,RRSE
import json
from DatabaseParser import DataParser
import numpy as np


def CART_RF_SVM(file, mode, RS, ProblemType):

    label_name = 'class'

    df = DataParser(file,ProblemType)

    df = shuffle(df, random_state=RS)

    TestSize = 0.2
    Test_df = df.iloc[:round(len(df) * TestSize)]
    Train_df = df.iloc[len(Test_df):]

    # split train set into features and labels
    X_train = Train_df.drop(columns=label_name)
    X_train = X_train.to_numpy()
    Y_train = Train_df[label_name]

    # split test set into features and labels
    X_test = Test_df.drop(columns=label_name)
    X_test = X_test.to_numpy()
    Y_test = Test_df[label_name]

    if ProblemType == 'Regression':
        if mode == 'RF':
            clf = RandomForestRegressor(n_estimators=100,random_state=RS)
        elif mode == 'CART':
            clf = DecisionTreeRegressor(random_state=RS)
        elif mode == 'SVM':
            clf = SVR(kernel='linear')
        else:
            raise ValueError('WRONG REGRESSION MODE')
    elif ProblemType == 'Classification':
        if mode == 'RF':
            clf = RandomForestClassifier(n_estimators=100, random_state=RS)
        elif mode == 'CART':
            clf = DecisionTreeClassifier(random_state=RS)
        elif mode == 'SVM':
            clf = SVC(kernel='linear', random_state=RS)
        else:
            raise ValueError('WRONG CLASSIFICATION MODE')
    else:
        raise ValueError('WRONG PROBLEM TYPE')

    clf.fit(X_train,Y_train)
    test_pred = clf.predict(X_test)

    if mode == 'CART':
        Leaves = clf.get_n_leaves()
    else:
        Leaves = None

    if ProblemType == 'Regression':
        Metric = {
                    "RelAbsErr" : RAE(Y_test,test_pred),
                    "RelRootSqErr" : RRSE(Y_test,test_pred)
        }
    if ProblemType == 'Classification':
        Metric = {
                    "Accuracy" : round(accuracy_score(Y_test, test_pred) * 100, 2)
        }
    for key,value in Metric.items():
        if Leaves == None:
            print(f'    {key}:{value}')
        else:
            print(f'    {key}:{value}',end='  ')
            print(f'Leaves:{Leaves}')
    # print(f'RAE {RelAbsErr}, RRSE {RelRootSqErr}')

    return Metric,Leaves


if __name__ == "__main__":
    ########### CLASSIFICATION
    ClassDataBases = [
        'blogger.arff',
        'boxing.arff',
        'mux6.arff',
        'corral.arff',
        'biomed.arff',
        'ionosphere.arff',
        'jEdit.arff',
        'schizo.arff',
        'colic.arff',
        'threeOf9.arff',
        'R_data_frame.arff',
        'australian.arff',
        'doa_bwin_balanced.arff',
        'blood-transf.arff',
        'autoUniv.arff',
        'parity.arff',
        'banknote.arff',
        'gametes_Epistasis.arff',
        'kr-vs-kp.arff',
        'banana.arff'
    ]
    ########### REGRESSION
    RegrDataBases = [
        # 'wisconsin.arff',
        # 'pwLinear.arff',
        # 'cpu.arff',
        # 'yacht_hydrodynamics.arff',
        # 'RAM_price.arff',
        # 'autoMpg.arff',
        'vineyard.arff',
        # 'boston_corrected.arff',
        # 'forest_fires.arff',
        # 'meta.arff',
        # 'arsenic-female-lung.arff',
        # 'arsenic-male-lung.arff',
        # 'titanic_1.arff',
        # 'stock.arff',
        # 'Bank-Note.arff',
        # 'balloon.arff',
        # 'debutanizer.arff',
        # 'analcatdata_supreme.arff',
        # 'Long.arff',
        # 'KDD.arff'
    ]

    # choice = [ClassDataBases, 'Classification']
    choice = [RegrDataBases,'Regression']

Runs = 30
methods = ['RF']#,'CART','SVM',]
for m in methods:
    for i in choice[0]:
        print(m,i.split('.')[0])
        allLeaves = []
        if choice[1] == 'Regression':
            allMetric = {"RelAbsErr":[],"RelRootSqErr":[]}
        else:
            allMetric = {"Accuracy":[]}
        for RS in range(Runs):
            Metric,Leaves = CART_RF_SVM(i, m, RS, choice[1])
            for key,value in Metric.items():
                allMetric[key].append(value)
            if Leaves != None:
                allLeaves.append(Leaves)

        prev_logs = {}
        #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
        with open(f'{choice[1]}Results/{m}.json', 'r+',) as logfile:
            try:
                prev_logs = json.load(logfile)
            except json.decoder.JSONDecodeError:
                pass
            # update previous content with current train
            prev_logs.update({
                i.split('.')[0]:{
                    key: [round(np.average(value), 2), round(np.var(value), 2)]
                        for key,value in allMetric.items()
                }
            })
            for key, value in allMetric.items():
                print(f'{key} -- Avg:{round(np.average(value), 2)},Variance:{round(np.var(value), 2)}')
            if allLeaves != []:
                prev_logs[i.split('.')[0]].update({
                    "Leaves":[round(np.average(allLeaves), 2), round(np.var(allLeaves), 2)]
                })
                print(f'LEAVES -- Avg:{round(np.average(allLeaves), 2)},Variance:{round(np.var(allLeaves), 2)}')
        with open(f'{choice[1]}Results/{m}.json', 'w') as logfile:
            # rewrite the file
            json.dump(prev_logs,logfile,indent=4)



