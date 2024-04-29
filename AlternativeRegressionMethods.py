from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_percentage_error
from TreeStructure import RAE,RRSE
import json
from DatabaseParser import DataParser
import os
import numpy as np
# import sklweka.jvm as jvm
# from sklweka.classifiers import WekaEstimator


def CART_RF_SVM_LMT(file, mode, RS):

    label_name = 'class'

    df = DataParser(file,'Regression')

    df = shuffle(df, random_state=RS)

    # if mode == 'LMT': # todo this part needs to be updated with M5P once I can use a linux machine again
    #     # LMT does not take RegressionProblems values for the clas
    #     df[label_name].replace(-1, 'a', inplace=True)
    #     df[label_name].replace(1, 'b', inplace=True)

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

    if mode == 'RF':
        clf = RandomForestRegressor(n_estimators=100,random_state=RS)
    elif mode == 'CART':
        clf = tree.DecisionTreeRegressor(random_state=RS)
    elif mode == 'SVM':
        clf = SVR(kernel='linear')
    # elif mode == 'LMT':
    #     clf = WekaEstimator(classname="weka.classifiers.trees.LMT")
    else:
        raise ValueError('WRONG CLASSIFICATION MODE')

    clf.fit(X_train,Y_train)
    test_pred = clf.predict(X_test)

    Acc = round(mean_absolute_percentage_error(Y_test, test_pred) * 100, 2)
    RelAbsErr = RAE(Y_test,test_pred)
    RelRootSqErr = RRSE(Y_test,test_pred)
    print(f'MAPE: {Acc}%, RAE {RelAbsErr}, RRSE {RelRootSqErr}')

    return Acc,RelAbsErr,RelRootSqErr


if __name__ == "__main__":

    # collection = os.listdir('ClassificationProblems')
    ########### CLASSIFICATION
    # collection = [
    #     'biomed.arff',
    #     'blogger.arff',
    #     'boxing.arff'
    # ]
    ########### REGRESSION
    collection = [
        'auto93.arff',
        # 'bolts.arff',
        # 'cpu.arff'
        # 'pwLinear.arff'
    ]

    Runs = 1
    methods = ['CART','RF','SVM'] #,'LMT'] todo Cannot use LMT on this machine
    for m in methods:
        for i in collection:
            print(m,i)
            all_acc = []
            allRelAbsErr = []
            allRelRootSqErr = []
            for RS in range(Runs):
                acc,RelAbsErr,RelRootSqErr = CART_RF_SVM_LMT(i, m, RS)
                all_acc.append(acc)
                allRelAbsErr.append(RelAbsErr)
                allRelRootSqErr.append(RelRootSqErr)

            prev_logs = {}
            #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
            with open(f'RegressionResults/{m}.json', 'r+',) as logfile:
                try:
                    prev_logs = json.load(logfile)
                except json.decoder.JSONDecodeError:
                    pass
                # update previous content with current train
                prev_logs.update({
                    i.split('.')[0]: {
                        "MAPE":[round(np.average(all_acc),2),round(np.var(all_acc),2)],
                        "RAE": [round(np.average(allRelAbsErr), 2), round(np.var(allRelAbsErr), 2)],
                        "RRSE": [round(np.average(allRelRootSqErr), 2), round(np.var(allRelRootSqErr), 2)]
                    }
                })
            with open(f'RegressionResults/{m}.json', 'w') as logfile:
                # rewrite the file
                json.dump(prev_logs,logfile,indent=4)



