from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import json
from DatabaseParser import DataParser
import os
import numpy as np
# import sklweka.jvm as jvm
# from sklweka.classifiers import WekaEstimator


def CART_RF_SVM_LMT(file, mode, RS):

    label_name = 'class'

    df = DataParser(file,'Classification')

    df = shuffle(df, random_state=RS)

    if mode == 'LMT':
        # LMT does not take RegressionProblems values for the clas
        df[label_name].replace(-1, 'a', inplace=True)
        df[label_name].replace(1, 'b', inplace=True)

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
        clf = RandomForestClassifier(n_estimators=100,random_state=RS)
    elif mode == 'CART':
        clf = tree.DecisionTreeClassifier(random_state=RS)
    elif mode == 'SVM':
        clf = SVC(kernel='linear',random_state=RS)
    elif mode == 'LMT':
        clf = WekaEstimator(classname="weka.classifiers.trees.LMT")
    else:
        raise ValueError('WRONG CLASSIFICATION MODE')

    clf.fit(X_train,Y_train)
    test_pred = clf.predict(X_test)

    Acc = round(accuracy_score(Y_test, test_pred) * 100, 2)
    print(f'Accuracy (Test Set): {Acc}%')

    return Acc


if __name__ == "__main__":

    # collection = os.listdir('ClassificationProblems')
    collection = [
        'biomed.arff',
        'blogger.arff',
        # 'colic.arff',
        # 'blood-transf.arff',
        'corral.arff',
        # 'boxing.arff',
        # 'australian.arff'
        # 'banknote.arff',
        # 'delta_ailerons.arff',
        # 'mushroom.arff',
        # 'mc1.arff',
        # 'R_data_frame.arff',
        # 'banana.arff',
        # 'doa_bwin_balanced.arff',
        # 'gametes_Epistasis.arff',
        # 'parity.arff'
    ]

    Runs = 3
    methods = ['CART','RF','SVM'] #,'LMT'] Cannot use LMT on this machine
    for m in methods:
        for i in collection:
            print(i)
            all_acc = []
            for RS in range(Runs):
                acc = CART_RF_SVM_LMT(i, m, RS)
                all_acc.append(acc)
            prev_logs = {}
            #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
            with open(f'ClassificationResults/{m}.json', 'r+',) as logfile:
                try:
                    prev_logs = json.load(logfile)
                except json.decoder.JSONDecodeError:
                    pass
                # update previous content with current train
                prev_logs.update({
                    i.split('.')[0]: [round(np.average(all_acc),2),round(np.var(all_acc),2)]
                })
            with open(f'ClassificationResults/{m}.json', 'w') as logfile:
                # rewrite the file
                json.dump(prev_logs,logfile,indent=4)



