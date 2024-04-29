import sklweka.jvm as jvm
from sklweka.classifiers import WekaEstimator
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from DatabaseParser import DataParser
import json
import os
import numpy as np

def LogisticTreeClassifier(file,RS):

    label_name = 'class'

    df = DataParser(file)

    # LMT does not take RegressionProblems values for the clas
    df[label_name].replace(-1,'a',inplace=True)
    df[label_name].replace(1, 'b', inplace=True)

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

    LMT = WekaEstimator(classname="weka.classifiers.trees.LMT")
    LMT.fit(X_train,Y_train)
    test_pred = LMT.predict_regr(X_test)

    # numLeaves = LMT.measureNumLeaves() #todo unfortunately this does not work in python
    # print(f'Nr of Leaves: {round(numLeaves,2)}') # todo wanna do it in Java?

    Acc = round(accuracy_score(Y_test, test_pred) * 100, 2)
    print(f'Accuracy (Test Set): {Acc}%')


    return Acc#,numLeaves

if __name__ == "__main__":

    collection = os.listdir('../ClassificationProblems')
    Runs = 10
    
    # start JVM with Weka package support
    jvm.start(packages=True)

    for i in  collection:
        all_acc = []
        # all_leaves = []
        for RS in range(7,7+Runs):
            acc = LogisticTreeClassifier(i,RS)
            all_acc.append(acc)
            # all_leaves.append(numLeaves)
            #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
        with open('../ClassificationResults/LMT.json', 'r+', ) as logfile:
            try:
                prev_logs = json.load(logfile)
            except json.decoder.JSONDecodeError:
                prev_logs = {}
            # erase content of the file
            logfile.seek(0)
            # update previous content with current train
            prev_logs.update({f'{i}': f'{round(np.average(all_acc),2)}({round(np.var(all_acc),2)})'})
                                      # f'{round(np.average(all_leaves),2)}({round(np.var(all_leaves),2)})'})
            # rewrite the file
            json.dump(prev_logs, logfile, indent=4)

    jvm.stop()






