from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from SVM_Ensamble import MultiLabel_SVM
from sklearn.utils import shuffle
from sklearn.metrics import hamming_loss,zero_one_loss
from DataSetsParser import DataParser
import numpy as np
import json


def CART_RF_SVM(file, mode, RS):

    df,features,labels = DataParser(name=file,one_hot=True)
    df = shuffle(df, random_state=RS)

    TestSize = 0.2
    Test_df = df.iloc[:round(len(df) * TestSize)]
    Train_df = df.iloc[len(Test_df):]

    # split train set into features and labels
    X_train = Train_df[features]
    X_train = X_train.to_numpy()

    Y_train = Train_df[labels]
    Y_train = Y_train.to_numpy()
    ############################################
    X_test = Test_df[features]
    X_test = X_test.to_numpy()

    Y_test = Test_df[labels]
    Y_test = Y_test.to_numpy().astype(int)

    if mode == 'RF':
        clf = RandomForestClassifier(n_estimators=100, random_state=RS)
    elif mode == 'CART':
        clf = DecisionTreeClassifier(random_state=RS)
    elif mode == 'SVM':
        clf = MultiLabel_SVM(random_state=RS,C=1)
    else:
        raise ValueError('WRONG CLASSIFICATION MODE')

    Metric = {}
    Leaves = None

    clf.fit(X_train,Y_train)
    if (mode == 'SVM' and clf.failed == False) or mode != 'SVM':
            test_pred = clf.predict(X_test)
            Metric = {
                "Hamming": hamming_loss(Y_test, test_pred),
                "ZeroOne": zero_one_loss(Y_test, test_pred)
            }

    if mode == 'CART':
        Leaves = clf.get_n_leaves()

    return Metric,Leaves


if __name__ == "__main__":

    collection = [
        'yeast.arff',
        'emotions.arff',
        'genbase.arff',
        'reuters.arff'
    ]

    Runs = 30
    methods = ['RF']  # CART,RF,SVM

    for m in methods:
        for i in collection:
            print(m,i.split('.')[0])
            allLeaves = []
            allMetric = {"Hamming":[],"ZeroOne":[]}
            RS = 0
            count = 0
            while count < Runs:
                Metric,Leaves = CART_RF_SVM(i, m, RS)
                if Metric != {}:
                    print(count,RS, Metric, Leaves)
                    for key,value in Metric.items():
                        allMetric[key].append(value)
                    if Leaves != None:
                        allLeaves.append(Leaves)
                    count += 1
                RS += 1


            #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
            prev_logs = {}
            with open(f'{m}.json', 'r+', ) as logfile:
                try:
                    prev_logs = json.load(logfile)
                except json.decoder.JSONDecodeError:
                    pass
                # update previous content with current train
                prev_logs.update({
                    i.split('.')[0]: {
                        key: [round(np.average(value), 2), round(np.var(value), 2)]
                        for key, value in allMetric.items()
                    }
                })
                for key, value in allMetric.items():
                    print(f'{key} -- Avg:{round(np.average(value), 2)},Variance:{round(np.var(value),2)}')
                if allLeaves != []:
                    prev_logs[i.split('.')[0]].update({
                        "Leaves": [round(np.average(allLeaves), 2), round(np.var(allLeaves), 2)]
                    })
                    print(f'LEAVES -- Avg:{round(np.average(allLeaves), 2)},Variance:{round(np.var(allLeaves), 2)}')
            with open(f'{m}.json', 'w') as logfile:
                # rewrite the file
                json.dump(prev_logs, logfile, indent=4)







