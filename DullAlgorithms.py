from interpretableai import iai
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from TreeStructure import RAE,RRSE
import json
from DatabaseParser import DataParser
import numpy as np
import multiprocessing as mp




import os
os.environ['IAI_DISABLE_UPDATE_CHECK'] = 'true'


def LS_OMT(args):

    file, RS, ProblemType = args
    print("Thread:",RS)
    label_name = 'class'

    df = DataParser(file,
                    ProblemType,
                    one_hot=True,
                    toInt=False
                    )

    df = shuffle(df, random_state=RS)

    TestSize = 0.2
    Test_df = df.iloc[:round(len(df) * TestSize)]
    Train_df = df.iloc[len(Test_df):]

    # Test_df = df.iloc[:round(len(df) * 0.2)]
    # Val_df = df.iloc[len(Test_df): len(Test_df) + round(len(df) * 0.2)]
    # Train_df = df.iloc[len(Test_df) + len(Val_df):]

    # split train set into features and labels
    X_train = Train_df.drop(columns=label_name)
    X_train = X_train.to_numpy()
    Y_train = Train_df[label_name]

    # for i in X_train:
    #     print(i)
    # print(Y_train)

    # split test set into features and labels
    X_test = Test_df.drop(columns=label_name)
    X_test = X_test.to_numpy()
    Y_test = Test_df[label_name]

    # X = df.iloc[:, 0:-1]
    # y = df.iloc[:, -1]
    # (X_train, Y_train), (X_test, Y_test) = iai.split_data('regression', X, y,seed=RS)

    if ProblemType == 'Regression':
        clf = iai.GridSearch(
            iai.OptimalTreeRegressor(
                random_seed=RS,
            ),
            max_depth=range(1,6)
        )
    else:
        clf = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=RS,
            ),
            max_depth=range(1, 6)
        )

    clf.fit(X_train,Y_train)
    test_pred = clf.predict(X_test)

    Leaves = 2**(clf.get_best_params()['max_depth']-1)
    # print(type(list(findkeys(clf.get_learner(), 'Predict'))))
    # print(clf.get_best_params())
    # print(clf.get_learner())

    if ProblemType == 'Regression':
        Metric = {
            "RelAbsErr": RAE(Y_test, test_pred),
            "RelRootSqErr": RRSE(Y_test, test_pred)
        }
    if ProblemType == 'Classification':
        Metric = {
            "Accuracy": round(accuracy_score(Y_test, test_pred) * 100, 2)
        }

    ####### PRINT SINGLE COMPUTATION #########
    # for key,value in Metric.items():
    #     if Leaves == None:
    #         print(f'    {key}:{value}')
    #     else:
    #         print(f'    {key}:{value}',end='  ')
    #         print(f'Leaves:{Leaves}')
    # print(f'RAE {RelAbsErr}, RRSE {RelRootSqErr}')

    return [Metric,Leaves]


if __name__ == "__main__":

    ########### CLASSIFICATION
    ClassDataBases = [
        ############ MULTICLASS ############
        # 'teachingAssistant.arff',
        # 'glass.arff',
        # 'balance-scale.arff',
        # 'autoUnivMulti.arff',
        'hypothyroid.arff',
        # 'Diabetes.arff',  #TOO MANY SYMBOLIC FEATURES...TRY IT AT SOME POINT MAYBE
        ########### BINARY ############
        # 'blogger.arff',
        # 'boxing.arff',
        # 'mux6.arff',
        # 'corral.arff',
        # 'biomed.arff',
        # 'ionosphere.arff',
        # 'jEdit.arff',
        # 'schizo.arff',
        # 'colic.arff',
        # 'threeOf9.arff',
        # 'R_data_frame.arff',
        # 'australian.arff',
        # 'doa_bwin_balanced.arff',
        # 'blood-transf.arff',
        # 'autoUniv.arff',
        # 'parity.arff',
        # 'banknote.arff',
        # 'gametes_Epistasis.arff',
        # 'kr-vs-kp.arff',
        # 'banana.arff'
    ]

    ########### REGRESSION
    RegrDataBases = [
        # 'sensory.arff',
        # 'wisconsin.arff',
        # 'pwLinear.arff',
        # 'cpu.arff',
        # 'yacht_hydrodynamics.arff',
        # 'RAM_price.arff',
        # 'autoMpg.arff',
        # 'vineyard.arff',
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
        # 'KDD.arff',
        'satellite_image.arff'

    ]

    # choice = [RegrDataBases,'Regression']
    choice = [ClassDataBases, 'Classification']

    Runs = 30
    for i in choice[0]:
        print(i.split('.')[0])
        allLeaves = []
        if choice[1] == 'Regression':
            allMetric = {"RelAbsErr": [], "RelRootSqErr": []}
        else:
            allMetric = {"Accuracy": []}

        for RS in range(1,Runs+1):
            Metric,Leaves = LS_OMT([i, RS, choice[1]])
            for key,value in Metric.items():
                allMetric[key].append(value)
            if Leaves != None:
                allLeaves.append(Leaves)

        # args = []
        # for RS in range(1,Runs+1):
        #     args.append([i, RS, choice[1]])
        # p = mp.Pool()
        # result = p.map(LS_OMT, args)
        #
        # for j in result:
        #     for key, value in j[0].items():
        #         allMetric[key].append(value)
        #     if j[1] != None:
        #         allLeaves.append(j[1])

        print(allMetric)
        print(allLeaves)

        prev_logs = {}
        #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
        with open(f'{choice[1]}Results/LS_OMT.json', 'r+',) as logfile:
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
                print(f'{key} -- Avg:{round(np.average(value), 2)},Variance:{round(np.var(value), 3)}')
            if allLeaves != []:
                prev_logs[i.split('.')[0]].update({
                    "Leaves":[round(np.average(allLeaves), 2), round(np.var(allLeaves), 2)]
                })
                print(f'LEAVES -- Avg:{round(np.average(allLeaves), 2)},Variance:{round(np.var(allLeaves), 2)}')
        with open(f'{choice[1]}Results/LS_OMT.json', 'w') as logfile:
            # rewrite the file
            json.dump(prev_logs,logfile,indent=4)



