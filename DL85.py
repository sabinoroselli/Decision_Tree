from pydl85 import DL85Classifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
from DatabaseParser import DataParser
import numpy as np


def SRTL_DL85(file, RS, ProblemType):

    df = np.genfromtxt(f'DL85_Problems/{file}')

    df = shuffle(df, random_state=RS)

    X,y = df[:,1:], df[:,0]
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)

    clf = DL85Classifier(max_depth=3,min_sup=1)

    clf.fit(X_train,Y_train)
    test_pred = clf.predict(X_test)

    print(clf.get_tree_without_transactions())

    Leaves = clf.get_nodes_count()

    Metric = {
                "Accuracy" : round(accuracy_score(Y_test, test_pred) * 100, 2)
    }
    ####### PRINT SINGLE COMPUTATION #########
    # for key,value in Metric.items():
    #     if Leaves == None:
    #         print(f'    {key}:{value}')
    #     else:
    #         print(f'    {key}:{value}',end='  ')
    #         print(f'Leaves:{Leaves}')
    # print(f'RAE {RelAbsErr}, RRSE {RelRootSqErr}')

    return Metric,Leaves


if __name__ == "__main__":
    ########### CLASSIFICATION
    ClassDataBases = [
        ############ MULTICLASS ############
        # 'teachingAssistant.arff',
        # 'glass.arff',
        # 'balance-scale.arff',
        # 'autoUnivMulti.arff',
        'hypothyroid.txt',
        # 'Diabetes.arff',  TOO MANY SYMBOLIC FEATURES...TRY IT AT SOME POINT MAYBE
        ############ BINARY ############
        # 'blogger.arff',
        # 'boxing.arff',
        # 'mux6.arff',
        # 'corral.arff',
        # 'biomed.arff',
        'ionosphere.txt',
        # 'jEdit.arff',
        # 'schizo.arff',
        # 'colic.arff',
        # 'threeOf9.arff',
        # 'R_data_frame.arff',
        'australian-credit.txt',
        # 'doa_bwin_balanced.arff',
        # 'blood-transf.arff',
        # 'autoUniv.arff',
        # 'parity.arff',
        # 'banknote.arff',
        # 'gametes_Epistasis.arff',
        'kr-vs-kp.txt',
        # 'banana.arff'
    ]


    choice = [ClassDataBases, 'Classification']

    Runs = 1

    for i in choice[0]:
        print(i.split('.')[0])
        allLeaves = []
        allMetric = {"Accuracy":[]}
        for RS in range(Runs):
            Metric,Leaves = SRTL_DL85(i, RS, choice[1])
            for key,value in Metric.items():
                allMetric[key].append(value)
            if Leaves != None:
                allLeaves.append(Leaves)

        prev_logs = {}
        #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
        with open(f'{choice[1]}Results/DL85.json', 'r+',) as logfile:
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
        with open(f'{choice[1]}Results/DL85.json', 'w') as logfile:
            # rewrite the file
            json.dump(prev_logs,logfile,indent=4)



