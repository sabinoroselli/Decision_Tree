from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from Utilities.OldTrees.TreeStructure import RAE,RRSE
import weka.core.jvm as jvm
from weka.classifiers import Classifier as Classifier
from weka.core.converters import Loader
import arff as rf
import json
import numpy as np


def LMT_M5P(file, ProbType, RS):

    label_name = 'class'

    loader = Loader(classname='weka.core.converters.ArffLoader')
    df = loader.load_file(f'{ProbType}Problems/{file}')
    df.class_is_last()


    shuffled_indexes = shuffle([str(i+1) for i in range(len(df))],random_state=RS)

    # TestSize = 0.2
    # Test_indexes = shuffled_indexes[:round(len(df) * TestSize)]
    # Train_indexes = shuffled_indexes[len(Test_indexes):]
    #
    # Train_df = df.subset(row_range=",".join(Train_indexes))
    # Test_df = df.subset(row_range=",".join(Test_indexes))

    Test_indexes = shuffled_indexes[:round(len(df) * 0.2)]
    Val_indexes = shuffled_indexes[len(Test_indexes): len(Test_indexes) + round(len(df) * 0.2)]
    Train_indexes = shuffled_indexes[len(Test_indexes) + len(Val_indexes):]

    Train_df = df.subset(row_range=",".join(Train_indexes))
    # Val_df = df.subset(row_range=",".join(Val_indexes))
    Test_df = df.subset(row_range=",".join(Test_indexes))


    # print(Test_df.subset(col_names=[label_name]))

    if ProbType == 'Classification':
        data = rf.load(open(f'{ProbType}Problems/{file}', 'rt'))
        labels = {
            str(data['attributes'][-1][1][0]): 0.0,
            str(data['attributes'][-1][1][1]): 1.0
        }
        Y_test = [labels[str(i)] for i in Test_df.subset(col_names=[label_name])]
        placeholder = 'Leaves'
        cls = Classifier(classname="weka.classifiers.trees.LMT")
    elif ProbType == 'Regression':
        Y_test = [float(str(i)) for i in Test_df.subset(col_names=[label_name])]
        placeholder = 'Rules'
        cls = Classifier(classname="weka.classifiers.trees.M5P")
    else:
        raise ValueError('WRONG PROBLEM TYPE')

    cls.build_classifier(Train_df)
    Leaves = cls.jwrapper.getMeasure(f"measureNum{placeholder}")

    test_pred = [cls.classify_instance(i) for i in Test_df]

    if ProbType == 'Regression':
        Metric = {
                    "RelAbsErr" : RAE(Y_test,test_pred),
                    "RelRootSqErr" : RRSE(Y_test,test_pred)
        }
    if ProbType == 'Classification':
        Metric = {
                    "Accuracy" : round(accuracy_score(Y_test, test_pred) * 100, 2)
        }

    for key,value in Metric.items():
        print(f'    {key}:{value}', end = " ")
    print(f'Leaves:{Leaves}')

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
        'wisconsin.arff',
        'pwLinear.arff',
        'cpu.arff',
        'yacht_hydrodynamics.arff',
        'RAM_price.arff',
        'autoMpg.arff',
        'vineyard.arff',
        'boston_corrected.arff',
        'forest_fires.arff',
        'meta.arff',
        'arsenic-female-lung.arff',
        'arsenic-male-lung.arff',
        'titanic_1.arff',
        'stock.arff',
        'Bank-Note.arff',
        'balloon.arff',
        'debutanizer.arff',
        'analcatdata_supreme.arff',
        'Long.arff',
        'KDD.arff'
    ]

    choice = [ClassDataBases, 'Classification']
    # choice = [RegrDataBases,'Regression']
    map = {'Classification':'LMT','Regression':'M5P'}
    Runs = 30
    jvm.start(packages=True)
    for i in choice[0]:
        print(f'solving {i.split(".")[0]} with {map[choice[1]]}')
        allLeaves = []
        if choice[1] == 'Regression':
            allMetric = {"RelAbsErr": [], "RelRootSqErr": []}
        else:
            allMetric = {"Accuracy": []}
        for RS in range(Runs):
            Metric,Leaves = LMT_M5P(i, choice[1], RS)
            for key,value in Metric.items():
                allMetric[key].append(value)
                allLeaves.append(Leaves)

        prev_logs = {}
        ### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
        with open(f'{choice[1]}Results/{map[choice[1]]}.json', 'r+',) as logfile:
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

            prev_logs[i.split('.')[0]].update({
                "Leaves": [round(np.average(allLeaves), 2), round(np.var(allLeaves), 2)]
            })

        with open(f'{choice[1]}Results/{map[choice[1]]}.json', 'w') as logfile:
            # rewrite the file
            json.dump(prev_logs,logfile,indent=4)
    jvm.stop()



