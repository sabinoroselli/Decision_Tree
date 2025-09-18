from pystreed import STreeDPiecewiseLinearRegressor
from sklearn.utils import shuffle
from TreeStructure import RAE,RRSE
from DatabaseParser import DataParser
import json
import numpy as np
import multiprocessing as mp

def SRTL(args):

    file, RS, ProblemType = args
    print('Thread:',RS)
    label_name = 'class'

    df = DataParser(file,
                    ProblemType,
                    one_hot=True,
                    toInt=False,
                    StdScale=False
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

    # split test set into features and labels
    X_test = Test_df.drop(columns=label_name)
    X_test = X_test.to_numpy()
    Y_test = Test_df[label_name]

    clf = STreeDPiecewiseLinearRegressor(max_depth=20,random_seed=RS)
    clf.fit(X_train,Y_train)
    test_pred = clf.predict(X_test)

    Leaves = clf.get_n_leaves()

    Metric = {
                "RelAbsErr" : RAE(Y_test,test_pred),
                "RelRootSqErr" : RRSE(Y_test,test_pred)
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

    ########### REGRESSION
    RegrDataBases = [
        # 'sensory.arff',
        'wisconsin.arff',
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
        # 'satellite_image.arff'
    ]

    choice = [RegrDataBases,'Regression']

    Runs = 30
    methods = ['SRTL']
    for i in choice[0]:
        print(i.split('.')[0])
        allLeaves = []
        allMetric = {"RelAbsErr":[],"RelRootSqErr":[]}

        args = []
        for RS in range(Runs):
            args.append([i,RS,choice[1]])
        p = mp.Pool()
        result = p.map(SRTL,args)

        for j in result:
            for key,value in j[0].items():
                allMetric[key].append(value)
            if j[1] != None:
                allLeaves.append(j[1])

        print(allMetric)
        print(allLeaves)

        prev_logs = {}
        #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####
        with open(f'{choice[1]}Results/SRTL.json', 'r+',) as logfile:
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
        with open(f'{choice[1]}Results/SRTL.json', 'w') as logfile:
            # rewrite the file
            json.dump(prev_logs,logfile,indent=4)



