import json
import numpy as np
import multiprocessing as mp
from TreeStructure import RAE,RRSE
from sklearn.metrics import accuracy_score as ClassMetr
from sklearn.utils import shuffle
from DatabaseParser import DataParser


def log_training(config):

    if config['ProbType'] == 'Classification':
        if config['ModelTree']:
            from OCMT_Learning import train_OCMT as TrainFunction
        else:
            from OCT_Learning import train_OCT as TrainFunction

    elif config['ProbType'] =='Regression':
        if config['ModelTree']:
            from ORMT_Learning import train_ORMT as TrainFunction
        else:
            from ORT_Learning import train_ORT as TrainFunction
    else:
        raise ValueError('WRONG MODEL TYPE')

    # Erase GurobiLog file content
    with open(f'GurobiLogs/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.txt', 'w'):
        pass

    # # Shuffle the dataset
    df = shuffle(config['df'], random_state=config["RandomSeed"])

    Test_df = df.iloc[:round(len(df) * config['TestSize'])]
    Val_df = df.iloc[len(Test_df): len(Test_df) + round(len(df) * config['ValSize'])]
    if config['Fraction'] == '':
        Train_df = df.iloc[len(Test_df) + len(Val_df):]
    else:
        print(f'Reducing Training data set to {float(config["Fraction"])*100}% of its original length')
        Train_df = df.iloc[len(Test_df) + len(Val_df):] # todo this could have been coded a little more elegantly
        print('Original Length:',len(Train_df))
        Train_df = df.iloc[:round(len(Train_df) * float(config['Fraction']))]
        print('Reduced Length:', len(Train_df))
    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df = Train_df.drop(columns=[i])
            Val_df = Val_df.drop(columns=[i])
            Test_df = Test_df.drop(columns=[i])

    best_solution,iteration_log,RunTimeLog = TrainFunction(
        config,
        Train_df,
        Val_df
    )
    if config['ModelTree']:
        print(f"######## TESTING ---Optimal hyperparameters for the {config['RandomSeed']}th run: NumLeaves = {best_solution['NumLeaves']}, C = {best_solution['C']}")
    else:
        print(f"######## TESTING ---Optimal hyperparameters for the {config['RandomSeed']}th run: NumLeaves = {best_solution['NumLeaves']}")
    ODT = best_solution['Tree']
    # Build the optimal decision tree out of the MILP solution
    the_tree = ODT.build_tree(ODT.root.value)

    # split validation set into features and labels
    X_test = Test_df.drop(columns=config['label_name'])
    X_test = X_test.to_dict('index')
    Y_test = Test_df[config['label_name']]


    if config['ProbType'] == 'Classification':
        test_pred = ODT.predict_class(X_test, the_tree)
        test_metric = round(ClassMetr(Y_test, test_pred) * 100, 2)
        print('     ACC (Test Set): ', test_metric, '%')
    else:
        test_pred = ODT.predict_regr(X_test, the_tree)
        test_metric = {
                "RAE":RAE(Y_test,test_pred),
                "RRSE":RRSE(Y_test,test_pred)
        }
        print('RAE (Test Set): ', round(test_metric['RAE'], 2))
        # print('RRSE (Test Set): ', round(test_metric['RRSE'], 2))

    train_log = {
        'NumLeaves': best_solution['NumLeaves'],
        'theTree':best_solution['Tree'],
        'TestMetric': test_metric,
        'IterLog': iteration_log,
        'RunTime': RunTimeLog
    }

    return train_log

def training_session(config):
    args = []
    for i in range(config['Runs']):
        config_copy = config.copy()
        config_copy['df'] = DataParser(name=config['df_name'],ProbType=config['ProbType'])
        config_copy['RandomSeed'] = i
        args.append(config_copy)

    p = mp.Pool()
    result = p.map(log_training,args)
    best_tree_struct = [i['NumLeaves'] for i in result]
    leaves_avg = round(np.average(best_tree_struct), 2)
    leaves_var = round(np.std(best_tree_struct), 2)
    if config['ProbType'] == 'Classification':
        best_scores = [i['TestMetric'] for i in result]
        Metric_average = round(np.average(best_scores), 2)
        Metric_var = round(np.std(best_scores), 2)
    else:
        RAE_scores = [i['TestMetric']['RAE'] for i in result]
        RAE_average = round(np.average(RAE_scores), 2)
        RAE_var = round(np.std(RAE_scores), 2)

        RRSE_scores = [i['TestMetric']['RRSE'] for i in result]
        RRSE_average = round(np.average(RRSE_scores), 2)
        RRSE_var = round(np.std(RRSE_scores), 2)



    RunTime_avg_and_std = {}
    for split in range(config['MinSplits'],config['MaxSplits'] + 1):
        RunTime_avg_and_std.update({
            split:
                (
                    round(np.average([i['RunTime'][split] for i in result]),2),
                    round(np.std([i['RunTime'][split] for i in result]),2)
                )
        })

    #### WRITE THE LOG OF THE TRAINING SESSION IN A JSON FILE ####

    name = config['df_name'].split(".")[0] + config['Fraction']

    if config['ModelTree']:
        TreeType = 'MOD'
    else:
        config['ModelTree'] = 'STD'
    prev_logs = {}

    with open(f'{config["ProbType"]}Results/{config["SplitType"]}_{TreeType}.json', 'r+', ) as logfile:
        try:
            prev_logs = json.load(logfile)
        except json.decoder.JSONDecodeError:
            pass
        if config['ProbType'] == 'Classification':
            prev_logs.update({
                name: {
                                        'Metric':(Metric_average,Metric_var),
                                        'Leaves':(leaves_avg,leaves_var),
                                        'RunTimes':RunTime_avg_and_std
                                        }
            })
        else:
            prev_logs.update({
                name: {
                                        'Metric': {
                                            'RAE':[RAE_average,RAE_var],
                                            'RRSE':[RRSE_average,RRSE_var]
                                        },
                                        'Leaves': (leaves_avg, leaves_var),
                                        'RunTimes': RunTime_avg_and_std
                }
            })

    # rewrite the file
    with open(f'{config["ProbType"]}Results/{config["SplitType"]}_{TreeType}.json', 'w') as logfile:
        json.dump(prev_logs,logfile,indent=4)

    return prev_logs

if __name__ == "__main__":

    # collection = os.listdir('RegressionProblems')

    ########### CLASSIFICATION
    ClassDataBases = [
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

    choice = [ClassDataBases,'Classification']
    # choice = [RegrDataBases,'Regression']
    Runs = 3
    config = {}
    for fraction in ['0.3','0.5','0.7']:
        for SplitType in ['Parallel']: # 'Oblique'
            for ModelTree in [True]:
                for i in choice[0]:
                    print(f" %%%%%%%%%%%%%%%%%%%% Solving {i.split('.')[0]} %%%%%%%%%%%%%%%%%%%%%%")
                    config.update({
                        'Runs': Runs,
                        'ProbType': choice[1],
                        'SplitType': SplitType,
                        'ModelTree': ModelTree,
                        'label_name': 'class',
                        'TestSize': 0.2,
                        'ValSize': 0.2,
                        'MinSplits': 3,
                        'MaxSplits': 6,
                        'df_name':i,
                        'Timeout': 60, # for the single iteration (IN MINUTES)
                        'Fraction': fraction
                    })
                    prev_log = training_session(config)
            # print(pd.DataFrame(prev_log).to_markdown())