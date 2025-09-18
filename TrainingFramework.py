import json
import numpy as np
import multiprocessing as mp
from sklearn.metrics import accuracy_score as ClassMetr
from sklearn.utils import shuffle
from DatabaseParser import DataParser
from TreeStructure import RRSE,RAE


def log_training(config):

    if config['ModelTree']:
        from OMT_Learning import train_OMT as TrainFunction
    else:
        from ODT_Learning import train_ODT as TrainFunction


    # Erase GurobiLog file content
    with open(f'GurobiLogs/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.txt', 'w'):
        pass

    # # Shuffle the dataset
    df = shuffle(config['df'], random_state=config["RandomSeed"])

    if config['Stratified']:
        Test_df = df.iloc[:round(len(df) * config['TestSize'])]
        Val_df = df.iloc[len(Test_df):].groupby('class',
                group_keys=False).apply(lambda x: x.sample(frac=config['ValSize'],
                                                           random_state=config['RandomSeed']))
        Train_df = df[~df.index.isin(Test_df.index.union(Val_df.index))]
    else:
        Test_df = df.iloc[:round(len(df) * 0.2)]
        Val_df = df.iloc[len(Test_df): len(Test_df) + round(len(df) * 0.2)]
        Train_df = df.iloc[len(Test_df) + len(Val_df):]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df = Train_df.drop(columns=[i])
            Val_df = Val_df.drop(columns=[i])
            Test_df = Test_df.drop(columns=[i])

    best_solution,iteration_log, RunTimeLog = TrainFunction(
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

    # split test set into features and labels
    X_test = Test_df.drop(columns='class')
    X_test = X_test.to_dict('index')
    Y_test = Test_df['class']


    if config['ProbType'] == 'Classification':
        test_pred = ODT.predict_class(X_test, the_tree,config['Leaf_feat'] if config['Meta'] else None)
        test_metric = round(ClassMetr(Y_test, test_pred) * 100, 2)
        print('     Test Accuracy: ', test_metric, '%')
    else:
        test_pred = ODT.predict_regr(X_test, the_tree,config['Leaf_feat'] if config['Meta'] else None)
        test_metric = {
                "RAE":RAE(Y_test,test_pred),
                "RRSE":RRSE(Y_test,test_pred)
        }
        print('Test RAE: ', round(test_metric['RAE'], 2))
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
        config_copy['df'] = DataParser(
                                        name=config['df_name'],
                                        ProbType=config['ProbType'],
                                        one_hot= not(config['Meta'])
                                        )
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

    WW_placeholder = '_WW' if config['WW'] else ''
    Strat_placeholder = '_Strat' if config['Stratified'] else ''
    name = config['df_name'].split(".")[0] + WW_placeholder + Strat_placeholder + '_' + str(int(config['Timeout']/60)) + '_' + config['Info']

    if config['ModelTree']:
        TreeType = 'MOD'
    else:
        TreeType = 'STD'
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
        ############ MULTICLASS ############
        # 'teachingAssistant.arff',
        # 'glass.arff',
        # 'balance-scale.arff',
        # 'autoUnivMulti.arff',
        # 'hypothyroid.arff',
        # 'iris.arff'
        # 'Diabetes.arff',  TOO MANY SYMBOLIC FEATURES...TRY IT AT SOME POINT MAYBE
        ############ BINARY ############
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
        ############ SUPER-SPARSE #############
        # 'adult.arff',
        # 'breastcancer.arff',
        # 'bankruptcy.arff',
        # 'haberman.arff',
        # 'heart.arff',
        # 'mammo.arff',
        # 'mushroom.arff',
        # 'spambase.arff'
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

    meta_features = {
        'autoMpg.arff':{'Branch_feat':[
                                        'cylinders',
                                        'model',
                                        'origin',
                                    ],
                        'Leaf_feat':[
                                    'displacement',
                                    'horsepower',
                                    'weight',
                                    'acceleration'
                                ]
                        },
        'blogger.arff':{'Branch_feat':[
                                        'V4',
                                        'V5'
                                    ],
                        'Leaf_feat':[
                                    'V1',
                                    'V2',
                                    'V3'
                                ]
        },
        'wisconsin.arff': {'Branch_feat': [
                                             'lymph_node_status',
                                             'radius_mean',
                                             'radius_se',
                                             'radius_worst',
                                             'texture_mean',
                                             'texture_se'
                                            ],
                            'Leaf_feat': [
                                            'texture_worst',
                                             'perimeter_mean',
                                             'perimeter_se',
                                             'perimeter_worst',
                                             'area_mean',
                                             'area_se',
                                             'area_worst',
                                             'smoothness_mean',
                                             'smoothness_se',
                                             'smoothness_worst',
                                             'compactness_mean',
                                             'compactness_se',
                                             'compactness_worst',
                                             'concavity_mean',
                                             'concavity_se',
                                             'concavity_worst',
                                             'concave_points_mean',
                                             'concave_points_se',
                                             'concave_points_worst',
                                             'symmetry_mean',
                                             'symmetry_se',
                                             'symmetry_worst',
                                             'fractal_dimension_mean',
                                             'fractal_dimension_se',
                                             'fractal_dimension_worst',
                                             'tumor_size'
                                        ]
        }
    }

    # choice = [ClassDataBases,'Classification']
    choice = [RegrDataBases,'Regression']
    Runs = 30
    config = {}
    for setting in [(0,'slim'),(3,'tree')]: # combinations of different settings for the experiments
        for SplitType in ['Parallel']: # 'Oblique','Parallel'
            for ModelTree in [True]:
                for i in choice[0]:
                    print(f" %%%%%%%%%%%%%%%%%%%% Solving {i.split('.')[0]} %%%%%%%%%%%%%%%%%%%%%%")
                    config.update({
                        'Runs': Runs,
                        'ProbType': choice[1],
                        'SplitType': SplitType,
                        'ModelTree': ModelTree,
                        'TestSize': 0.2,
                        'ValSize': 0.2,
                        'MinSplits': 0,
                        'MaxSplits': setting[0],
                        'df_name':i,
                        'Timeout': 60, # for the single iteration (IN MINUTES)
                        'WW':False,
                        'Stratified':True,
                        'SuperSparseIntegers':True,
                        'SumToZero':False,
                        'Meta':False, # TODO not implemented for standard ODT
                        # 'Branch_feat': meta_features[i]['Branch_feat'],
                        # 'Leaf_feat': meta_features[i]['Leaf_feat']
                        'Info':setting[1],
                        'ConsoleLog':False
                    })
                    prev_log = training_session(config)
                # print(pd.DataFrame(prev_log).to_markdown())