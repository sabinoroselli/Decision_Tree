import json
import numpy as np
import multiprocessing as mp
from sklearn.metrics import hamming_loss,zero_one_loss
from sklearn.utils import shuffle
from DataSetsParser import DataParser
from Trainer import HyperParametersTuner


def log_training(config):


    # # Shuffle the dataset
    df = shuffle(config['df'], random_state=config["RandomSeed"])

    df = df.iloc[:300] # Shorten the train dataset for testing todo comment this off when done

    if config['Stratified']: # TODO how to stratify multi label datasets?
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
    to_remove = []
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            # print(f'Eliminating {i}')
            to_remove.append(i)
            Train_df = Train_df.drop(columns=[i])
            Val_df = Val_df.drop(columns=[i])
            Test_df = Test_df.drop(columns=[i])
    config['labels'] = [i for i in config['labels'] if i not in to_remove]
    config['features'] = [i for i in config['features'] if i not in to_remove]

    best_solution,iteration_log, RunTimeLog = HyperParametersTuner(
        config,
        Train_df,
        Val_df
    )

    print(f"######## TESTING ---Optimal hyperparameters for the {config['RandomSeed']}th run:"
          f" NumLeaves = {best_solution['NumLeaves']}, C = {best_solution['C']}")
    if best_solution['Tree'] != None:
        ODT = best_solution['Tree']
        # Build the optimal decision tree out of the MILP solution
        the_tree = ODT.build_tree(ODT.root.value)

        # split test set into features and labels
        X_test = Test_df[config['features']]
        X_test = X_test.to_dict('index')
        Y_test = Test_df[config['labels']]
        Y_test = Y_test.to_numpy().astype(int)
        Y_test[Y_test == -1] = 0

        test_pred = ODT.predict_class(X_test, the_tree)
        Hamming = round(hamming_loss(Y_test,test_pred),2)
        ZeroOne = round(zero_one_loss(Y_test,test_pred),2)
        print(f'Test -- Hamming Loss: {Hamming}, Zero-One Loss: {ZeroOne}')
    else:
        Hamming = np.nan
        ZeroOne = np.nan
        print(f'Test -- MODEL NOT AVAILABLE')

    train_log = {
        'NumLeaves': best_solution['NumLeaves'],
        'theTree':best_solution['Tree'],
        'TestMetric': {'Hamming':Hamming,'ZeroOne':ZeroOne},
        'IterLog': iteration_log,
        'RunTime': RunTimeLog
    }

    return train_log

def training_session(config):
    args = []
    for i in range(config['Runs']):
        config_copy = config.copy()
        df,features,labels = DataParser(
                                        name=config['df_name'],
                                        one_hot= True
                                        )
        config_copy['df'] = df
        config_copy['features'] = features
        config_copy['labels'] = labels
        config_copy['RandomSeed'] = i
        args.append(config_copy)

    p = mp.Pool()
    result = p.map(log_training,args)
    best_tree_struct = [i['NumLeaves'] for i in result]
    leaves_avg = round(np.average(best_tree_struct), 2)
    leaves_var = round(np.std(best_tree_struct), 2)
    Hamming_scores = [i['TestMetric']['Hamming'] for i in result]
    Hamming_average = round(np.average(Hamming_scores), 2)
    Hamming_var = round(np.std(Hamming_scores), 2)
    Hamming_count = len([i for i in Hamming_scores if i != np.nan])

    ZeroOne_scores = [i['TestMetric']['ZeroOne'] for i in result]
    ZeroOne_average = round(np.average(ZeroOne_scores), 2)
    ZeroOne_var = round(np.std(ZeroOne_scores), 2)
    ZeroOne_count = len([i for i in ZeroOne_scores if i != np.nan])

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

    Strat_placeholder = '_Strat' if config['Stratified'] else ''
    name = (config['df_name'].split(".")[0] + '_'
            +  config['loss']
            + Strat_placeholder
            + '_' + str(int(config['Timeout']/60)) + 'H')

    prev_logs = {}

    with open(f'MILP_Tree.json', 'r+', ) as logfile:
        try:
            prev_logs = json.load(logfile)
        except json.decoder.JSONDecodeError:
            pass
        prev_logs.update({
            name: {
                                    'Metric': {
                                        'Hamming':[Hamming_average,Hamming_var,f'{Hamming_count}/{len(Hamming_scores)}'],
                                        'ZeroOne':[ZeroOne_average,ZeroOne_var,f'{ZeroOne_count}/{len(ZeroOne_scores)}']
                                    },
                                    'Leaves': (leaves_avg, leaves_var),
                                    'RunTimes': RunTime_avg_and_std
            }
        })

    # rewrite the file
    with open(f'MILP_Tree.json', 'w') as logfile:
        json.dump(prev_logs,logfile,indent=4)

    return prev_logs

if __name__ == "__main__":

    collection = [
        'yeast.arff',
        'emotions.arff',
        'genbase.arff',
        'reuters.arff'
    ]

    Runs = 7
    config = {}
    for loss in ['hamming']: # 'hamming','zero-one'
        for i in collection:
            print(f" %%%%%%%%%%%%%%%%%%%% Solving {i.split('.')[0]} %%%%%%%%%%%%%%%%%%%%%%")
            config.update({
                'Runs': Runs,
                'TestSize': 0.2,
                'ValSize': 0.2,
                'MinSplits': 0,
                'MaxSplits': 3,
                'df_name':i,
                'Timeout': 60 * 2, # for the single iteration (IN MINUTES)
                'Stratified':False,
                'loss': loss # hamming or zero-one
            })
            prev_log = training_session(config)
        # print(pd.DataFrame(prev_log).to_markdown())