import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss,zero_one_loss
from DataSetsParser import DataParser
from ModelTreeGurobi import MILP_Model
from datetime import datetime as dt


def HyperParametersTuner(config, Train_df, Val_df):

    best_perf = float('inf')
    best_solution = {}
    iteration_log = {}

    RunTimeLog = {}
    for Splits in range(config['MinSplits'], config['MaxSplits'] + 1):
        RunTime_per_C = []
        for C in [0.1, 1, 10, 100]:
            # Solve the optimization problem to find the optimal tree structure and the optimal splits
            ODT,runtime = MILP_Model(
                                Train_df,
                                config['features'],
                                config['labels'],
                                Splits,
                                C,
                                config['Timeout']
            )

            if ODT != None:

                # Build the optimal decision tree out of the MILP solution
                the_tree = ODT.build_tree(ODT.root.value)

                # split train into features and labels
                X_train = Train_df[config['features']]
                X_train = X_train.to_dict('index')
                Y_train = Train_df[config['labels']]
                Y_train = Y_train.to_numpy().astype(int)
                Y_train[Y_train == -1] = 0

                # split validation set into features and labels
                X_val = Val_df[config['features']]
                X_val = X_val.to_dict('index')
                Y_val = Val_df[config['labels']]
                Y_val = Y_val.to_numpy().astype(int)
                Y_val[Y_val == -1] = 0

                # Predict the train and validation sets
                train_pred = np.array(ODT.predict_class(X_train, the_tree)).astype(int)
                val_pred = np.array(ODT.predict_class(X_val, the_tree)).astype(int)
                if config['loss'] == 'hamming':
                    train_performance = round(hamming_loss(Y_train, train_pred), 2)
                    val_performance = round(hamming_loss(Y_val, val_pred), 2)
                elif config['loss'] == 'zero-one':
                    train_performance = round(zero_one_loss(Y_train, train_pred), 2)
                    val_performance = round(zero_one_loss(Y_val, val_pred), 2)
                else:
                    raise ValueError('Loss function not recognized')

                print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits},'
                      f' C:{C} (Train Set): {train_performance} (Val Set): {val_performance} '
                      f'-- Runtime: {round(runtime, 2)} '
                      f'-- {dt.now()}')

                iteration_log.update({
                    f'{(Splits)}':{
                                '(train)':train_performance,
                                '(val)': val_performance
                    }
                })

                if val_performance < best_perf:
                    best_perf = val_performance
                    best_solution = {
                        'Splits': Splits,
                        'NumLeaves': len(ODT.splitting_nodes) + 1,
                        'C': C,
                        'Tree':ODT
                    }

            else:
                iteration_log.update({
                    f'{(Splits)}': {
                        '(train)': np.nan,
                        '(val)': np.nan
                    }
                })

                print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits},'
                      f' C:{C} NO SOLUTION AVAILABLE '
                      f'-- Runtime: {round(runtime, 2)} '
                      f'-- {dt.now()}')

            RunTimeLog.update({Splits: runtime})

            RunTime_per_C.append(runtime)
        RunTimeLog.update({Splits: round(np.average(RunTime_per_C), 2) })

    # Merge Train and Validation set and re-run the optimization with the optimal parameters
    NewTrain_df = pd.concat([Train_df, Val_df])
    print('Computation with optimal parameters')
    ODT,_ = MILP_Model(
        NewTrain_df,
        config['features'],
        config['labels'],
        best_solution['NumLeaves'] - 1,
        best_solution['C'],
        config['Timeout']
    )
    best_solution['Tree'] = ODT

    return best_solution,iteration_log,RunTimeLog

if __name__ == "__main__":

    dfName = 'yeast.arff'
    df, features, labels = DataParser(dfName, one_hot=True, toInt=True)

    config = {
        'df_name': dfName,
        'features':features,
        'labels':labels,
        'MinSplits':0,
        'MaxSplits':0,
        'RandomSeed':7,
        'loss':'zero-one'
    }

    Test_df = df.iloc[:round(len(df) * 0.2)]
    Val_df = df.iloc[len(Test_df): len(Test_df) + round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df) + len(Val_df):]

    best_solution, iteration_log, RunTimeLog = HyperParametersTuner(config,Train_df,Val_df)

    ODT = best_solution['Tree']
    the_tree = ODT.build_tree(ODT.root.value)

    X_test = Test_df[config['features']]
    X_test = X_test.to_dict('index')
    Y_test = Test_df[config['labels']]
    Y_test = Y_test.to_numpy().astype(int)
    Y_test[Y_test == -1] = 0


    # Predict the validation set
    test_pred = np.array(ODT.predict_class(X_test, the_tree)).astype(int)
    loss = zero_one_loss(Y_test, test_pred)
    print('Zero-One', round(loss, 2))

    loss = hamming_loss(Y_test, test_pred)
    print('Hamming', round(loss, 2))

