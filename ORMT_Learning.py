import numpy as np
import pandas as pd
from TreeStructure import RAE
from ORMT import optimal_RMT
from datetime import  datetime as dt


def train_ORMT(config, Train_df, Val_df):
    # empty WARM START log file
    open(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst', 'w').close()

    features = list(Train_df.columns.drop(['class']))
    labels = Train_df['class'].unique()
    labels = ('class', labels)

    best_RAE = float('inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}
    for Splits in range(config['MinSplits'], config['MaxSplits'] + 1):
        RunTime_per_split = []
        for C in [0.1, 1, 10, 100]:
            # Solve the optimization problem to find the optimal tree structure and the optimal splits
            ODT,runtime = optimal_RMT(
                                Train_df,
                                features,
                                labels,
                                Splits,
                                C,
                                config
            )

            # Build the optimal decision tree out of the MILP solution
            the_tree = ODT.build_tree(ODT.root.value)

            # split train into features and labels
            X_train = Train_df.drop(columns='class')
            X_train = X_train.to_dict('index')
            Y_train = Train_df['class']

            # Predict the train set
            train_pred = ODT.predict_regr(X_train, the_tree)
            train_RAE = RAE(Y_train, train_pred)

            # split validation set into features and labels
            X_val = Val_df.drop(columns='class')
            X_val = X_val.to_dict('index')
            Y_val = Val_df['class']

            # Predict the validation set
            val_pred = ODT.predict_regr(X_val, the_tree)
            val_RAE = RAE(Y_val,val_pred)
            print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits}, C:{C} RAE (Train Set): {train_RAE} RAE (val Set): {val_RAE} -- {dt.now()}')

            iteration_log.update({
                f'{(Splits)}':{
                            'RAE (train)':train_RAE,
                            'RAE (val)': val_RAE
                }
            })

            if val_RAE <  best_RAE:
                best_RAE = val_RAE
                best_solution = {
                                 'Splits':Splits,
                                 'NumLeaves': len(ODT.splitting_nodes) + 1,
                                 'C':C,
                                 }
            RunTime_per_split.append(runtime)
        RunTimeLog.update({Splits: (
                            round(np.average(RunTime_per_split), 2),
                            round(np.std(RunTime_per_split), 2)
        )})

    # Merge Train and Validation set and re-run the optimization with the optimal parameters
    NewTrain_df = pd.concat([Train_df, Val_df])
    print('Computation with optimal parameters')
    ODT,_ = optimal_RMT(
        NewTrain_df,
        features,
        labels,
        best_solution['NumLeaves'] - 1,
        best_solution['C'],
        config
    )

    best_solution['Tree'] = ODT

    return best_solution,iteration_log,RunTimeLog