from datetime import  datetime as dt
from TreeStructure import RAE
from sklearn.metrics import accuracy_score
from OptimalDecisionTree import optimal_DT
import pandas as pd

def train_ODT(config, Train_df, Val_df):

    # empty WARM START log file
    open(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst', 'w').close()

    features = list(Train_df.columns.drop(['class']))
    labels = Train_df['class'].unique()
    labels = ('class', labels)

    if config['ProbType'] == 'Classification':
        best_perf = float('-inf')
    elif config['ProbType'] == 'Regression':
        best_perf = float('inf')

    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}
    for Splits in range(config['MinSplits'], config['MaxSplits'] + 1):

        # Solve the optimization problem to find the optimal tree structure and the optimal splits
        ODT,runtime = optimal_DT(
                            Train_df,
                            features,
                            labels,
                            Splits,
                            config
        )
        # Build the optimal decision tree out of the MILP solution

        the_tree = ODT.build_tree(ODT.root.value)

        # split train into features and labels
        X_train = Train_df.drop(columns='class')
        X_train = X_train.to_dict('index')
        Y_train = Train_df['class']

        # split validation set into features and labels
        X_val = Val_df.drop(columns='class')
        X_val = X_val.to_dict('index')
        Y_val = Val_df['class']

        # Predict the train set
        if config['ProbType'] == 'Classification':
            train_pred = ODT.predict_class(X_train, the_tree)
            train_performance = round(accuracy_score(Y_train, train_pred) * 100, 2)
        elif config['ProbType'] == 'Regression':
            train_pred = ODT.predict_regr(X_train, the_tree)
            train_performance = RAE(Y_train, train_pred)

        # Predict the validation set
        if config['ProbType'] == 'Classification':
            val_pred = ODT.predict_class(X_val, the_tree)
            val_performance = round(accuracy_score(Y_val, val_pred) * 100, 2)
            print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits}  Train: {train_performance}% Val: {val_performance}% -- {dt.now()}')
        elif config['ProbType'] == 'Regression':
            val_pred = ODT.predict_regr(X_val, the_tree)
            val_performance = RAE(Y_val, val_pred)
            print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits} Train RAE: {train_performance} Val RAE: {val_performance} -- {dt.now()}')

        iteration_log.update({
            f'{Splits}':{
                        'Train Perf':train_performance,
                         'Validation Perf': val_performance
            }
        })
        if config['ProbType'] == 'Classification':
            if val_performance > best_perf:
                best_perf = val_performance
                best_solution = {
                                 'Splits':Splits,
                                 'NumLeaves': len(ODT.splitting_nodes) + 1,
                                 'Tree':ODT
                                 }
            RunTimeLog.update({Splits:runtime})
        if config['ProbType'] == 'Regression':
            if val_performance < best_perf:
                best_perf = val_performance
                best_solution = {
                                 'Splits':Splits,
                                 'NumLeaves': len(ODT.splitting_nodes) + 1,
                                 'Tree':ODT
                                 }
            RunTimeLog.update({Splits:runtime})

    # Merge Train and Validation set and re-run the optimization with the optimal parameters
    NewTrain_df = pd.concat([Train_df, Val_df])
    print('Computation with optimal parameters')
    ODT, _ = optimal_DT(
        NewTrain_df,
        features,
        labels,
        best_solution['NumLeaves'] - 1,
        config
    )

    best_solution['Tree'] = ODT

    return best_solution,iteration_log,RunTimeLog





