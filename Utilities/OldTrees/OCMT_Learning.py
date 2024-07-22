from datetime import  datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from OCMT import optimal_CMT

def train_OCMT(config, Train_df, Val_df):

    #empty WARM START log file
    open(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst', 'w').close()

    features = list(Train_df.columns.drop(['class']))
    labels = Train_df['class'].unique()
    labels = ('class', labels)

    best_acc = float('-inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}

    for Splits in range(config['MinSplits'],config['MaxSplits'] + 1):
        RunTime_per_split = []
        for C in [0.1,1,10,100]:

            # Solve the optimization problem to find the optimal tree structure and the optimal splits
            ODT,runtime = optimal_CMT(
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
            train_pred = ODT.predict_class(X_train, the_tree)
            train_acc = round(accuracy_score(Y_train, train_pred) * 100, 2)

            # split validation set into features and labels
            X_val = Val_df.drop(columns='class')
            X_val = X_val.to_dict('index')
            Y_val = Val_df['class']

            # Predict the validation set
            val_pred = ODT.predict_class(X_val,the_tree)
            val_Accuracy = round(accuracy_score(Y_val,val_pred) * 100, 2)
            print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits}, C: {C} Train: {train_acc}% Val: {val_Accuracy}% -- {dt.now()}')

            iteration_log.update({
                f'{Splits,C}':{
                    'Acc (train)':train_acc,
                    'Acc (val)': val_Accuracy
                        }
            })

            if val_Accuracy > best_acc:
                best_acc = val_Accuracy
                best_solution = {
                                 'Splits':Splits,
                                 'C':C,
                                 'NumLeaves': len(ODT.splitting_nodes) + 1,
                                 # 'Tree': ODT
                                 }
            RunTime_per_split.append(runtime)
        RunTimeLog.update({Splits:(
                        round(np.average(RunTime_per_split), 2),
                        round(np.std(RunTime_per_split), 2)
        )})

    # Merge Train and Validation set and re-run the optimization with the optimal parameters
    NewTrain_df = pd.concat([Train_df,Val_df])
    print('Computation with optimal parameters')
    ODT,_ = optimal_CMT(
        NewTrain_df,
        features,
        labels,
        best_solution['NumLeaves'] -1,
        best_solution['C'],
        config
    )

    best_solution['Tree'] = ODT


    return best_solution,iteration_log,RunTimeLog