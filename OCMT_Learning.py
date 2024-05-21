from time import process_time as tm # todo double-check this
from datetime import  datetime as dt
import numpy as np
from sklearn.metrics import accuracy_score
from OCMT import optimal_CMT
from TreeStructure import OptimalTree

def train_OCMT(config, Train_df, Val_df):

    #empty WARM START log file
    open(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst', 'w').close()

    features = list(Train_df.columns.drop([config['label_name']]))
    labels = Train_df[config['label_name']].unique()
    labels = (config['label_name'], labels)

    best_acc = float('-inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}

    for Splits in range(config['MinSplits'],config['MaxSplits'] + 1):
        RunTime_per_split = []
        for C in [0.1,1,10,100]:
            start = tm()
            # Solve the optimization problem to find the optimal tree structure and the optimal splits
            splitting_nodes,non_empty_nodes = optimal_CMT(
                                Train_df,
                                features,
                                labels,
                                Splits,
                                C,
                                config
            )
            runtime = tm() - start
            # Build the optimal decision tree out of the MILP solution
            ODT = OptimalTree(
                non_empty_nodes,
                splitting_nodes,
                int(np.ceil(np.log2(Splits + 1))),
                config["SplitType"],
                config["ModelTree"]
            )
            the_tree = ODT.build_tree(ODT.root.value)

            # split train into features and labels
            X_train = Train_df.drop(columns=config['label_name'])
            X_train = X_train.to_dict('index')
            Y_train = Train_df[config['label_name']]

            # Predict the train set
            train_pred = ODT.predict_class(X_train, the_tree)
            train_acc = round(accuracy_score(Y_train, train_pred) * 100, 2)

            # split validation set into features and labels
            X_val = Val_df.drop(columns=config['label_name'])
            X_val = X_val.to_dict('index')
            Y_val = Val_df[config['label_name']]

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
                                 'NumLeaves': len(splitting_nodes) + 1,
                                 'Tree':ODT
                                 }
            RunTime_per_split.append(runtime)
        RunTimeLog.update({Splits:(
                        round(np.average(RunTime_per_split), 2),
                        round(np.std(RunTime_per_split), 2)
        )})

    return best_solution,iteration_log,RunTimeLog