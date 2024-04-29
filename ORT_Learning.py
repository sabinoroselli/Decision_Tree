from time import time as tm
from sklearn.metrics import mean_absolute_percentage_error
from ORT import optimal_RT
from TreeStructure import OptimalTree

def train_ORT(Max_depth, Train_df, Val_df, label_name, RS, df_name, SplitType, ModelTree):

    # empty WARM START log file
    open(f'WarmStarts/{df_name.split(".")[0]}_{RS}.mst', 'w').close()

    features = list(Train_df.columns.drop([label_name]))
    labels = Train_df[label_name].unique()
    labels = (label_name, labels)

    best_MAPE = float('inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}
    for depth in range(Max_depth + 1):
        Max_C = 2 ** (depth) - 1
        Min_C = Max_C - int(2 ** (depth - 1) - 1)
        for Splits in range(Min_C,Max_C + 1):
            print('######################################################################')
            print(f'Splits: {Splits}')
            start = tm()
            # Solve the optimization problem to find the optimal tree structure and the optimal splits
            splitting_nodes,non_empty_nodes = optimal_RT(
                                Train_df,
                                features,
                                labels,
                                depth,
                                Splits,
                                RS,
                                df_name,
                                SplitType
            )
            runtime = tm() - start
            # Build the optimal decision tree out of the MILP solution
            ODT = OptimalTree(non_empty_nodes,splitting_nodes,depth, SplitType, ModelTree)
            the_tree = ODT.build_tree(ODT.root.value)

            # split train into features and labels
            X_train = Train_df.drop(columns=label_name)
            X_train = X_train.to_dict('index')
            Y_train = Train_df[label_name]

            # Predict the train set
            train_pred = ODT.predict_regr(X_train, the_tree)
            train_MAPE = round(mean_absolute_percentage_error(Y_train, train_pred) * 100, 2)
            print(f'MAPE (Train Set): {train_MAPE}%')

            # split validation set into features and labels
            X_val = Val_df.drop(columns=label_name)
            X_val = X_val.to_dict('index')
            Y_val = Val_df[label_name]

            # Predict the validation set
            val_pred = ODT.predict_regr(X_val, the_tree)
            val_MAPE = round(mean_absolute_percentage_error(Y_val,val_pred) * 100, 2)
            print(f'MAPE (val Set): {val_MAPE}%')

            iteration_log.update({
                f'{Splits}':{
                            'MAPE (train)':train_MAPE,
                            'MAPE (val)': val_MAPE
                }
            })

            if val_MAPE < best_MAPE:
                best_MAPE = val_MAPE
                best_solution = {
                                 'Depth':depth,
                                 'NumLeaves': len(splitting_nodes) + 1,
                                 'Tree':ODT
                                 }
            RunTimeLog.update({Splits:runtime})

    return best_solution,iteration_log,RunTimeLog





