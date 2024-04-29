from time import process_time as tm # todo double-check this
from datetime import  datetime as dt
from TreeStructure import RAE
from ORT import optimal_RT
from TreeStructure import OptimalTree

def train_ORT(Min_depth,Max_depth, Train_df, Val_df, label_name, RS, df_name, SplitType, ModelTree):

    # empty WARM START log file
    open(f'WarmStarts/{df_name.split(".")[0]}_{RS}.mst', 'w').close()

    features = list(Train_df.columns.drop([label_name]))
    labels = Train_df[label_name].unique()
    labels = (label_name, labels)

    best_RAE = float('inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}
    for depth in range(Min_depth,Max_depth + 1):
        Max_C = 2 ** (depth) - 1
        Min_C = Max_C - int(2 ** (depth - 1) - 1)
        for Splits in range(Min_C,Max_C + 1):
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
            train_RAE = RAE(Y_train, train_pred)

            # split validation set into features and labels
            X_val = Val_df.drop(columns=label_name)
            X_val = X_val.to_dict('index')
            Y_val = Val_df[label_name]

            # Predict the validation set
            val_pred = ODT.predict_regr(X_val, the_tree)
            val_RAE = RAE(Y_val, val_pred)
            print(f'{df_name}({RS}) Splits: {Splits} RAE (Train Set): {train_RAE} RAE (val Set): {val_RAE} -- {dt.now()}')

            iteration_log.update({
                f'{Splits}':{
                            'RAE (train)':train_RAE,
                             'RAE (val)': val_RAE
                }
            })

            if val_RAE < best_RAE:
                best_RAE = val_RAE
                best_solution = {
                                 'Depth':depth,
                                 'NumLeaves': len(splitting_nodes) + 1,
                                 'Tree':ODT
                                 }
            RunTimeLog.update({Splits:runtime})

    return best_solution,iteration_log,RunTimeLog





