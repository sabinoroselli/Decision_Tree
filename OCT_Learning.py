from time import process_time as tm # todo double-check this
from sklearn.metrics import accuracy_score
from OCT import optimal_CT
from TreeStructure import OptimalTree
from datetime import  datetime as dt


def train_OCT(Min_depth,Max_depth, Train_df, Val_df, label_name,RS,df_name,SplitType,ModelTree):

    #empty WARM START log file
    open(f'WarmStarts/{df_name.split(".")[0]}_{RS}.mst', 'w').close()

    features = list(Train_df.columns.drop([label_name]))
    labels = Train_df[label_name].unique()
    labels = (label_name, labels)

    best_acc = float('-inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}
    for depth in range(Min_depth,Max_depth + 1):
        MaxSplits = 2 ** (depth) - 1
        MinSplits = MaxSplits - int(2 ** (depth - 1) - 1)
        for Splits in range(MinSplits,MaxSplits + 1):
            start = tm()
            # Solve the optimization problem to find the optimal tree structure and the optimal splits
            splitting_nodes,non_empty_nodes = optimal_CT(
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
            ODT = OptimalTree(non_empty_nodes, splitting_nodes, depth, SplitType, ModelTree)
            the_tree = ODT.build_tree(ODT.root.value)

            # split train into features and labels
            X_train = Train_df.drop(columns=label_name)
            X_train = X_train.to_dict('index')
            Y_train = Train_df[label_name]

            # Predict the train set
            train_pred = ODT.predict_class(X_train, the_tree)
            train_acc = round(accuracy_score(Y_train, train_pred) * 100, 2)

            # split validation set into features and labels
            X_val = Val_df.drop(columns=label_name)
            X_val = X_val.to_dict('index')
            Y_val = Val_df[label_name]

            # Predict the validation set
            val_pred = ODT.predict_class(X_val,the_tree)
            val_Accuracy = round(accuracy_score(Y_val,val_pred) * 100, 2)
            print(f'{df_name}({RS}) Splits: {Splits} Accuracy (Train Set): {train_acc}% Acc (val Set): {val_Accuracy}% -- {dt.now()}')

            iteration_log.update({
                f'{Splits}':{
                    'Acc (train)':train_acc,
                    'Acc (val)': val_Accuracy
                        }
            })

            if val_Accuracy > best_acc:
                best_acc = val_Accuracy
                best_solution = {
                                 'Depth':depth,
                                 'NumLeaves': len(splitting_nodes) + 1,
                                 'Tree':ODT
                                 }
            RunTimeLog.update({Splits:runtime})

    return best_solution,iteration_log,RunTimeLog












