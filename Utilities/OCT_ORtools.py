from ortools.sat.python import cp_model
from binarytree import build
import numpy as np
from DatabaseParser import DataParser
from TreeStructure import Parent, OptimalTree
from time import process_time as tm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def optimal_DT(df,Splits):

    print(df)

    # bigM = int(max([abs(i) for i in df.drop(['class'],axis=1).max().values]))

    I = df.index.values
    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(Splits + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    T_L = [i.value for i in binary_tree.leaves] # leave nodes
    T_B = [i for i in binary_tree.values if i not in T_L]

    print(binary_tree)

    labels = df['class'].unique()
    features = list(df.columns.drop(['class']))

    occs = {i:df['class'].value_counts()[i] for i in labels}

    A_l = {
        i: [j.value for j in list(root) if j != i and j.left != None and i in j.left.values] for i in binary_tree.values
    }

    A_r = {
        i: [j.value for j in list(root) if j != i and j.left != None and i in j.right.values] for i in
        binary_tree.values
    }

    D_l = {
        i: [k.value for k in j.left.leaves]
        for i in T_B
        for j in list(root)
        if j.value == i
    }

    D_r = {
        i: [k.value for k in j.right.leaves]
        for i in T_B
        for j in list(root)
        if j.value == i
    }

    P = {
        i:Parent(root,i)  for i in binary_tree.values
    }

    model = cp_model.CpModel()

    # binary variables
    d = { t:model.NewBoolVar(f'd_{t}') for t in T_B } # d_t = 1 if node splits
    a = { (j,t):model.NewBoolVar(f'a_{j}_{t}') for t in T_B for j in features}
    b = { t:model.NewIntVar(-int(max([abs(i) for i in df.drop(['class'],axis=1).max().values])),
                            int(max([abs(i) for i in df.drop(['class'],axis=1).max().values])),
                            f'b_{t}') for t in T_B }
    z = { (i,t):model.NewBoolVar(f'z_{i}_{t}') for t in T_L for i in I} # point 'i' is in node 't'
    l = { t:model.NewBoolVar(f'l_{t}') for t in T_L } # leaf 't' contains any points at all
    c = { (k,t):model.NewBoolVar(f'c_{k}_{t}') for t in T_L for k in labels} # label of node t
    N_k = { (k,t):model.NewIntVar(0,
                                  occs[k],
                                  f'n_{k}_{t}') for t in T_L for k in labels} # number of points of label k in node t
    N = { t:model.NewIntVar(0,
                            len(I),
                            f'N_{t}') for t in T_L } # number of points in node t
    L = { t:model.NewIntVar(0,
                            len(I),
                            f'L_{t}') for t in T_L } # number of points in node t minus the number of points of the most common label

    for t in T_B:
        model.Add( sum([a[j,t] for j in features ]) == d[t] )

    for t in [i for i in T_B if i != root.value]:
        model.Add( d[t] <= d[P[t]] )

    for t in T_L:
        for i in I:
            model.Add( z[i,t] <= l[t] )

    for t in T_L:
        model.Add( sum([z[i,t] for i in I]) >= l[t] )

    # each point to exactly one leaf
    for i in I:
        model.AddExactlyOne([z[i,t] for t in T_L])

    for t in T_B:
        model.Add( sum([l[m] for m in D_l[t]]) >= d[t] )

    for t in T_B:
        model.Add( sum([l[m] for m in D_r[t]]) >= d[t] )


    for i in I:
        for t in T_L:
            for m in A_r[t]:
                model.Add(
                    sum([a[j, m] * int(df.loc[i, j]) for j in features]) >= b[m]
                ).OnlyEnforceIf(z[i,t])

    for i in I:
        for t in T_L:
            for m in A_l[t]:
                model.Add(
                    sum([a[feature,m] * int(df.loc[i,feature]) for feature in features ]) < b[m]
                ).OnlyEnforceIf(z[i,t])

    for k in labels:
        for t in T_L:
            model.Add( N_k[k,t] == sum([ z[i,t] for i in I if k == df.loc[i,'class'] ]) )

    for t in T_L:
        model.Add( N[t] == sum([ z[i,t] for i in I]) )

    for t in T_L:
        model.Add( sum([c[k,t] for k in labels]) == l[t] )

    for k in labels:
        for t in T_L:
            model.Add( L[t] == N[t] - N_k[k, t] ).OnlyEnforceIf(c[k,t])

    model.Add( sum([d[t]  for t in T_B]) <= Splits )

    model.Minimize(sum([L[t] for t in T_L]))

    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 60 * 60
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 1
    StartTime = tm()
    # solution_printer = cp_model.ObjectiveSolutionPrinter()
    # status = solver.SolveWithSolutionCallback(model, solution_printer)
    status = solver.Solve(model)
    print('Finished Solving')
    runtime = round(tm() - StartTime, 2)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Maximum of objective function: {solver.ObjectiveValue()}\n")

        # for t in T_B:
        #     print(f'd_{t} = {solver.Value(d[t])}')
        #     for f in features:
        #         print(f'a_{f,t} = {solver.Value(a[f,t])}')
        #     print(f'b_{t} = {solver.Value(b[t])}')
        # for t in T_L:
        #     print(f'N_{t} = {solver.Value(N[t])}')

        splitting_nodes = {
            t: {
                'a': [f for f in features if solver.Value(a[f,t]) == 1][0],
                'b': solver.Value(b[t])
            }
            for t in T_B if solver.Value(d[t]) == 1
        }

        non_empty_nodes = {
            t: [k for k in labels if solver.Value(c[k,t]) == True][0]
            for t in T_L if solver.Value(l[t]) == 1
        }

        ODT = OptimalTree(
            non_empty_nodes,
            splitting_nodes,
            int(np.ceil(np.log2(Splits + 1))),
            'Parallel',
            False
        )
    else:
        print('MODEL IS INFEASIBLE')
        ODT = None


    return ODT, runtime


if __name__ == "__main__":

    file = 'autoUnivMulti'
    # file = 'iris'
    RS = 7
    Splits = 7

    df = DataParser(f'{file}.arff', 'Classification', one_hot=False, toInt=True)

    df = shuffle(df, random_state=RS)

    # Test_df = df.iloc[:round(len(df) * 0.2)]
    # Train_df = df.iloc[len(Test_df):]
    ################### STRATIFIED SPLIT ############################################################
    Test_df = df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=RS))
    Train_df = df[~df.index.isin(Test_df.index)]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        # print(i,Train_df[i].nunique())
        if Train_df[i].nunique() <= 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    ODT, runtime = optimal_DT(Train_df, Splits)

    print('RunTime:', runtime)
    if ODT != None:
        the_tree = ODT.build_tree(ODT.root.value)

        # split train into features and labels
        X_train = Train_df.drop(columns='class')
        X_train = X_train.to_dict('index')
        Y_train = Train_df['class']
        # split test set into features and labels
        X_test = Test_df.drop(columns='class')
        X_test = X_test.to_dict('index')
        Y_test = Test_df['class']

        train_pred = ODT.predict_class(X_train, the_tree, None)
        print('Accuracy (Train Set): ', round(accuracy_score(Y_train, train_pred) * 100, 2), '%')
        test_pred = ODT.predict_class(X_test, the_tree, None)
        print('Accuracy (Test Set): ', round(accuracy_score(Y_test, test_pred) * 100, 2), '%')


