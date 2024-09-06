from z3 import *
from binarytree import build
import numpy as np
from DatabaseParser import DataParser
from time import process_time as tm
from TreeStructure import OptimalTree, Parent
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def optimal_DT(df,Splits):

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

    # binary variables
    d = { t:Bool(f'd_{t}') for t in T_B } # d_t = 1 if node splits
    a = { (j,t):Bool(f'a_{j}_{t}') for t in T_B for j in features}
    b = { t:Real(f'b_{t}') for t in T_B }
    z = { (i,t):Bool(f'z_{i}_{t}') for t in T_L for i in I} # point 'i' is in node 't'
    l = { t:Bool(f'l_{t}') for t in T_L } # leaf 't' contains any points at all
    c = { (k,t):Bool(f'c_{k}_{t}') for t in T_L for k in labels} # label of node t
    N_k = { (k,t):Int(f'n_{k}_{t}') for t in T_L for k in labels} # number of points of label k in node t
    N = { t:Int(f'N_{t}') for t in T_L } # number of points in node t
    L = { t:Int(f'L_{t}') for t in T_L } # number of points in node t minus the number of points of the most common label

    Const_2_1 = [
        Implies(d[t],PbEq([ (a[j,t],1) for j in features],1))  for t in T_B
    ]
    Const_2_2 = [
        Implies(Not(d[t]), And([Not(a[j, t]) for j in features])) for t in T_B
    ]

    Const_2 = Const_2_1 + Const_2_2

    Const_3 = [
        # Implies(Not(d[t]),b[t] < max([abs(i) for i in df.drop(['class'],axis=1).max().values]))
        # for t in T_B
    ]

    Const_5 = [
        Implies(Not(d[P[t]]),Not(d[t])) for t in [i for i in T_B if i != root.value]
    ]

    Const_6 = [
        Implies(And([Not(z[i,t]) for i in I]),Not(l[t]))
        for t in T_L
    ]

    Const_7 = [
        Implies(Or([z[i,t] for i in I]),l[t]) for t in T_L
    ]
    # each point to exactly one leaf
    Const_8 = [
        PbEq([ (z[i,t],1) for t in T_L], 1 ) for i in I
    ]

    Const_9 = [
        Implies(d[t],
                PbGe([ (l[m],1) for m in D_l[t]],1) )
        for t in T_B
    ]

    Const_10 = [
        Implies(d[t],
                PbGe([ (l[m],1) for m in D_r[t]],1) )
        for t in T_B
    ]

    Const_13 = [
        Implies(
            z[i, t],
            Sum([a[j, m] * df.loc[i, j] for j in features]) >= b[m]
        )
        for i in I
        for t in T_L
        for m in A_r[t]
    ]

    Const_14 = [
        Implies(
            z[i, t],
            Sum([a[feature, m] * df.loc[i, feature] for feature in features]) < b[m]
        )
        for i in I
        for t in T_L
        for m in A_l[t]
    ]

    Const_15 = [
        N_k[k,t] == Sum([ z[i,t] for i in I if k == df.loc[i,'class'] ])
        for k in labels
        for t in T_L
    ]

    Const_16 = [
        N[t] == Sum([ z[i,t] for i in I]) for t in T_L
    ]

    Const_18 = [
        Implies(
                l[t],
                Or([c[k,t] for k in labels])
            )
        for t in T_L
    ]

    Const_20 = [
        Implies(
            c[k, t],
            L[t] == N[t] - N_k[k, t]
        )
        for k in labels
        for t in T_L
    ]

    Const_22 = [
        L[t] >= 0 for t in T_L
    ]

    Const_23 = [
        Sum([d[t]  for t in T_B]) <= Splits
    ]

    s = Optimize()
    set_option(rational_to_decimal=True)
    set_option(precision=6)
    set_option(verbose=1)
    s.add(
        Const_2 +
        Const_3 +
        Const_5 +
        Const_6 +
        Const_7 +
        Const_8 +
        Const_9 +
        Const_10 +
        Const_13 +
        Const_14 +
        Const_15 +
        Const_16 +
        Const_18 +
        Const_20 +
        Const_22 +
        Const_23
    )

    s.minimize(
        Sum([L[t] for t in T_L])
    )

    solution = {}
    StartTime = tm()
    if s.check() == sat:
        print('sat')
        m = s.model()
        for i in m:
            solution.update({i.name():m[i]})

        for a,b in solution.items():
            if a[0] == 'L':
                print(a,b)
        for a,b in solution.items():
            if a[0] == 'l':
                print(a,b)
        for a,b in solution.items():
            if a[0] == 'N':
                print(a,b)
        for a,b in solution.items():
            if a[0] == 'd':
                print(a,b)
        for a,b in solution.items():
            if a[0] == 'a':
                print(a,b)
        for a,b in solution.items():
            if a[0] == 'b':
                print(a,b)

        splitting_nodes = {
            i: {
                'a': [f for f in features if solution[f'a_{f}_{i}'] == True ][0],
                'b': int(str(solution[f'b_{i}']))#float(''.join(solution[f'b_{i}'].as_decimal(6)[:-1]))
                                # if str(solution[f'b_{i}']) != '0' else 0
            }
            for i in T_B if solution[f'd_{i}'] == True
        }

        non_empty_nodes = {
            i: [c for c in labels if solution[f'c_{c}_{i}'] == True][0]
            for i in T_L if solution[f'l_{i}'] == True
        }

        print(splitting_nodes)
        print(non_empty_nodes)

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

    runtime = round(tm() - StartTime, 2)
    return ODT, runtime

if __name__ == "__main__":

    file = 'iris'
    RS = 7
    Splits = 2

    df = DataParser(f'{file}.arff','Classification', one_hot=True,toInt=True)

    df = shuffle(df, random_state=RS)
    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    ODT,runtime = optimal_DT(Train_df, Splits)

    print('RunTime:',runtime)

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






