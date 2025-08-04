from z3 import *
from binarytree import build
from sklearn.utils import shuffle
from DataSetsParser import DataParser
from SupportFunctions import Parent
import numpy as np

def optimal_CT(df, features, labels,splits,C):

    to_drop = ['Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9',
               'Class10', 'Class11', 'Class12', 'Class13', 'Class14'
               ]

    df = df.iloc[:20]

    df = df.drop(columns=to_drop)
    labels = [i for i in labels if i not in to_drop ]

    print(df)

    I = df.index.values

    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(len(labels) - 1 + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    print(binary_tree)

    T_L = [i.value for i in binary_tree.leaves]  # leave nodes
    T_B = [i for i in binary_tree.values if i not in T_L] # branch nodes

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
        i: Parent(root, i) for i in binary_tree.values
    }

    # variables
    d = {t:Bool(f'd_{t}') for t in T_B}
    a = { (j,t):Bool(f'a_{j}_{t}') for j in features for t in T_B }
    b = { t:Int(f'b_{t}') for t in T_B }
    z = { (i,t):Bool(f'z_{i}_{t}') for i in I for t in T_L} # point 'i' is in node 't'
    l = {t:Bool(f'l_{t}') for t in T_L}
    Beta = { (k,j,t):Real(f'Beta_{k}_{j}_{t}') for k in labels for j in features for t in T_L} # coefficient for feature i at node t
    Delta = {(k,t):Real(f'Delta_{k}_{t}') for k in labels for t in T_L}
    e = {(k,i,t):Real(f'e_{k}_{i}_{t}') for k in labels for i in I for t in T_L}

    s = Optimize()
    set_option(rational_to_decimal=True)
    set_option(precision=6)
    set_option(verbose=2)

    const_0 = s.add([
        e[k,i,t] >= 0 for k in labels for i in I for t in T_L
    ])

    const_1 = s.add([
        Sum([ a[j,t] for j in features]) == d[t] for t in T_B
    ])

    const_2 = s.add([
        d[t] <= d[P[t]] for t in [i for i in T_B if i != root.value]
    ])

    const_3 = s.add([
        Implies(z[i,t],l[t]) for i in I for t in T_L
    ])

    const_4 = s.add([
        Sum([ z[i,t] for i in I]) >= l[t] for t in T_L
    ])

    const_5 = s.add([
        Sum([z[i, t] for t in T_L]) == 1 for i in I
    ])

    const_6 = s.add([
        Implies(
            z[i, l],
            Sum([ a[j,t] * df.loc[i, j] for j in features ]) < b[t]
        )
        for i in I
        for l in T_L
        for t in A_l[l]
    ])

    const_7 = s.add([
        Implies(
            z[i, l],
            Sum([a[j,t] * df.loc[i, j] for j in features ]) >= b[t]
        )
        for i in I
        for l in T_L
        for t in A_r[l]
    ])

    const_8 = s.add([
        d[t] <= Sum([l[m] for m in D_l[t]]) for t in T_B
    ])

    const_9 = s.add([
        d[t] <= Sum([l[m] for m in D_r[t]]) for t in T_B
    ])

    const_10 = s.add([
        Implies(
                z[i, t],
                1 - e[c, i, t] <= (Sum([Beta[c, j, t] * df.loc[i, j] for j in features]) + Delta[c, t]) * df.loc[i,c]
        )
        for i in I
        for t in T_L
        for c in labels
    ])

    const_99 = s.add([
        Sum([ d[t] for t in T_B]) <= splits
    ])

    s.minimize(
        Sum([ Abs(Beta[c,f,t]) for c in labels for f in features for t in T_L])
        +
        C * Sum([ e[c,i,t] for c in labels for i in I for t in T_L])
    )



    solution = {}

    if s.check() == sat:
        print('MODEL IS FEASIBLE')
        m = s.model()
        for i in m:
            solution.update({i.name(): m[i]})

        # for a, b in solution.items():
        #     if a[0] == 'h' and b == True:
        #         print(a)
        # print('------------------------')
        # for a, b in solution.items():
        #     if a[0] == 'c' and b == True:
        #         print(a)
        # print('------------------------')
        # for a, b in solution.items():
        #     if a[0] == 'w' and b == True:
        #         print(a)
        # print('------------------------')
        # for a, b in solution.items():
        #     if a[0] == 'z':
        #         print(a, b)

    else:
        print('MODEL IS INFEASIBLE')
    return solution

if __name__ == "__main__":

    label_name = 'Class'

    name = 'yeast.arff'
    splits = 3
    C = 1

    df = DataParser(name, one_hot=True, toInt=True)

    df = shuffle(df,random_state=7)
    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    features = list(Train_df.columns.drop([i for i in df.columns if label_name in i ]))
    labels = [i for i in df.columns if label_name in i ]

    solution = optimal_CT(
        df= Train_df,
        features= features,
        labels= labels,
        splits= splits,
        C=C
    )
