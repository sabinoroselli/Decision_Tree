from gurobipy import *
from TreeStructure import Parent, OptimalTree,RAE,RRSE
from binarytree import build
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from DatabaseParser import DataParser
from time import process_time as tm


def optimal_DT(df, features, labels, Splits, config):
    I = df.index.values

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }
    #
    # mu_max = max(mu.values())
    mu_min = min(mu.values())

    # depth of the tree does not account for root level
    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(Splits + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    # print(binary_tree)

    T_L = [i.value for i in binary_tree.leaves]  # leave nodes
    T_B = [i for i in binary_tree.values if i not in T_L]  # branch nodes

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

    m = Model('OCT')
    m.setParam('LogToConsole', 0)
    m.setParam('Threads', 1)
    m.setParam("LogFile", f'GurobiLogs/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.txt')
    m.setParam('TimeLimit', 60 * config['Timeout'])

    # variables
    d = m.addVars(T_B, vtype=GRB.BINARY, name='d')  # d_t = 1 if node splits
    if config["SplitType"] == "Parallel":
        a = m.addVars(features, T_B, lb=0, ub=1, vtype=GRB.INTEGER, name='a')
    elif config["SplitType"] == "Oblique":
        a = m.addVars(features, T_B, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='a')
        a_abs = m.addVars(features, T_B, vtype=GRB.CONTINUOUS, name='a_abs')
        s = m.addVars(features, T_B, lb=0, ub=1, vtype=GRB.INTEGER, name='s')
    b = m.addVars(T_B, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    z = m.addVars(I, T_L, vtype=GRB.BINARY, name='z')  # point 'i' is in node 't'
    l = m.addVars(T_L, vtype=GRB.BINARY, name='l')  # leaf 't' contains any points at all
    if config['ProbType'] == 'Classification':
        c = m.addVars(labels[1], T_L, vtype=GRB.BINARY, name='c')  # label of node t
        n_k = m.addVars(labels[1], T_L, vtype=GRB.INTEGER, name='n_k')  # number of points of label k in node t
        N = m.addVars(T_L, vtype=GRB.INTEGER, name='N')  # number  of points in node t
        L = m.addVars(T_L, vtype=GRB.INTEGER, name='L')  # number of mis-classifications
    elif config['ProbType'] == 'Regression':
        c = m.addVars(T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='c')  # value in node t
        n = m.addVars(I, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='n')  # absolute error of point i in node t
        n_abs = m.addVars(I, T_L, vtype=GRB.CONTINUOUS, name='n_abs')  # absolute value of n

    # Load previous solution for warm start
    m.update()
    try:
        m.read(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
    except:
        pass
        # print('NO WARM START')
    # else:
    #     print('USING WARM START')

    if config["SplitType"] == "Parallel":
        Const_1 = m.addConstrs(
            quicksum([a[j, t] for j in features]) == d[t] for t in T_B
        )
    elif config["SplitType"] == "Oblique":
        Const_0 = m.addConstrs(
            a_abs[j, t] == abs_(a[j, t]) for j in features for t in T_B
        )

        Const_01 = m.addConstrs(
            s[j, t] >= a_abs[j, t] for j in features for t in T_B
        )
        # Guarantee that the sum of fractions for the split is equal to 1
        Const_1 = m.addConstrs(
            quicksum([a_abs[j, t] for j in features]) <= d[t] for t in T_B
        )

        Const_11 = m.addConstrs(
            s[j, t] <= d[t] for j in features for t in T_B
        )

        Const12 = m.addConstrs(
            quicksum([s[j, t] for j in features]) >= d[t] for t in T_B
        )

    # Const_2 = m.addConstrs(
    #     b[t] <= (1 + mu_max) * d[t] for t in T_B
    # )
    # Const_3 = m.addConstrs(
    #     b[t] >= - (1 + mu_max) * d[t] for t in T_B
    # )

    Const_5 = m.addConstrs(
        d[t] <= d[P[t]] for t in [i for i in T_B if i != root.value]
    )

    Const_6 = m.addConstrs(
        z[i, t] <= l[t] for t in T_L for i in I
    )

    Const_7 = m.addConstrs(
        quicksum([z[i, t] for i in I]) >= l[t] for t in T_L
    )

    Const_8 = m.addConstrs(
        quicksum([z[i, t] for t in T_L]) == 1 for i in I
    )

    if config["SplitType"] == "Parallel":
        Const_12 = m.addConstrs(
            (z[i, l] == 1)
            >>
            (quicksum([a[j, t] * (df.loc[i, j] + mu[j] - mu_min) for j in features]) + mu_min <= b[t])
            # + (1-z[i,l]) * (bigM[i] + mu_max)
            for i in I
            for l in T_L
            for t in A_l[l]
        )

    elif config["SplitType"] == "Oblique":
        Const_12 = m.addConstrs(
            (z[i, l] == 1)
            >>
            (quicksum([a[j, t] * df.loc[i, j] for j in features]) + 0.0001 <= b[t])
            # + (1-z[i,l]) * (bigM[i] + mu_max)
            for i in I
            for l in T_L
            for t in A_l[l]
        )

    Const_13 = m.addConstrs(
        (z[i, l] == 1)
        >>
        (quicksum([a[j, t] * df.loc[i, j] for j in features]) >= b[t])  # - (1 + mu_max) * (1 - z[i, l])
        for i in I
        for l in T_L
        for t in A_r[l]
    )

    Const_14 = m.addConstrs(
        d[t] <= quicksum([l[m] for m in D_l[t]]) for t in T_B
    )

    Const_15 = m.addConstrs(
        d[t] <= quicksum([l[m] for m in D_r[t]]) for t in T_B
    )

    Const_22 = m.addConstr(
        quicksum([d[t] for t in T_B]) <= Splits
    )

    if config['ProbType'] == 'Classification':

        Const_16 = m.addConstrs(
            n_k[k, t] == quicksum([z[i, t] for i in I if k == df.loc[i, labels[0]]])
            for k in labels[1]
            for t in T_L
        )

        Const_17 = m.addConstrs(
            N[t] == quicksum([z[i, t] for i in I])
            for t in T_L
        )

        Const_18 = m.addConstrs(
            quicksum([c[k, t] for k in labels[1]]) == l[t]
            for t in T_L
        )

        Const_20 = m.addConstrs(
            L[t] >= N[t] - n_k[k, t] - len(I) * (1 - c[k, t])
            for k in labels[1]
            for t in T_L
        )

        Const_21 = m.addConstrs(
            L[t] <= N[t] - n_k[k, t] + len(I) * c[k, t]
            for k in labels[1]
            for t in T_L
        )

        m.setObjective(
            quicksum([L[t] for t in T_L])  # cost of miscalculate
        )

    elif config['ProbType'] == 'Regression':

        Const_16 = m.addConstrs(
            (z[i, t] == 1) >> (n[i, t] == c[t] - df.loc[i, labels[0]]) for i in I for t in T_L
        )

        Const_17 = m.addConstrs(
            n_abs[i, t] == abs_(n[i, t]) for i in I for t in T_L
        )

        m.setObjective(
            quicksum([n_abs[i, t] for i in I for t in T_L])  # overall absolute error
        )

    start = tm()
    m.optimize()
    runtime = tm() - start

    if m.status != GRB.INFEASIBLE:
        m.write(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
        vars = m.getVars()
        solution = {
            i.VarName: i.X
            for i in vars}

        non_zero_vars = [key for key, value in solution.items() if value > 0]

        if config["SplitType"] == "Parallel":
            splitting_nodes = {
                i: {
                    'a': [f for f in features if solution[f'a[{f},{i}]'] > 0][0],
                    'b': round(solution[f'b[{i}]'], 6)
                }
                for i in T_B if f'd[{i}]' in non_zero_vars
            }
        elif config["SplitType"] == "Oblique":
            splitting_nodes = {
                i: {
                    'a': {f: round(solution[f'a[{f},{i}]'], 6)
                          for f in features
                          },
                    'b': round(solution[f'b[{i}]'], 6)
                }
                for i in T_B if f'd[{i}]' in non_zero_vars
            }
        if config['ProbType'] == 'Classification':
            non_empty_nodes = {
                i: [c for c in labels[1] if solution[f'c[{c},{i}]'] > 0][0]
                for i in T_L if f'l[{i}]' in non_zero_vars
            }
        elif config['ProbType'] == 'Regression':
            non_empty_nodes = {
                i: solution[f'c[{i}]']
                for i in T_L if f'l[{i}]' in non_zero_vars
            }

        ODT = OptimalTree(
            non_empty_nodes,
            splitting_nodes,
            int(np.ceil(np.log2(Splits + 1))),
            config["SplitType"],
            config["ModelTree"]
        )

    else:
        print('MODEL IS INFEASIBLE')
        ODT = None
    return ODT, runtime


if __name__ == "__main__":

    ProbType = 'Classification';file = 'blogger'
    # ProbType = 'Regression';file = 'RAM_price'

    RS = 7
    Splits = 1

    config = {
        'RandomSeed': 0,
        'ProbType': ProbType,
        "ModelTree": False,
        'SplitType': 'Parallel',
        'label_name': 'class',
        'TestSize': 0.4,
        'df_name': file,
        'Timeout': 60,  # for the single iteration (IN MINUTES)
        'Fraction': 1,  # fraction
        'Meta': False
    }

    df = DataParser(f'{file}.arff', ProbType, one_hot=True)

    df = shuffle(df, random_state=RS)
    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    features = list(Train_df.columns.drop(['class']))

    labels = df['class'].unique()
    labels = ('class', labels)


    ODT,runtime = optimal_DT(
        df=Train_df,
        features=features,
        labels=labels,
        Splits=Splits,
        config=config
    )

    the_tree = ODT.build_tree(ODT.root.value)
    # ODT.print_tree(the_tree)

    # split train into features and labels
    X_train = Train_df.drop(columns='class')
    X_train = X_train.to_dict('index')
    Y_train = Train_df['class']
    # split test set into features and labels
    X_test = Test_df.drop(columns='class')
    X_test = X_test.to_dict('index')
    Y_test = Test_df['class']

    # Predict the train set
    if ProbType == 'Classification':
        train_pred = ODT.predict_class(X_train, the_tree, None)
        print('Accuracy (Train Set): ', round(accuracy_score(Y_train, train_pred) * 100, 2), '%')
    elif ProbType == 'Regression':
        train_pred = ODT.predict_regr(X_train, the_tree, None)
        print('Train -- RAE:', RAE(Y_train, train_pred), 'RRSE:', RRSE(Y_train, train_pred))

    # Predict the test set
    if ProbType == 'Classification':
        test_pred = ODT.predict_class(X_test, the_tree, None)
        print('Accuracy (Test Set): ', round(accuracy_score(Y_test, test_pred) * 100, 2), '%')
    elif ProbType == 'Regression':
        test_pred = ODT.predict_regr(X_test, the_tree, None)
        print('Test -- RAE:', RAE(Y_test, test_pred), 'RRSE:', RRSE(Y_test, test_pred))


