from gurobipy import *
from TreeStructure import Parent
from binarytree import build
from TreeStructure import OptimalTree
from TreeStructure import RRSE,RAE
from sklearn.utils import shuffle
from DatabaseParser import DataParser
from time import process_time as tm # todo double-check this
import numpy as np

def optimal_RMT(df, features, labels, Splits, C, config ):

    I = df.index.values

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())
    # mu_max = max(mu.values())
    #
    # bigM = { # todo figure out why this does not work sometimes
    #     i: max([abs(df.loc[i, j]) for j in features])
    #     for i in I
    # }

    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(Splits + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    # print(binary_tree)

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
            i : [k.value for k in j.left.leaves]
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

    m = Model('ORMT')
    m.setParam('LogToConsole', 0)
    m.setParam('Threads',1)
    m.setParam("LogFile", f'GurobiLogs/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.txt')
    m.setParam('TimeLimit', 60 * 60)

    # variables
    d = m.addVars(T_B,vtype=GRB.BINARY,name='d') # d_t = 1 if node splits
    if config["SplitType"] == "Parallel":
        a = m.addVars(features, T_B, lb=0, ub=1, vtype=GRB.INTEGER, name='a')
    elif config["SplitType"] == "Oblique":
        a = m.addVars(features, T_B, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='a')
        a_abs = m.addVars(features, T_B, vtype=GRB.CONTINUOUS, name='a_abs')
        s = m.addVars(features, T_B,lb=0,ub=1, vtype=GRB.INTEGER, name='s')
    b = m.addVars(T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    z = m.addVars(I,T_L,vtype=GRB.BINARY,name='z') # point 'i' is in node 't'
    l = m.addVars(T_L,vtype=GRB.BINARY,name='l') # leaf 't' contains any points at all
    Beta = m.addVars(features,T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Beta')  # coefficient for feature i at node t
    Bet_abs = m.addVars(features,T_L, vtype=GRB.CONTINUOUS, name='Beta_abs')  # coefficient for feature i at node t
    Delta = m.addVars(T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')  # constant at node t
    gamma = m.addVars(I,T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='gamma')  # function f(x) - y_i at node t
    gamm_abs = m.addVars(I,T_L, vtype=GRB.CONTINUOUS, name='gamm_abs')  # abs of gamma

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

    # Const_3_1 = m.addConstrs( do not think these are actually required
    #         b[t] <= bigM * d[t] for t in T_B
    #     )
    # Const_3_2 = m.addConstrs(
    #         b[t] >= -bigM * d[t] for t in T_B
    #     )


    Const_5 = m.addConstrs(
        d[t] <= d[P[t]] for t in [i for i in T_B if i != root.value]
    )

    Const_6 = m.addConstrs(
        z[i,t] <= l[t] for t in T_L for i in I
    )

    Const_7 = m.addConstrs(
        quicksum([z[i,t] for i in I]) >= l[t] for t in T_L
    )

    Const_8 = m.addConstrs(
        quicksum([z[i,t] for t in T_L]) == 1 for i in I
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
        (quicksum([a[j, t] * df.loc[i, j] for j in features]) >= b[t]) # - (1 - z[i,l]) * bigM[i]
        for i in I
        for l in T_L
        for t in A_r[l]
    )

    Const_15_0 = m.addConstrs(
        d[t] <= quicksum([ z[i,m] for i in I for m in D_l[t] ]) for t in T_B
    )

    Const_15_1 = m.addConstrs(
        d[t] <= quicksum([z[i, m] for i in I for m in D_r[t]]) for t in T_B
    )

    Const_17 = m.addConstrs(
        (z[i,t] == 1)
        >>
        (quicksum([Beta[j, t] * df.loc[i, j] for j in features]) + Delta[t] - df.loc[i,labels[0]] == gamma[i, t] )
        for i in I
        for t in T_L
    )

    Const_18 = m.addConstrs(
        gamm_abs[i, t] == abs_(gamma[i, t]) for i in I for t in T_L
    )

    Const_19 = m.addConstrs(
        Bet_abs[f, t] == abs_(Beta[f, t]) for f in features for t in T_L
    )

    Const_20 = m.addConstr(
        quicksum([d[t] for t in T_B]) <= Splits
    )

    m.setObjective(
        # variance cost
        quicksum([ Bet_abs[f,t] for f in features for t in T_L ]) + C * quicksum([ gamm_abs[i,t] for i in I for t in T_L])
        )

    start = tm()
    m.optimize()
    runtime = tm() - start

    splitting_nodes = {}
    non_empty_nodes = {}

    if m.status != GRB.INFEASIBLE:
        m.write(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
        vars = m.getVars()
        solution = {
                i.VarName:i.X
                for i in vars}

        non_zero_vars = [key for key,value in solution.items() if value > 0]

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

        non_empty_nodes = {
            i:{
                'Beta':{
                    j: solution[f'Beta[{j},{i}]']
                    for j in features
                },
                'Delta':solution[f'Delta[{i}]']
            }
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
    return ODT,runtime

if __name__ == "__main__":

    file = 'vineyard'
    Splits = 0

    config = {
        'RandomSeed': 0,
        'ProbType': 'Regression',
        "ModelTree": True,
        'SplitType': 'Parallel',
        'label_name': 'class',
        'TestSize': 0.2,
        'ValSize': 0.2,
        'df_name': file,
        'Timeout': 60,  # for the single iteration (IN MINUTES)
        'Fraction': 1  # fraction
    }

    df = DataParser(f'{file}.arff','Regression', one_hot=True)

    df = shuffle(df, random_state=config['RandomSeed'])

    Test_df = df.iloc[:round(len(df) * config['TestSize'])]
    Train_df = df.iloc[len(Test_df):]

    # Test_df = df.iloc[:round(len(df) * config['TestSize'])]
    # Val_df = df.iloc[len(Test_df): len(Test_df) + round(len(df) * config['ValSize'])]
    # Train_df = df.iloc[len(Test_df) + len(Val_df):]


    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    features = list(Train_df.columns.drop(['class']))

    labels = df['class'].unique()
    labels = ('class', labels)

    splitting_nodes,non_empty_nodes = optimal_RMT(
        df=Train_df,
        features=features,
        labels=labels,
        Splits=Splits,
        C=1,
        config=config
    )

    print('Splitting Nodes')
    for i in splitting_nodes.items():
        print(i[0], i[1])
    print('Non-Empty Nodes')
    for i in non_empty_nodes.items():
        print(i[0], i[1])

    ODT = OptimalTree(
        non_empty_nodes,
        splitting_nodes,
        int(np.ceil(np.log2(Splits + 1))),
        config["SplitType"],
        config["ModelTree"]
    )
    the_tree = ODT.build_tree(ODT.root.value)
    # ODT.print_tree(the_tree)

    # split train into features and labels
    X_train = Train_df.drop(columns='class')
    X_train = X_train.to_dict('index')
    Y_train = Train_df['class']

    # Predict the train set
    train_pred = ODT.predict_regr(X_train, the_tree)

    # split test set into features and labels
    X_test = Test_df.drop(columns='class')
    X_test = X_test.to_dict('index')
    Y_test = Test_df['class']

    # Predict the test set
    test_pred = ODT.predict_regr(X_test, the_tree)

    # for ind,i in enumerate(list(Y_test)):
    #     print(i,test_pred[ind])

    print('RAE Train: ', RAE(Y_train, train_pred),'Test: ', RAE(Y_test, test_pred))
    print('RRSE Train: ', RRSE(Y_train, train_pred),'Test: ', RRSE(Y_test, test_pred))


