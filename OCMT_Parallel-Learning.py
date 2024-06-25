from gurobipy import *
from TreeStructure import Parent
from binarytree import build
from time import process_time as tm # todo double-check this
from datetime import  datetime as dt
import numpy as np
from sklearn.metrics import accuracy_score
from OCMT import optimal_CMT
from TreeStructure import OptimalTree
import multiprocessing as mp

def optimal_CMT(df, features, labels, Splits, C, config):

    # print(df.to_markdown())


    I = df.index.values

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())
    #
    # mu_max = max(mu.values())

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

    m = Model('OCMT')
    # m.setParam('LogToConsole', 0)
    # m.setParam('Threads',1)
    # m.setParam("LogFile",f'GurobiLogs/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.txt')
    m.setParam('TimeLimit', 60 * config['Timeout'])

    # variables
    d = m.addVars(T_B,lb=0,ub=1,vtype=GRB.INTEGER,name='d') # d_t = 1 if node splits
    if config["SplitType"] == "Parallel":
        a = m.addVars(features,T_B,lb=0,ub=1,vtype=GRB.INTEGER,name='a')
    elif config["SplitType"] == "Oblique":
        a = m.addVars(features, T_B, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='a')
        a_abs = m.addVars(features, T_B, vtype=GRB.CONTINUOUS, name='a_abs')
        s = m.addVars(features,T_B,lb=0,ub=1, vtype=GRB.INTEGER,name='s')
    b = m.addVars(T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    z = m.addVars(I,T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='z') # point 'i' is in node 't'
    l = m.addVars(T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='l') # leaf 't' contains any points at all
    Beta = m.addVars(features,T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Beta')  # coefficient for feature i at node t
    Bet_abs = m.addVars(features, T_L, vtype=GRB.CONTINUOUS,name='Beta_abs')  # coefficient for feature i at node t
    Delta = m.addVars(T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')  # constant at node t
    e = m.addVars(I,T_L, lb=0, vtype=GRB.CONTINUOUS, name='e')

    # Load previous solution for warm start
    # m.update()
    # try:
    #     m.read(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
    # except:
    #     # print('NO WARM START')
    #     pass
    # else:
    #     print('USING WARM START')

    # lz = m.addVars(T_L, vtype=GRB.INTEGER, name='lz')  #points that ended up in l
    # m.addConstrs(
    #     lz[t] == quicksum([ z[i,t] for i in I]) for t in T_L
    # )

    if config["SplitType"] == "Parallel":
        Const_1 = m.addConstrs(
            quicksum([a[j, t] for j in features]) == d[t] for t in T_B
        )
    elif config["SplitType"] == "Oblique":
        Const_0 = m.addConstrs(
            a_abs[j,t] == abs_(a[j,t]) for j in features for t in T_B
        )

        Const_01 = m.addConstrs(
            s[j,t] >= a_abs[j,t] for j in features for t in T_B
        )

        # Guarantee that the sum of fractions for the split is equal to 1
        Const_1 = m.addConstrs(
            quicksum([a_abs[j,t] for j in features]) <= d[t] for t in T_B
        )

        Const_11 = m.addConstrs(
            s[j, t] <= d[t] for j in features for t in T_B
        )

        Const12 = m.addConstrs(
            quicksum([s[j,t] for j in features]) >= d[t] for t in T_B
        )

        # Const12 = m.addConstrs(
        #     quicksum([s[j, t] for j in features]) <= 1 for t in T_B
        # )


    # Const_2 = m.addConstrs(
    #         b[t] <= mu_max * d[t] for t in T_B
    #     )
    # Const_3 = m.addConstrs(
    #         b[t] >= -mu_max * d[t] for t in T_B
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
            (quicksum([a[j, t] * (df.loc[i, j] + mu[j]-mu_min) for j in features]) + mu_min <= b[t])  # + (1-z[i,l]) * (bigM[i] + mu_max)
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
        (quicksum([a[j, t] * df.loc[i, j] for j in features]) >= b[t])  # - (1 - z[i,l]) * bigM[i]
        for i in I
        for l in T_L
        for t in A_r[l]
    )

    ###### If a node splits, at least one leaf node descendant on each side must have points
    Const_14 = m.addConstrs(
        d[t] <= quicksum([ l[m] for m in D_l[t] ]) for t in T_B
    )

    Const_15 = m.addConstrs(
        d[t] <= quicksum([ l[m] for m in D_r[t]]) for t in T_B
    )


    Const_16 = m.addConstrs(
        (1 == z[i, t])
        >>
        (1 - e[i,t] <= (quicksum([ Beta[j,t] * df.loc[i,j] for j in features ]) + Delta[t] ) * df.loc[i,labels[0]])
        for i in I
        for t in T_L
    )

    Const_18 = m.addConstrs(
        Bet_abs[f,t] == abs_(Beta[f,t])  for f in features for t in T_L
    )

    Const_19 = m.addConstr(
        quicksum([ d[t] for t in T_B]) <= Splits
    )

    m.setObjective(
        quicksum([ Bet_abs[f,t] for f in features for t in T_L]) + C * quicksum([e[i,t] for i in I for t in T_L])
        )
    start = tm()
    m.optimize()
    runtime = tm() - start

    splitting_nodes = {}
    non_empty_nodes = {}

    if m.status != GRB.INFEASIBLE:
        # m.write(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
        # print('ObjFunVal: ',m.getObjective().getValue())
        vars = m.getVars()
        solution = {
                i.VarName:i.X
                for i in vars}

        # for i in T_L:
        #     print(f'LeafNode {i}: ',solution[f'lz[{i}]'])

        non_zero_vars = [key for key,value in solution.items() if value > 0]

        if config["SplitType"] == "Parallel":
            splitting_nodes = {
                i:{
                    'a': [f for f in features if solution[f'a[{f},{i}]'] > 0][0],
                    'b': round(solution[f'b[{i}]'],6)
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
                    j: round(solution[f'Beta[{j},{i}]'],6)
                    for j in features
                },
                'Delta':round(solution[f'Delta[{i}]'],6)
            }
            for i in T_L if f'l[{i}]' in non_zero_vars
        }

        for i in splitting_nodes.items():
            print(i)
        for i in non_empty_nodes.items():
            print(i)

        # Build the optimal decision tree out of the MILP solution
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
    return ODT,runtime,C

def train_OCMT(config, Train_df, Val_df):

    # empty WARM START log file
    # open(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst', 'w').close()

    features = list(Train_df.columns.drop([config['label_name']]))
    labels = Train_df[config['label_name']].unique()
    labels = (config['label_name'], labels)

    # split train into features and labels
    X_train = Train_df.drop(columns=config['label_name'])
    X_train = X_train.to_dict('index')
    Y_train = Train_df[config['label_name']]

    # split validation set into features and labels
    X_val = Val_df.drop(columns=config['label_name'])
    X_val = X_val.to_dict('index')
    Y_val = Val_df[config['label_name']]

    best_acc = float('-inf')
    best_solution = {}
    iteration_log = {}
    RunTimeLog = {}

    for Splits in range(config['MinSplits'], config['MaxSplits'] + 1):
        RunTime_per_split = []

        args = []
        for C in [1]:#[0.1, 1, 10, 100]:
            args.append((Train_df, features, labels, Splits, C, config))
        # print(args)
        p = mp.Pool()
        result = p.starmap(optimal_CMT, args)
        for ODT, runtime, the_C in result:
            the_tree = ODT.build_tree(ODT.root.value)

            # Predict the train set
            train_pred = ODT.predict_class(X_train, the_tree)
            train_acc = round(accuracy_score(Y_train, train_pred) * 100, 2)

            # Predict the validation set
            val_pred = ODT.predict_class(X_val, the_tree)
            val_Accuracy = round(accuracy_score(Y_val, val_pred) * 100, 2)
            print(f'{config["df_name"].split(".")[0]}({config["RandomSeed"]}) Splits: {Splits}, C: {the_C} Train: {train_acc}% Val: {val_Accuracy}% -- {dt.now()}')

            iteration_log.update({
                f'{Splits, C}': {
                    'Acc (train)': train_acc,
                    'Acc (val)': val_Accuracy
                }
            })

            if val_Accuracy > best_acc:
                best_acc = val_Accuracy
                best_solution = {
                    'Splits': Splits,
                    'C': the_C,
                    'NumLeaves': len(ODT.splitting_nodes) + 1,
                    'Tree': ODT
                }
            RunTime_per_split.append(runtime)
        RunTimeLog.update({Splits: (
            round(np.average(RunTime_per_split), 2),
            round(np.std(RunTime_per_split), 2)
        )})

    return best_solution, iteration_log, RunTimeLog


if __name__ == "__main__":
    from sklearn.utils import shuffle
    from DatabaseParser import DataParser

    probType = 'Classification'
    name = 'ionosphere.arff'

    config = {
        'RandomSeed': 7,
        'ProbType': probType,
        'SplitType': 'Parallel',
        'ModelTree': True,
        'label_name': 'class',
        'TestSize': 0.2,
        'ValSize': 0.2,
        'MinSplits': 1,
        'MaxSplits': 1,
        'df_name': name,
        'Timeout': 0.5,  # for the single iteration (IN MINUTES)
        'Fraction': 1,  # fraction
        'df': DataParser(name, probType)#,False) # comment on if you want to turn off the one-hot function
    }

    # # Shuffle the dataset
    df = shuffle(config['df'], random_state=config["RandomSeed"])

    Test_df = df.iloc[:round(len(df) * config['TestSize'])]
    Val_df = df.iloc[len(Test_df): len(Test_df) + round(len(df) * config['ValSize'])]
    if config['Fraction'] == 1:
        Train_df = df.iloc[len(Test_df) + len(Val_df):]
    elif 0 < config['Fraction'] < 1:
        print(f'Reducing Training data set to {config["Fraction"]*100}% of its original length')
        Train_df = df.iloc[len(Test_df) + len(Val_df):] # todo this could have been coded a little more elegantly
        print('Original Length:',len(Train_df))
        Train_df = df.iloc[:round(len(Train_df) * config['Fraction'])]
        print('Reduced Length:', len(Train_df))
    else:
        raise ValueError('WRONG TRAINING DATASET FRACTION')

    best_solution, iteration_log, RunTimeLog = train_OCMT(
        config,
        Train_df,
        Val_df
    )

    print(f"######## TESTING ---Optimal hyperparameters for the {config['RandomSeed']}th run: NumLeaves = {best_solution['NumLeaves']}, C = {best_solution['C']}")
    ODT = best_solution['Tree']
    # Build the optimal decision tree out of the MILP solution
    the_tree = ODT.build_tree(ODT.root.value)

    # split validation set into features and labels
    X_test = Test_df.drop(columns=config['label_name'])
    X_test = X_test.to_dict('index')
    Y_test = Test_df[config['label_name']]

    test_pred = ODT.predict_class(X_test, the_tree)
    test_metric = round(accuracy_score(Y_test, test_pred) * 100, 2)
    print('     Test Accuracy: ', test_metric, '%')

    # FROM 3 TO 6 SPLITS
    # "banana0.3": 91.04,5.3
    # "banana0.5": 91.13,5
    # "banana0.7": 91.23,5
    # "banana1": 90.94,6
    # FROM 7 TO 10 SPLITS
    # "banana0.2": 92.17,8