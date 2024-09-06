import numpy as np
from time import process_time as tm
from gurobipy import *
from binarytree import build
from TreeStructure import OptimalTree,Parent,RAE,RRSE
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from DatabaseParser import DataParser

def optimal_OMT(df, features, labels, Splits, C, config):

    if config['Meta']:
        branchFeat = features[0]  # for branch nodes
        leafFeat = features[1]  # for leaf nodes
    else:
        branchFeat = features
        leafFeat = features

    I = df.index.values

    classes = df[labels[0]].unique()

    LabelsPerClass = {
        c: {
            i: 1 if df.loc[i, labels[0]] == c else -1
            for i in I
        }
        for c in classes
    }

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in branchFeat
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
    m.setParam('LogToConsole', 0)
    m.setParam('Threads',1)
    m.setParam("LogFile",f'GurobiLogs/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.txt')
    m.setParam('TimeLimit', 60 * config['Timeout'])

    # variables
    d = m.addVars(T_B,lb=0,ub=1,vtype=GRB.INTEGER,name='d') # d_t = 1 if node splits
    if config["SplitType"] == "Parallel":
        a = m.addVars(branchFeat,T_B,lb=0,ub=1,vtype=GRB.INTEGER,name='a')
    elif config["SplitType"] == "Oblique":
        a = m.addVars(branchFeat, T_B, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='a')
        a_abs = m.addVars(branchFeat, T_B, vtype=GRB.CONTINUOUS, name='a_abs')
        s = m.addVars(branchFeat, T_B,lb=0,ub=1, vtype=GRB.INTEGER, name='s')
    b = m.addVars(T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    z = m.addVars(I,T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='z') # point 'i' is in node 't'
    l = m.addVars(T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='l') # leaf 't' contains any points at all
    if config['ProbType'] == 'Classification' and len(classes) == 2:
        Beta = m.addVars(leafFeat, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Beta')  # coefficient for feature i at node t
        Bet_abs = m.addVars(leafFeat, T_L, vtype=GRB.CONTINUOUS, name='Beta_abs')  # coefficient for feature i at node t
        Delta = m.addVars(T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')
        e = m.addVars(I,T_L, lb=0, vtype=GRB.CONTINUOUS, name='e')
    if config['ProbType'] == 'Classification' and len(classes) > 2:
        Beta = m.addVars(classes,leafFeat, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Beta')  # coefficient for feature i at node t
        Bet_abs = m.addVars(classes,leafFeat, T_L, vtype=GRB.CONTINUOUS, name='Beta_abs')  # coefficient for feature i at node t
        Delta = m.addVars(classes,T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')
        e = m.addVars(classes,I,T_L, lb=0, vtype=GRB.CONTINUOUS, name='e')
    elif config['ProbType'] == 'Regression':
        Beta = m.addVars(leafFeat, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Beta')  # coefficient for feature i at node t
        Bet_abs = m.addVars(leafFeat, T_L, vtype=GRB.CONTINUOUS, name='Beta_abs')  # coefficient for feature i at node t
        Delta = m.addVars(T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')
        e = m.addVars(I, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='e')
        e_abs = m.addVars(I, T_L, vtype=GRB.CONTINUOUS, name='e_abs')

    # Load previous solution for warm start
    m.update()
    try:
        m.read(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
    except:
        # print('NO WARM START')
        pass
    # else:
    #     print('USING WARM START')

    if config["SplitType"] == "Parallel":
        Const_1 = m.addConstrs(
            quicksum([a[j, t] for j in branchFeat]) == d[t] for t in T_B
        )
    elif config["SplitType"] == "Oblique":
        Const_0 = m.addConstrs(
            a_abs[j, t] == abs_(a[j, t]) for j in branchFeat for t in T_B
        )

        Const_01 = m.addConstrs(
            s[j, t] >= a_abs[j, t] for j in branchFeat for t in T_B
        )
        # Guarantee that the sum of fractions for the split is equal to 1
        Const_1 = m.addConstrs(
            quicksum([a_abs[j, t] for j in branchFeat]) <= d[t] for t in T_B
        )

        Const_11 = m.addConstrs(
            s[j, t] <= d[t] for j in branchFeat for t in T_B
        )

        Const12 = m.addConstrs(
            quicksum([s[j, t] for j in branchFeat]) >= d[t] for t in T_B
        )

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
            (quicksum([a[j, t] * (df.loc[i, j] + mu[j] - mu_min) for j in branchFeat]) + mu_min <= b[t])
            for i in I
            for l in T_L
            for t in A_l[l]
        )

    elif config["SplitType"] == "Oblique":
        Const_12 = m.addConstrs(
            (z[i, l] == 1)
            >>
            (quicksum([a[j, t] * df.loc[i, j] for j in branchFeat]) + 0.0001 <= b[t])
            for i in I
            for l in T_L
            for t in A_l[l]
        )

    Const_13 = m.addConstrs(
        (z[i, l] == 1)
        >>
        (quicksum([a[j, t] * df.loc[i, j] for j in branchFeat]) >= b[t])  # - (1 - z[i,l]) * bigM[i]
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

    if config['ProbType'] == 'Classification':
        if len(classes) > 2:
            if config['WW']:
                m.addConstrs(
                    (1 == z[i, t])
                    >>
                    ( quicksum([Beta[df.loc[i, labels[0]], j, t] * df.loc[i, j] for j in leafFeat]) + Delta[df.loc[i, labels[0]], t]
                      >=
                      quicksum([Beta[m, j, t] * df.loc[i, j] for j in leafFeat]) + Delta[m, t] + 2 - e[m, i, t])
                    for i in I
                    for m in classes
                    for t in T_L
                    if m != df.loc[i, labels[0]]
                )
            else:
                m.addConstrs(
                    (1 == z[i, t])
                    >>
                    (1 - e[c,i,t] <= (quicksum([ Beta[c,j,t] * df.loc[i,j] for j in leafFeat ]) + Delta[c,t] ) * LabelsPerClass[c][i])
                    for i in I
                    for t in T_L
                    for c in classes
                )

            # comment this on if you want the sum-to-zero constraint
            # sum_to_zero = m.addConstrs(
            #     (1 == z[i, t])
            #     >>
            #     (quicksum([ quicksum([Beta[c,f] * df.loc[i,f] for f in features]) + Delta[c] for c in classes]) == 0)
            #     for i in I
            #     for t in T_L
            # )

        else:
            m.addConstrs(
                (1 == z[i, t])
                >>
                (1 - e[i, t] <= (quicksum([Beta[j, t] * df.loc[i, j] for j in leafFeat]) + Delta[t]) * df.loc[i, labels[0]])
                for i in I
                for t in T_L
            )
    elif config['ProbType'] == 'Regression':
        Const_16 = m.addConstrs(
            (z[i, t] == 1)
            >>
            (quicksum([Beta[j, t] * df.loc[i, j] for j in leafFeat]) + Delta[t] - df.loc[i, labels[0]] == e[i, t])
            for i in I
            for t in T_L
        )

        Const_17 = m.addConstrs(
            e_abs[i, t] == abs_(e[i, t]) for i in I for t in T_L
        )
    if config['ProbType'] == 'Classification' and len(classes) > 2:
        Const_18 = m.addConstrs(
            Bet_abs[c,f,t] == abs_(Beta[c,f,t])  for c in classes for f in leafFeat for t in T_L
        )
    else:
        Const_18 = m.addConstrs(
            Bet_abs[f, t] == abs_(Beta[f, t]) for f in leafFeat for t in T_L
        )

    Const_19 = m.addConstr(
        quicksum([d[t] for t in T_B]) <= Splits
    )
    if config['Meta']:
        bounds = {
            i: [df[i].min(), df[i].max()] for i in branchFeat
        }

        Const_200 = m.addConstrs(
            (a[j, t] == 1) >> (b[t] <= bounds[j][1])
            for t in T_B for j in branchFeat
        )
        Const_201 = m.addConstrs(
            (a[j, t] == 1) >> (b[t] >= bounds[j][0])
            for t in T_B for j in branchFeat
        )

    if config['ProbType'] == 'Classification' and len(classes)>2:
        if config['WW']:
            m.setObjective(
                quicksum([Bet_abs[c, f, t] for c in classes for f in leafFeat for t in T_L])
                +
                C * quicksum([e[c, i, t] for c in classes for i in I for t in T_L
                              if c != df.loc[i,labels[0]]
                              ])
            )
        else:
            m.setObjective(
                quicksum([ Bet_abs[c,f,t] for c in classes for f in leafFeat for t in T_L])
                +
                C * quicksum([e[c,i,t] for c in classes for i in I for t in T_L
                              # if c != df.loc[i,labels[0]] # comment this off when switching to the normal SVM formulation
                              ])
                )
    elif config['ProbType'] == 'Classification' and len(classes) == 2:
        m.setObjective(
            quicksum([Bet_abs[f, t] for f in leafFeat for t in T_L]) + C * quicksum([e[i, t] for i in I for t in T_L])
        )
    elif config['ProbType'] == 'Regression':
        m.setObjective(
            quicksum([Bet_abs[f, t] for f in leafFeat for t in T_L]) + C * quicksum([e_abs[i, t] for i in I for t in T_L])
        )
    start = tm()
    m.optimize()
    runtime = tm() - start

    splitting_nodes = {}

    if m.status != GRB.INFEASIBLE:
        m.write(f'WarmStarts/{config["df_name"].split(".")[0]}_{config["RandomSeed"]}.mst')
        vars = m.getVars()
        solution = {
                i.VarName:i.X
                for i in vars}

        non_zero_vars = [key for key,value in solution.items() if value > 0]

        if config["SplitType"] == "Parallel":
            splitting_nodes = {
                i:{
                    'a': [f for f in branchFeat if solution[f'a[{f},{i}]'] > 0][0],
                    'b': round(solution[f'b[{i}]'],6)
                }
                for i in T_B if f'd[{i}]' in non_zero_vars
            }
        elif config["SplitType"] == "Oblique":
            splitting_nodes = {
                i: {
                    'a': {f: round(solution[f'a[{f},{i}]'], 6)
                          for f in branchFeat
                          },
                    'b': round(solution[f'b[{i}]'], 6)
                }
                for i in T_B if f'd[{i}]' in non_zero_vars
            }
        if config['ProbType'] == 'Classification' and len(classes) == 2 or config['ProbType'] == 'Regression':
            non_empty_nodes = {
                i: {
                    'Beta': {
                        j: round(solution[f'Beta[{j},{i}]'], 6)
                        for j in leafFeat
                    },
                    'Delta': round(solution[f'Delta[{i}]'], 6)
                }
                for i in T_L if f'l[{i}]' in non_zero_vars
            }
        else:
            non_empty_nodes = {
                i:{
                    c:{
                        'Beta':{
                            j: round(solution[f'Beta[{c},{j},{i}]'],6)
                            for j in leafFeat
                        },
                        'Delta':round(solution[f'Delta[{c},{i}]'],6)
                    }
                    for c in classes
                }
                for i in T_L if f'l[{i}]' in non_zero_vars
            }

        ODT = OptimalTree(
            non_empty_nodes,
            splitting_nodes,
            int(np.ceil(np.log2(Splits + 1))),
            config["SplitType"],
            config["ModelTree"],
            classes
        )

    else:
        print('MODEL IS INFEASIBLE')
        ODT = None

    return ODT,runtime

if __name__ == "__main__":

    ProbType = 'Classification'
    # ProbType = 'Regression'
    # file = 'autoUnivMulti'
    file = 'iris'
    Splits = 1

    config ={
        'RandomSeed':7,
        'ProbType': ProbType,
        "ModelTree": True,
        'SplitType': 'Parallel',
        'label_name': 'class',
        'TestSize': 0.2,
        'df_name': file,
        'Timeout': 60,  # for the single iteration (IN MINUTES)
        'Fraction': 1,  # fraction
        'Meta': False,
        'WW':False
    }

    df = DataParser(f'{file}.arff',ProbType, one_hot=True)

    df = shuffle(df,random_state=config['RandomSeed'])

    # Test_df = df.iloc[:round(len(df) * config['TestSize'])]
    # Train_df = df.iloc[len(Test_df):]
    ################### STRATIFIED SPLIT ############################################################
    Test_df = df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=config['TestSize'],
                                                                             random_state=config['RandomSeed']))
    Train_df = df[~df.index.isin(Test_df.index)]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            print(i)
            Train_df = Train_df.drop(columns=[i])
            Test_df = Test_df.drop(columns=[i])

    features = list(Train_df.columns.drop(['class']))
    labels = df['class'].unique()
    labels = ('class', labels)

    for C in [1]:#[0.1, 1, 10, 100, 1000]:
        ODT,runtime = optimal_OMT(
            df= Train_df,
            features= features,
            labels= labels,
            Splits= Splits,
            C= C,
            config=config
        )

        print('Runtime:',round(runtime,3),end=" ")
        print('C:',C,end=' ')
        the_tree = ODT.build_tree(ODT.root.value)

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
            print('Train:', round(accuracy_score(Y_train, train_pred) * 100, 2), '%',end=' ')
        elif ProbType == 'Regression':
            train_pred = ODT.predict_regr(X_train, the_tree, None)
            print('Train -- RAE:', RAE(Y_train, train_pred),'RRSE:', RRSE(Y_train, train_pred))

        # Predict the test set
        if ProbType == 'Classification':
            test_pred = ODT.predict_class(X_test, the_tree,None)
            print('Test:', round(accuracy_score(Y_test, test_pred)*100,2),'%')
        elif ProbType == 'Regression':
            test_pred = ODT.predict_regr(X_test, the_tree, None)
            print('Test -- RAE:', RAE(Y_test, test_pred),'RRSE:', RRSE(Y_test, test_pred))



