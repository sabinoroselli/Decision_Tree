from gurobipy import *
from binarytree import build
from sklearn.utils import shuffle
from DataSetsParser import DataParser
from SupportFunctions import Parent,OptimalTree
from sklearn.metrics import zero_one_loss,hamming_loss
from time import process_time as tm
import numpy as np

def MILP_Model(df, features, labels, splits, C, timeout):

    I = df.index.values

    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(splits + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    # print(binary_tree)

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())

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

    m = Model('OCMT')
    # m.setParam('LogToConsole', 0)
    # m.setParam('Threads',1)
    # m.setParam('TimeLimit', 60 * timeout)

    # variables
    d = m.addVars(T_B, lb=0, ub=1, vtype=GRB.INTEGER, name='d')  # d_t = 1 if node splits

    a = m.addVars(features, T_B, lb=0, ub=1, vtype=GRB.INTEGER, name='a')
    b = m.addVars(T_B, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    z = m.addVars(I,T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='z') # point 'i' is in node 't'
    l = m.addVars(T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='l') # leaf 't' contains any points at all
    Beta = m.addVars(labels, features, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                     name='Beta')  # coefficient for feature i at node t
    Bet_abs = m.addVars(labels, features, T_L, vtype=GRB.CONTINUOUS,
                        name='Beta_abs')  # coefficient for feature i at node t
    Delta = m.addVars(labels, T_L, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')
    e = m.addVars(labels, I, T_L, lb=0, vtype=GRB.CONTINUOUS, name='e')

    const_1 = m.addConstrs(
            quicksum([a[j, t] for j in features]) == d[t] for t in T_B
        )

    const_2 = m.addConstrs(
        d[t] <= d[P[t]] for t in [i for i in T_B if i != root.value]
    )

    const_3 = m.addConstrs(
        z[i,t] <= l[t] for t in T_L for i in I
    )

    const_4 = m.addConstrs(
        quicksum([z[i,t] for i in I]) >= l[t] for t in T_L
    )

    const_5 = m.addConstrs(
        quicksum([z[i,t] for t in T_L]) == 1 for i in I
    )

    const_6 = m.addConstrs(
        (z[i, l] == 1)
        >>
        (quicksum([a[j, t] * (df.loc[i, j] + mu[j] - mu_min) for j in features]) + mu_min <= b[t])
        for i in I
        for l in T_L
        for t in A_l[l]
    )

    const_7 = m.addConstrs(
        (z[i, l] == 1)
        >>
        (quicksum([a[j, t] * df.loc[i, j] for j in features]) >= b[t])  # - (1 - z[i,l]) * bigM[i]
        for i in I
        for l in T_L
        for t in A_r[l]
    )

    const_8 = m.addConstrs(
        d[t] <= quicksum([ l[m] for m in D_l[t] ]) for t in T_B
    )

    const_9 = m.addConstrs(
        d[t] <= quicksum([ l[m] for m in D_r[t]]) for t in T_B
    )

    const_10 = m.addConstrs(
        (1 == z[i, t])
        >>
        (1 - e[c,i,t] <= (quicksum([ Beta[c,j,t] * df.loc[i,j] for j in features ]) + Delta[c,t] ) * df.loc[i,c])
        for i in I
        for t in T_L
        for c in labels
                )

    const_11 = m.addConstrs(
        Bet_abs[c, f, t] == abs_(Beta[c, f, t]) for c in labels for f in features for t in T_L
    )

    const_99 = m.addConstr(
        quicksum([d[t] for t in T_B]) == splits
    )

    m.setObjective(
        quicksum([Bet_abs[c, f, t] for c in labels for f in features for t in T_L])
        +
        C * quicksum([e[c, i, t] for c in labels for i in I for t in T_L
                      # if c != df.loc[i,labels[0]] # comment this off when switching to the normal SVM formulation
                      ])
    )

    start = tm()
    m.optimize()
    runtime = tm() - start

    if m.SolCount > 0:
        # print('MODEL IS FEASIBLE')
        vars = m.getVars()
        solution = {
            i.VarName: i.X
            for i in vars}

        non_zero_vars = [key for key, value in solution.items() if value > 0]

        splitting_nodes = {
            i: {
                'a': [f for f in features if solution[f'a[{f},{i}]'] > 0][0],
                'b': round(solution[f'b[{i}]'], 6)
            }
            for i in T_B if f'd[{i}]' in non_zero_vars
        }

        non_empty_nodes = {
            i: {
                c: {
                    'Beta': {
                        j: solution[f'Beta[{c},{j},{i}]']
                        for j in features
                    },
                    'Delta': solution[f'Delta[{c},{i}]']
                }
                for c in labels
            }
            for i in T_L if f'l[{i}]' in non_zero_vars
        }

        ODT = OptimalTree(
            non_empty_nodes,
            splitting_nodes,
            int(np.ceil(np.log2(splits + 1))),
            "Parallel",
            True,
            labels
        )

    else:
        print('NO SOLUTION AVAILABLE')
        ODT = None
    return ODT,runtime

if __name__ == "__main__":

    name = 'yeast.arff'
    splits = 3
    C = 1

    df,features,labels = DataParser(name, one_hot=True, toInt=False)
    df = shuffle(df,random_state=7)

    ################### COMMENT ON TO REDUCE NUMBER OF CLASSES ###############
    # to_drop = [
    #            'Class5',# 'Class6', 'Class7', 'Class8', 'Class9',
    #            'Class10', 'Class11', 'Class12', 'Class13', 'Class14']
    # df = df.drop(columns=to_drop)
    # labels = [i for i in labels if i not in to_drop] # drop labels for testing

    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    # Train_df = df.iloc[:500] # Shorten the train dataset for testing

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    ODT,_ = MILP_Model(
        df= Train_df,
        features= features,
        labels= labels,
        splits= splits,
        C=C,
        timeout= 30
    )
    if ODT != None:
        the_tree = ODT.build_tree(ODT.root.value)
        # ODT.print_tree(the_tree)

        # X_train = Train_df[features]
        # X_train = X_train.to_dict('index')
        #
        # Y_train = Train_df[labels]
        # Y_train = Y_train.to_dict('index')

        X_test = Test_df[features]
        X_test = X_test.to_dict('index')

        Y_test = Test_df[labels]
        Y_test = Y_test.to_numpy().astype(int)
        Y_test[Y_test == -1] = 0

        predictions = np.array(ODT.predict_class(X_test, the_tree)).astype(int)
        predictions[predictions == -1] = 0

        # f1_Score = classification_report(
        #     Y_test,
        #     predictions,
        #     target_names=labels,
        #     output_dict=True
        # )['weighted avg']['f1-score']
        #
        # print(round(f1_Score,2))

        loss = zero_one_loss(Y_test,predictions)
        print('Zero-One',round(loss,2))

        loss = hamming_loss(Y_test, predictions)
        print('Hamming', round(loss, 2))
    else:
        print('Gurobi could not compute a feasible model')



