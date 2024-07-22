from Utilities.OldTrees.TreeStructure import Parent
from binarytree import build
from sklearn.utils import shuffle
# import pandas as pd
# import json
from DatabaseParser import DataParser
# import random
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING,Pricer

class OCMT_pricer(Pricer):
    def pricerredcost(self, *args, **kwargs):
        # Retrieving the dual solutions
        dualSolutions = []
        for i, c in enumerate(self.data['cons']):
            dualSolutions.append(self.model.getDualsolLinear(c))
        # Building a MIP to solve the subproblem
        subMIP = Model("SubOCMT")

        # Turning off presolve
        subMIP.setPresolve(SCIP_PARAMSETTING.OFF)

        # Setting the verbosity level to 0
        subMIP.hideOutput()

        # The initialisation function for the variable pricer to retrieve the transformed constraints of the problem
    def pricerinit(self):
        for i, c in enumerate(self.data['cons']):
            self.data['cons'][i] = self.model.getTransformedCons(c)


def optimal_CT(df, features, labels, depth, Splits, C):

    F = features

    I = df.index.values

    bigM = max([df[i].max() for i in features])

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())

    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (depth + 1) - 1)]
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

    master = Model('OCMT')
    pricer = OCMT_pricer()
    master.includePricer(pricer,"","")

    # variables
    d = {t: master.addVar(vtype='B', name=f'd[{t}]') for t in T_B}
    a = {(f, t): master.addVar(vtype='B',name=f'a[{f, t}]') for f in F for t in T_B}
    b = {t: master.addVar(vtype='C',lb=float("-inf"), name=f'b[{t}]') for t in T_B}
    z = {(i, t): master.addVar(vtype='B', name=f'z[{i, t}]') for i in I for t in T_L}
    l = {t: master.addVar(vtype='B', name=f'l[{t}]') for t in T_L}
    Beta = {(f, t): master.addVar(vtype='C',lb=float("-inf"), name=f'Beta[{f, t}]') for f in F for t in T_L}
    Beta1 = {(f, t): master.addVar(vtype='C', name=f'Beta1[{f, t}]') for f in F for t in T_L}
    Beta2 = {(f, t): master.addVar(vtype='C', name=f'Beta2[{f, t}]') for f in F for t in T_L}
    Delta = {t: master.addVar(vtype='C',lb=float("-inf"), name=f'Delta[{t}]') for t in T_L}
    e = {(i, t): master.addVar(vtype='C',lb=0, name=f'e[{i, t}]') for i in I for t in T_L}

    master.data = d,a,b,z,l,Beta,Delta,e

    for t in T_B:

        master.addCons( quicksum([a[j,t] for j in features]) == d[t] )

        master.addCons( b[t] <= bigM * d[t] )

        master.addCons( b[t] >= -bigM * d[t] )

        master.addCons( d[t] <= quicksum([ l[m] for m in D_l[t] ]) )

        master.addCons( d[t] <= quicksum([ l[m] for m in D_r[t]]) )

    for t in [i for i in T_B if i != root.value]:

        master.addCons( d[t] <= d[P[t]] )

    for i in I:

        master.addCons( quicksum([z[i, t] for t in T_L]) == 1 )

        for t in T_L:
            master.addCons( z[i,t] <= l[t] )

            master.addCons( 1 - e[i,t] <= (quicksum([ Beta[j,t] * df.loc[i,j] for j in features ]) + Delta[t] ) * df.loc[i,labels[0]] + bigM * (1 - z[i,t]) )

            for f in features:

                master.addCons( Beta[f,t] == Beta1[f,t] - Beta2[f,t]  )

            for t1 in A_l[t]:

                master.addCons( quicksum([ a[j,t1] * (df.loc[i, j] + mu[j] - mu_min) for j in features ]) + mu_min <= b[t1] + bigM * (1 - z[i,t]) )

            for t1 in A_r[t]:

                master.addCons( quicksum([a[j,t1] * df.loc[i, j] for j in features ]) >= b[t1] - bigM * (1 - z[i,t]) )

        master.addCons( quicksum([z[i,t] for i in I]) >= l[t] )

    master.addCons( quicksum([ d[t] for t in T_B]) <= Splits )

    master.setObjective( quicksum([ Beta1[f,t] + Beta2[f,t] for f in features for t in T_L]) + C * quicksum([e[i,t] for i in I for t in T_L]) )

    master.optimize()

    splitting_nodes = {}
    non_empty_nodes = {}

    if master.getStatus() == 'optimal':
        d,a,b,z,l,Beta,Delta,e = master.data

        splitting_nodes = {
            t: {
                'a': [round(master.getVal(a[f, t]), 2) for f in F],
                'b': round(master.getVal(b[t]), 2)
            }
            for t in T_B
        }

        non_empty_nodes = {
            t: {
                'Beta': [round(master.getVal(Beta[f, t]), 2) for f in F],
                'Delta': round(master.getVal(Delta[t]), 2)
            }
            for t in T_L
        }

        print('OPT: ', master.getObjVal())

    else:
        print('MODEL IS INFEASIBLE')
    return splitting_nodes,non_empty_nodes

if __name__ == "__main__":

    label_name = 'class'

    df = DataParser('blogger-simplified.arff', one_hot=True)

    df = shuffle(df,random_state=7)
    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    # Train_df = Train_df.sample(n=round(len(df)*0.03)) # todo if you want to reduce the trainset
    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    features = list(Train_df.columns.drop([label_name]))
    labels = df[label_name].unique()
    labels = (label_name, labels)

    depth = 2
    Splits = 3

    splitting_nodes,non_empty_nodes = optimal_CT(
        df= Train_df,
        features= features,
        labels= labels,
        depth= depth,
        Splits= Splits,
        C= 1
    )

    print('Splitting Nodes')
    for i in splitting_nodes.items():
        print(i[0],i[1])
    print('Non-Empty Nodes')
    for i in non_empty_nodes.items():
        print(i[0],i[1])

    # ODT = OptimalModelTree(non_empty_nodes, splitting_nodes, depth)
    # the_tree = ODT.build_tree(ODT.root.value)
    # # ODT.print_tree(the_tree)
    #
    # # split train into features and labels
    # X_train = Train_df.drop(columns=label_name)
    # X_train = X_train.to_dict('index')
    # Y_train = Train_df[label_name]
    #
    # # Predict the train set
    # train_pred = ODT.predict_class(X_train, the_tree)
    #
    # print('Accuracy (Train Set): ', round(accuracy_score(Y_train, train_pred) * 100, 2), '%')
    # # split test set into features and labels
    # X_test = Test_df.drop(columns=label_name)
    # X_test = X_test.to_dict('index')
    # Y_test = Test_df[label_name]
    #
    # # Predict the test set
    # test_pred = ODT.predict_class(X_test, the_tree)
    #
    # # for ind,i in enumerate(list(Y_test)):
    # #     print(i,test_pred[ind])
    #
    # print('Accuracy (Test Set): ', round(accuracy_score(Y_test, test_pred)*100,2),'%')



