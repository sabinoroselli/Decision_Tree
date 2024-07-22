from gurobipy import *
from binarytree import build
from sklearn.utils import shuffle
from DatabaseParser import DataParser

def optimal_CT(df, features, labels, depth, C):

    I = df.index.values

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

    m = Model('OCMT')
    m.setParam('NonConvex',2)
    m.setParam('DualReductions',0) # so that Gurobi knows if it is UNBOUNDED or INFEASIBLE
    # m.setParam('LogToConsole', 0)
    # m.setParam("LogFile",'GurobiLog.txt')
    m.setParam('TimeLimit', 60*60)

    # variables
    a = m.addVars(features,T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='a')
    b = m.addVars(T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    z = m.addVars(I,T_L,lb=0,ub=1,vtype=GRB.CONTINUOUS,name='z') # point 'i' is in node 't'
    Beta = m.addVars(features,T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Beta')  # coefficient for feature i at node t
    Beta1 = m.addVars(features, T_L, vtype=GRB.CONTINUOUS,name='Beta1')
    Beta2 = m.addVars(features, T_L, vtype=GRB.CONTINUOUS, name='Beta2')
    Delta = m.addVars(T_L,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Delta')  # constant at node t
    e = m.addVars(I,T_L, lb=0, vtype=GRB.CONTINUOUS, name='e')

    # Load previous solution for warm start
    # m.update()
    # m.read('PrevSol.mst')

    Const_8 = m.addConstrs(
        quicksum([z[i,t] for t in T_L]) == 1 for i in I
    )

    Const_12 = m.addConstrs(
        z[i, l] <= quicksum([ a[j,t] * (df.loc[i, j] + mu[j] - mu_min) for j in features ]) + mu_min - b[t]
        for i in I
        for l in T_L
        for t in A_l[l]
    )

    Const_13 = m.addConstrs(
        z[i, l] >= quicksum([a[j,t] * df.loc[i, j] for j in features ]) - b[t]
        for i in I
        for l in T_L
        for t in A_r[l]
    )

    Const_16 = m.addConstrs(
        (quicksum([ Beta[j,t] * df.loc[i,j] for j in features ]) + Delta[t] ) * df.loc[i,labels[0]] >= 1 - e[i,t]
        for i in I
        for t in T_L
    )

    Const_18 = m.addConstrs(
        Beta[f,t] == Beta1[f,t] - Beta2[f,t]  for f in features for t in T_L
    )

    m.setObjective(
        quicksum([ Beta1[f,t] + Beta2[f,t] for f in features for t in T_L]) + C * quicksum([ e[i,t] * z[i,t]  for i in I for t in T_L])
        )
    m.optimize()

    solution = {}

    if m.status != GRB.INFEASIBLE:
        m.write('PrevSol.mst')
        vars = m.getVars()
        solution = {
                i.VarName:i.X
                for i in vars
                if i.VarName[0] not in ['z','e']
                if i.X != 0
                }

    else:
        print('MODEL IS INFEASIBLE')
    return solution

if __name__ == "__main__":

    label_name = 'class'
    probType = 'Classification'
    name = 'ionosphere.arff'

    df = DataParser(name, probType)

    # print(df.head(20).to_markdown())

    df = shuffle(df,random_state=7)
    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    Train_df = Train_df.sample(n=round(len(df)*0.03)) # todo if you want to reduce the trainset
    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    features = list(Train_df.columns.drop([label_name]))
    labels = df[label_name].unique()
    labels = (label_name, labels)

    depth = 3

    solution = optimal_CT(
        df= Train_df,
        features= features,
        labels= labels,
        depth= depth,
        C= 1
    )

    for i in solution.items():
        print(i)

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



