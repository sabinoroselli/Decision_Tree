from gurobipy import *
from binarytree import build
from sklearn.utils import shuffle
from DataSetsParser import DataParser
import numpy as np

def optimal_CT(df, features, labels, C):

    df = df.iloc[:10]

    print(df)

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

    m = Model('OCMT')
    m.setParam('NonConvex',2)
    m.setParam('DualReductions',0) # so that Gurobi knows if it is UNBOUNDED or INFEASIBLE
    m.setParam('LogToConsole', 0)
    # m.setParam("LogFile",'GurobiLog.txt')
    m.setParam('TimeLimit', 60*60)

    # variables
    a = m.addVars(features,T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='a')
    b = m.addVars(T_B,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    z = m.addVars(I,T_L,lb=0,vtype=GRB.CONTINUOUS,name='z') # point 'i' is in node 't'
    c = m.addVars(labels,T_L,lb=0,ub=1,vtype=GRB.INTEGER,name='c')
    z_int = m.addVars(I, T_L, lb=0, ub=1, vtype=GRB.INTEGER, name='z_int')
    h = m.addVars(labels,I,lb=0,ub=1,vtype=GRB.INTEGER,name='h') # class 'k' is correctly predicted for 'i'

    # Const_1 = m.addConstrs(
    #     quicksum([z[i,t] for t in T_L]) == 1 for i in I
    # )

    Const_2 = m.addConstrs(
        z[i, l] <= quicksum([ a[j,t] * (df.loc[i, j] + mu[j] - mu_min) for j in features ]) + mu_min - b[t]
        for i in I
        for l in T_L
        for t in A_l[l]
    )

    Const_3 = m.addConstrs(
        z[i, l] >= quicksum([a[j,t] * df.loc[i, j] for j in features ]) - b[t]
        for i in I
        for l in T_L
        for t in A_r[l]
    )

    Const_4 = m.addConstrs(
        quicksum([c[l,t] for t in T_L]) == 1 for l in labels
    )

    Const_5 = m.addConstrs(
        quicksum([c[l,t] for l in labels ]) <= 1  for t in T_L
    )

    Const_6 = m.addConstrs(
        z_int[i,t] >= z[i,t] - 0.5 for i in I for t in T_L
    )

    Const_7 = m.addConstrs(
        z_int[i, t] <= z[i, t] + 0.5 for i in I for t in T_L
    )

    # Const_8 = m.addConstrs(
    #
    # )
    #
    # m.setObjective(
    #     )
    m.optimize()

    solution = {}

    if m.status != GRB.INFEASIBLE:
        m.write('PrevSol.mst')
        vars = m.getVars()
        solution = {
                i.VarName:i.X
                for i in vars
                if i.VarName[0] not in ['e']
                if i.X != 0
                }

    else:
        print('MODEL IS INFEASIBLE')
    return solution

if __name__ == "__main__":

    label_name = 'Class'
    probType = 'Classification'
    name = 'yeast.arff'

    df = DataParser(name, probType)

    # print(df.head(20).to_markdown())

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

    depth = 2

    solution = optimal_CT(
        df= Train_df,
        features= features,
        labels= labels,
        C= 1
    )

    for i in solution.items():
        print(i)