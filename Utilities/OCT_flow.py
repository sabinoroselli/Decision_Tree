from gurobipy import *
from binarytree import build
from sklearn.utils import shuffle
from DatabaseParser import DataParser
from Utilities.OldTrees.TreeStructure import Parent,Children
import numpy as np

def optimal_CT(df, features, labels, splits):

    # depth of the tree does not account for root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(Splits + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    print(binary_tree)

    T_l = [i.value for i in binary_tree.leaves]  # leave nodes
    T_b = [i for i in binary_tree.values if i not in T_l] # branch nodes

    T = T_l + T_b

    T_g = T + ['s','w']


    J = features # just renaming according to the paper

    K = labels[1]

    I = df.index.values

    A = {
        i: Parent(root, i)  if i != root.value else 's' for i in binary_tree.values
    }

    l = {
        i:Children(root,i)[0].value for i in T_b
    }

    r = {
        i: Children(root,i)[1].value for i in T_b
    }

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())
    mu_max = max(mu.values())

    df_max = max([abs(df[j].max()) for j in J]) # larger element in the dataset

    m = Model('OCT_flow')
    # m.setParam('OutputFlag', 0)
    # m.setParam('TimeLimit', 60*60)

    # variables

    g = m.addVars(K,T,vtype=GRB.BINARY, name='g') # if node t is assigned class k
    a = m.addVars(J, T_b, vtype=GRB.BINARY, name='a')  # if node t splits on feature j
    b = m.addVars(T_b,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name='b')  # value of the split at node t
    u = m.addVars(I,T_g,T_g,vtype=GRB.BINARY, name='u') # point i flows from node t to t'

    const_a = m.addConstrs(
        quicksum([ a[j,t] for j in J ]) + quicksum([ g[k,t] for k in K ]) == 1
        for t in T_b
    )

    # not sure I actually need this constraint but it does make convergence faster
    m.addConstrs(
          quicksum([a[j,t] for j in J ]) * df_max >= b[t] for t in T_b
    )

    const_b = m.addConstrs(
        quicksum([ g[k,t] for k in K ]) <= 1 for t in T_l
    )

    m.addConstrs(
        (1 == u[i, t, l[t]])
        >>
        (quicksum([ a[j,t] * (df.loc[i, j] + mu[j] - mu_min) for j in J]) + mu_min
        <=
        b[t] )
        for t in T_b for i in I
    )

    m.addConstrs(
        (1 == u[i, t, r[t]])
        >>
        (quicksum([a[j, t] * df.loc[i, j]  for j in J]) >= b[t] )
        for t in T_b for i in I
    )

    constr_c = m.addConstrs(
        u[i,A[t],t] == u[i,t,l[t]] + u[i,t,r[t]] + u[i,t,'w']
        for t in T_b
        for i in I
    )

    constr_d = m.addConstrs(
        u[i,A[t],t] == u[i,t,'w'] for t in T_l for i in I
    )

    const_e = m.addConstrs(
        u[i, t, l[t]] <= quicksum([a[j, t] for j in J]) for i in I for t in T_b
    )

    const_f = m.addConstrs(
        u[i, t, r[t]] <= quicksum([a[j, t] for j in J]) for i in I for t in T_b
    )

    constr_g = m.addConstrs(
        u[i,t,'w'] <= g[df.loc[i,label_name],t] for i in I for t in T
    )

    m.addConstr(
        quicksum([ a[j,t] for j in J for t in T_b]) <= splits
    )

    m.setObjective(
        quicksum([ u[i,t,'w'] for t in T for i in I]),GRB.MAXIMIZE
    )

    # m.addConstrs(
    #     2 - 2 * quicksum([ g[k,t] for k in K])
    #     >=
    #     quicksum([ g[k,l[t]] for k in K]) + quicksum([ g[k,r[t]] for k in K])
    #     for t in T_b
    # )

    m.optimize()

    solution = {}
    if m.status != GRB.INFEASIBLE:
        vars = m.getVars()
        print(f"Acc: {round(m.ObjVal/len(df),2) * 100}%")
        for i in vars:
            if i.X != 0:
                if i.VarName[0] in ['g','a','b']:
                #     # print(i.VarName,i.X)
                    solution.update({i.VarName: i.X})

    else:
        print('MODEL IS INFEASIBLE')
    return solution


if __name__ == "__main__":

    df = DataParser('autoUnivMulti.arff','Classification',one_hot=True,toInt=False)
    label_name = 'class'
    df = shuffle(df,random_state=7)
    # df = df.iloc[:10]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in df.columns:
        if df[i].nunique() == 1:
            df.drop(columns=[i], inplace=True)


    features = list(df.columns.drop([label_name]))
    labels = df[label_name].unique()
    labels = (label_name, labels)

    print(df)

    Splits = 2

    solution = optimal_CT(
        df=df,
        features=features,
        labels=labels,
        splits=Splits
    )

    for i in solution.items():
        print(i)

