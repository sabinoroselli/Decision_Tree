from z3 import *
from binarytree import bst
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def Ancestors(root,target):
    ancestors = []
    def findAncestors(root, target):
        # Base case
        if root == None:
            return False
    
        if root.value == target:
            return True
    
        # If target is present in either left or right subtree
        # of this node, then print this node
        if (findAncestors(root.left, target) or
                findAncestors(root.right, target)):
            ancestors.append(root.value)
            # print(root.value,end=' ')
            return True
    
        # Else return False
        return False
    findAncestors(root,target)
    return ancestors



def Parent (node,val):    
    the_parent = []
    def findParent(node, val, parent = None):
        if (node is None):
            return 
    
        # If current node is the required node
        if (node.value == val):
            # assign its parent
            the_parent.append(parent)
            
        else:
            # Recursive calls for the children of the current node. current node is now the new parent
            findParent(node.left, val, node.value)
            findParent(node.right, val, node.value)
            
    findParent(node,val)
    return the_parent[0]

def optimal_DT(df,D,alpha,N_min):

    num_points = len(df)
    binary_tree = bst(D,True) # create the maximal tree of depth D
    root = binary_tree.levels[0][0]
    T = binary_tree.values # number of nodes of the maximal tree of depth D
    T_L = [i.value for i in binary_tree.leaves] # leave nodes
    T_B = [i for i in binary_tree.values if i not in T_L]
    print(binary_tree)
    # print(T_L)
    # print(T_B)

    A = {
        i:Ancestors(root,i)  for i in binary_tree.values
    }

    A_l = {
        key:[second for first,second in zip([key]+value[:-1],value) if first < second]
        for key,value in A.items()
    }

    A_r = {
        key:[second for first,second in zip([key]+value[:-1],value) if first > second]
        for key,value in A.items()
    }

    P = {
        i:Parent(root,i)  for i in binary_tree.values
    }

    # binary variables
    d = { t:Int(f'd_{t}') for t in T_B } # d_t = 1 if node splits
    a = { (j,t):Int(f'a_{j}_{t}') for t in T_B for j in features}
    b = { t:Real(f'b_{t}') for t in T_B }
    z = { (i,t):Int(f'z_{i}_{t}') for t in T_L for i in range(num_points)} # point 'i' is in node 't'
    l = { t:Int(f'l_{t}') for t in T_L } # leaf 't' contains any points at all
    c = { (k,t):Int(f'c_{k}_{t}') for t in T_L for k in labels} # label of node t
    N_k = { (k,t):Int(f'n_{k}_{t}') for t in T_L for k in labels} # number of points of label k in node t
    N = { t:Int(f'N_{t}') for t in T_L } # number of points in node t
    L = { t:Int(f'L_{t}') for t in T_L } # number of points in node t minus the number of points of the most common label

    Const_1_1 = [
        And(
            l[t]<=1,
            l[t]>=0
        )
        for t in T_L
    ]

    Const_1_2 = [
        And(
            c[k,t] <= 1,
            c[k,t] >= 0
        )
        for t in T_L for k in labels
    ]

    Const_1_3 = [
        And(
            z[i,t] <= 1,
            z[i,t] >= 0
        )
        for t in T_L for i in range(num_points)
    ]

    Const_1 = Const_1_1 + Const_1_2 + Const_1_3

    Const_2 = [
        Sum([a[j,t] for j in features ]) == d[t] for t in T_B
    ]

    Const_3 = [
        And(
            b[t] >= 0,
            b[t] <= d[t],
            d[t] <= 1
        )
        for t in T_B
    ]

    Const_4 = [
        And(
            a[j,t] >= 0,
            a[j,t] <= 1
        )
        for j in features
        for t in T
    ]
    Const_5 = [
        d[t] <= d[P[t]] for t in [i for i in T_B if i != root.value]
    ]

    Const_6 = [
        z[i,t] <= l[t] for t in T_L for i in range(num_points)
    ]

    Const_7 = [
        Sum([z[i,t] for i in range(num_points)]) >= N_min * l[t] for t in T_L
    ]
    # each point to exactly one leaf
    Const_8 = [
        Sum([z[i,t] for t in T_L]) == 1 for i in range(num_points)
    ]

    Const_13 = [
        Implies(
                z[i,t] == 1,
                Sum([ a[j,m] * df.loc[i,j] for j in features ]) >= b[m]
        )
        for i in range(num_points)
        for t in T_L
        for m in A_r[t]
    ]

    Const_14 = [
        Implies(
            z[i,t] == 1,
            Sum([a[feature,m] * df.loc[i,feature] for feature in features ]) < b[m]
        )
        for i in range(num_points)
        for t in T_L
        for m in  A_l[t]
    ]

    Const_15 = [
        N_k[k,t] == Sum([ z[i,t] for i in range(num_points) if k == df.loc[i,'type'] ])
        for k in labels
        for t in T_L
    ]

    Const_16 = [
        N[t] == Sum([ z[i,t] for i in range(num_points)]) for t in T_L
    ]

    Const_18 = [
        Sum([c[k,t] for k in labels]) == l[t] for t in T_L
    ]

    Const_20 = [
        L[t] >= N[t] - N_k[k, t] - num_points * (1 - c[k, t])
        for k in labels
        for t in T_L
    ]

    Const_21 = [
        L[t] <= N[t] - N_k[k, t] + num_points * c[k, t]
        for k in labels
        for t in T_L
    ]

    Const_22 = [
        L[t] >= 0 for t in T_L
    ]

    L_star = max(df['type'].value_counts()) # most popular class in the entire dataset ( the number of occurrences)

    s = Optimize()
    set_option(rational_to_decimal=True)
    set_option(precision=4)
    set_option(verbose=2)
    s.add(
        Const_1 +
        Const_2 +
        Const_3 +
        Const_4 +
        Const_5 +
        Const_6 +
        Const_7 +
        Const_8 +
        Const_13 +
        Const_14 +
        Const_15 +
        Const_16 +
        Const_18 +
        Const_20 +
        Const_21 +
        Const_22
    )

    s.minimize(
        1 / L_star * Sum([L[t] for t in T_L])
        +
        alpha * Sum([d[t] for t in T_B])
    )

    print(s.check())
    solution = {}
    if s.check() == sat:
        m = s.model()
        for i in m:
            # if i.name()[0] not in ['z','c']:
            #     if float(str(m[i]).split('?')[0]) > 0:
                    # print(i,m[i])
                    solution.update({i.name():m[i]})
    return solution

if __name__ == "__main__":

    df = pd.read_csv('flowers/iris.data')
    features = list(df.columns.drop(['type']))
    scaler = MinMaxScaler()
    df_scaled = df.drop(columns='type')
    df_scaled = scaler.fit_transform(df_scaled.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)
    df_scaled.insert(len(features), 'type', df['type'], True)
    df = df_scaled
    labels = df.type.unique()

    D = 2
    alpha = 0.3
    N_min = 3

    solution = optimal_DT(df, D, alpha, N_min)

    for a,b in solution.items():
        if a[0] == 'L':
            print(a,b)
    for a,b in solution.items():
        if a[0] == 'N':
            print(a,b)
    for a,b in solution.items():
        if a[0] == 'd':
            print(a,b)
    for a,b in solution.items():
        if a[0] == 'a':
            print(a,b)
    for a,b in solution.items():
        if a[0] == 'b':
            print(a,b)

