from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING
from TreeStructure import Parent
from binarytree import build
from TreeStructure import OptimalTree
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from DatabaseParser import DataParser


def optimal_CT(df, F, labels, depth, Splits, C):

    I = df.index.values

    bigM = max([df[f].max() for f in F])

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in F
    }

    mu_min = min(mu.values())

    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (depth + 1) - 1)]
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

    master = Model("OCMT-master")

    d = {t: master.addVar(vtype='B', name=f'd[{t}]') for t in T_B}
    l = {t: master.addVar(vtype='B', name=f'l[{t}]') for t in T_L}
    a = {(f, t): master.addVar(vtype='B',name=f'a[{f, t}]') for f in F for t in T_B}


    master.data = d, l, a

    for t in [i for i in T_B if i != root.value]:
        master.addCons(d[t] <= d[P[t]])

    for t in T_B:

        master.addCons(quicksum(a[f, t] for f in F) == d[t])

        master.addCons(d[t] <= quicksum(l[m] for m in D_l[t]))

        master.addCons(d[t] <= quicksum(l[m] for m in D_r[t]))

    master.addCons(quicksum(d[t] for t in T_B) <= Splits)

    # master.setObjective( quicksum(d[t] for t in T_B))

    subprob1 = Model("OCMT-feasCut")

    b = {t: subprob1.addVar(vtype='C',lb=float("-inf"), name=f'b[{t}]') for t in T_B}
    Beta = {(f, t): subprob1.addVar(vtype='C',lb=float("-inf"), name=f'Beta[{f, t}]') for f in F for t in T_L}
    Beta1 = {(f, t): subprob1.addVar(vtype='C',lb=0, name=f'Beta1[{f, t}]') for f in F for t in T_L}
    Beta2 = {(f, t): subprob1.addVar(vtype='C',lb=0, name=f'Beta2[{f, t}]') for f in F for t in T_L}
    Delta = {t: subprob1.addVar(vtype='C',lb=float("-inf"), name=f'Delta[{t}]') for t in T_L}
    e = {(i, t): subprob1.addVar(vtype='C',lb=0, name=f'e[{i, t}]') for i in I for t in T_L}
    z = {(i, t): subprob1.addVar(vtype='B', name=f'z[{i, t}]') for i in I for t in T_L}
    d = {t: subprob1.addVar(vtype='B', name=f'd[{t}]') for t in T_B}
    l = {t: subprob1.addVar(vtype='B', name=f'l[{t}]') for t in T_L}
    a = {(f, t): subprob1.addVar(vtype='B', name=f'a[{f, t}]') for f in F for t in T_B}

    for t in T_L:
        for i in I:
            subprob1.addCons(z[i, t] <= l[t])

        subprob1.addCons(quicksum(z[i, t] for i in I) >= l[t])

    for i in I:
        subprob1.addCons(quicksum(z[i, t] for t in T_L) == 1)

    for i in I:
        for l in T_L:
            for t in A_l[l]:
                subprob1.addCons(
                    quicksum(a[j, t] * (df.loc[i, j] + mu[j] - mu_min) for j in F) + mu_min
                    <=
                    b[t] + bigM * (1 - z[i, l])
                )
            for t in A_r[l]:
                subprob1.addCons(
                    quicksum(a[j, t] * df.loc[i, j] for j in F) >= b[t] - bigM * (1 - z[i, l])
                )

    for t in T_B:

        subprob1.addCons(b[t] <= bigM * d[t])

        subprob1.addCons(b[t] >= -bigM * d[t])

    for t in T_L:
        for i in I:
            subprob1.addCons(
                1 - e[i, t] <= (quicksum(Beta[j, t] * df.loc[i, j] for j in features)
                                + Delta[t]) * df.loc[i, labels[0]] + bigM * (1 - z[i, t])
            )
        for f in F:
            subprob1.addCons(Beta[f, t] == Beta1[f, t] - Beta2[f, t])

    subprob1.setObjective(
        quicksum(Beta1[f, t] + Beta2[f, t] for f in F for t in T_L) + C * quicksum(e[i, t] for i in I for t in T_L),
        'minimize'
    )
    subprob1.data = b, Beta, Beta1, Beta2, Delta, e, z

    subprobs = {'sub1': subprob1}

    master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam('misc/allowstrongdualreds', False)
    master.setBoolParam('benders/copybenders', False)
    master.initBendersDefault(subprobs)

    master.optimize()

    splitting_nodes = {}
    non_empty_nodes = {}

    if master.getStatus() == 'optimal':

        # solving the subproblems to get the best solution
        master.computeBestSolSubproblems()

        b, Beta, Beta1, Beta2, Delta, e, z = subprob1.data

        d, l, a, = master.data

        splitting_nodes = {
            t: {
                'a': [round(subprob1.getVal(a[f, t]), 2) for f in F],
                'b': round(subprob1.getVal(b[t]), 2)
            }
            for t in T_B
        }

        non_empty_nodes = {
            t: {
                'Beta': [round(subprob1.getVal(Beta[f, t]), 2) for f in F],
                'Delta': round(subprob1.getVal(Delta[t]), 2)
            }
            for t in T_L
        }

        print('OPT: ', master.getObjVal())

        # since computeBestSolSubproblems() was called above, we need to free the
        # subproblems. This must happen after the solution is extracted, otherwise
        # the solution will be lost
        master.freeBendersSubproblems()

    return splitting_nodes, non_empty_nodes


if __name__ == "__main__":

    label_name = 'class'

    df = DataParser('blood-transf.arff', one_hot=True)

    df = shuffle(df, random_state=7)
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

    splitting_nodes, non_empty_nodes = optimal_CT(
        df=Train_df,
        F=features,
        labels=labels,
        depth=depth,
        Splits=Splits,
        C=1
    )

    print('Splitting Nodes')
    for i in splitting_nodes.items():
        print(i[0], i[1])
    print('Non-Empty Nodes')
    for i in non_empty_nodes.items():
        print(i[0], i[1])


