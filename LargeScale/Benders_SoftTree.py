from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING
from TreeStructure import Parent
from binarytree import build
from TreeStructure import OptimalTree
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from DatabaseParser import DataParser


def optimal_CT(df, F, labels, depth, C):

    I = df.index.values

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

    master = Model("softTree")

    a = {(f, t): master.addVar(vtype='C', lb=float('-inf'), name=f'a[{f, t}]') for f in F for t in T_B}
    b = {t: master.addVar(vtype='C', lb=float('-inf'), name=f'b[{t}]') for t in T_B}
    z = {(i, t): master.addVar(vtype='C',lb=0,ub=1, name=f'z[{i, t}]') for i in I for t in T_L}

    master.data = a, b, z

    for i in I:
        master.addCons( quicksum(z[i, t] for t in T_L) == 1 )

        for l in T_L:
            for t in A_l[l]:
                master.addCons(
                    z[i,l] <= quicksum(a[j, t] * (df.loc[i, j] + mu[j] - mu_min) for j in F) + mu_min - b[t]
                )
            for t in A_r[l]:
                master.addCons(
                    z[i,l] >= quicksum(a[j, t] * df.loc[i, j] for j in F) - b[t]
                )

    subprob2 = Model("SVM")

    Beta = {(f, t): subprob2.addVar(vtype='C', lb=float('-inf'), name=f'Beta[{f, t}]') for f in F for t in T_L}
    Beta1 = {(f, t): subprob2.addVar(vtype='C',lb=0, name=f'Beta1[{f, t}]') for f in F for t in T_L}
    Beta2 = {(f, t): subprob2.addVar(vtype='C',lb=0, name=f'Beta2[{f, t}]') for f in F for t in T_L}
    Delta = {t: subprob2.addVar(vtype='C', lb=float('-inf'), name=f'Delta[{t}]') for t in T_L}
    e = {(i, t): subprob2.addVar(vtype='C', lb=0, name=f'e[{i, t}]') for i in I for t in T_L}
    z = {(i, t): subprob2.addVar(vtype='C',lb=0,ub=1, name=f'z[{i, t}]') for i in I for t in T_L}

    d = {(i,t): subprob2.addVar(vtype='C',lb=0, name=f'd[{i, t}]') for i in I for t in T_L}

    for t in T_L:
        for i in I:
            subprob2.addCons(
                (quicksum(Beta[j, t] * df.loc[i, j] for j in features) + Delta[t]) * df.loc[i, labels[0]] >= 1 - e[i, t]
            )

            subprob2.addCons(
                d[i,t] ==  e[i,t] * z[i,t]
            )

        for f in F:
            subprob2.addCons(Beta[f, t] == Beta1[f, t] - Beta2[f, t])

    subprob2.setObjective(
        quicksum(Beta1[f, t] + Beta2[f, t] for f in F for t in T_L) + C * quicksum( d[i,t] for i in I for t in T_L),
        'minimize'
    )

    subprob2.data = Beta, Beta1, Beta2, Delta, e, z, d

    subprobs = {'sub2': subprob2}

    # master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam('misc/allowstrongdualreds', False)
    master.setBoolParam('benders/copybenders', False)
    master.initBendersDefault(subprobs)

    master.optimize()

    # solving the subproblems to get the best solution
    master.computeBestSolSubproblems()

    a, b, z = master.data

    splitting_nodes = {
        t: {
            'a': [round(master.getVal(a[f, t]), 2) for f in F],
            'b': round(master.getVal(b[t]), 2)
        }
        for t in T_B
    }

    Beta, Beta1, Beta2, Delta, e, z, d = subprob2.data

    non_empty_nodes = {
        t: {
            'Beta': [round(subprob2.getVal(Beta[f, t]), 2) for f in F],
            'Delta': round(subprob2.getVal(Delta[t]), 2)
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

    df = DataParser('blogger-simplified.arff', one_hot=True)

    df = shuffle(df, random_state=7)
    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df  # .iloc[len(Test_df):]

    # Train_df = Train_df.sample(n=round(len(df)*0.03)) # todo if you want to reduce the trainset
    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df.drop(columns=[i], inplace=True)
            Test_df.drop(columns=[i], inplace=True)

    features = list(Train_df.columns.drop([label_name]))
    labels = df[label_name].unique()
    labels = (label_name, labels)

    depth = 1

    splitting_nodes, non_empty_nodes = optimal_CT(
        df=Train_df,
        F=features,
        labels=labels,
        depth=depth,
        C=1
    )

    print('Splitting Nodes')
    for i in splitting_nodes.items():
        print(i[0], i[1])
    print('Non-Empty Nodes')
    for i in non_empty_nodes.items():
        print(i[0], i[1])


