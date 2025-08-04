from ortools.sat.python import cp_model
from binarytree import build
from sklearn.utils import shuffle
from DatabaseParser import DataParser
from TreeStructure import Parent, Children, OptimalTree
from time import process_time as tm
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

    df_max = max([abs(df[j].max()) for j in J])

    model = cp_model.CpModel()

    # variables

    g = { (k,t):model.NewBoolVar(f'g_{k}_{t}') for k in K for t in T } # if node t is assigned class k
    a = { (j,t):model.NewBoolVar(f'a_{j}_{t}') for j in J for t in T_b }  # if node t splits on feature j
    b = { t:model.NewIntVar(-int(max([abs(i) for i in df.drop(['class'],axis=1).max().values])),
                            int(max([abs(i) for i in df.drop(['class'],axis=1).max().values])),
                            f'b_{t}') for t in T_b} # value of the split at node t
    u = {(i,t1,t2):model.NewBoolVar(f'u_{i}_{t1}_{t2}') for i in I for t1 in T_g for t2 in T_g} # point i flows from node t to t'

    for t in T_b:

        model.Add( sum([ a[j,t] for j in J ]) + sum([ g[k,t] for k in K ]) == 1 )

        # model.Add( sum([a[j,t] for j in J ]) * int(df_max) >= b[t] )

        # model.Add(
        #     2 - 2 * sum([g[k, t] for k in K])
        #     >=
        #     sum([g[k, l[t]] for k in K])
        #     +
        #     sum([g[k, r[t]] for k in K])
        # )

        for i in I:
            model.Add( sum([a[j, t] * int(df.loc[i, j]) for j in J]) < b[t] ).OnlyEnforceIf(u[i, t, l[t]])

            model.Add( sum([a[j, t] * int(df.loc[i, j])  for j in J]) >= b[t] ).OnlyEnforceIf(u[i, t, r[t]])

            model.Add( u[i,t,l[t]] <= sum([ a[j,t] for j in J]) )

            model.Add( u[i, t, r[t]] <= sum([a[j, t] for j in J]) )

            model.Add( u[i,A[t],t] == u[i,t,l[t]] + u[i,t,r[t]] + u[i,t,'w'] )

    for t in T_l:
        for i in I:
            model.Add( u[i, A[t], t] == u[i, t, 'w'] )

    for t in T:
        model.Add(sum([g[k, t] for k in K]) <= 1)

        for i in I:
            model.Add( u[i, t, 'w'] <= g[df.loc[i, label_name], t] )

    model.Add( sum([ a[j,t] for j in J for t in T_b]) <= splits )

    model.Maximize( sum([ u[i,t,'w'] for t in T for i in I]) )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60 * 60
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 20
    StartTime = tm()
    status = solver.Solve(model)
    print(f'End of computation, runtime: {round(tm() - StartTime, 2)}')

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Maximum of objective function: {solver.ObjectiveValue()},"
              f" Accuracy:{round(solver.ObjectiveValue()/len(df),2)*100}%\n")

    #     splitting_nodes = {
    #         t: {
    #             'a': [f for f in features if solver.Value(a[f, t]) == 1][0],
    #             'b': solver.Value(b[t])
    #         }
    #         for t in T_b if solver.Value(d[t]) == 1
    #     }
    #
    #     non_empty_nodes = {
    #         t: [k for k in labels if solver.Value(c[k, t]) == True][0]
    #         for t in T_L if solver.Value(l[t]) == 1
    #     }
    #
    #     ODT = OptimalTree(
    #         non_empty_nodes,
    #         splitting_nodes,
    #         int(np.ceil(np.log2(Splits + 1))),
    #         'Parallel',
    #         False
    #     )
    # else:
    #     print('MODEL IS INFEASIBLE')
    #     ODT = None
    #
    # return ODT, runtime



if __name__ == "__main__":

    df = DataParser('autoUnivMulti.arff','Classification',one_hot=True,toInt=True)
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

    optimal_CT(
        df=df,
        features=features,
        labels=labels,
        splits=Splits
    )


