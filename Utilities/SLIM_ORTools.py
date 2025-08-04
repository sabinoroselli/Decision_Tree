from ortools.sat.python import cp_model
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from DatabaseParser import DataParser
from time import process_time as tm

def MILP_SVM(df, features, label_name, C):
    I = [i for i in df.index]

    labels = df[label_name]

    bigM = 1000

    model = cp_model.CpModel()

    w = {f:model.NewIntVar(-bigM,bigM,name=f'w_{f}') for f in features}
    # w_abs = {f: model.NewIntVar(0, bigM, name=f'w_abs_{f}') for f in features}
    W = {f:model.NewBoolVar(name=f'W_{f}') for f in features}
    b = model.NewIntVar(-bigM,bigM,name='b')
    # e = {i: model.NewIntVar(0,bigM,name=f'e_{i}') for i in I}
    E = {i: model.NewBoolVar(name=f'E_{i}') for i in I}

    for i in I:
        model.Add(
            int(labels[i]) * (sum([w[f] * int(df.loc[i, f]) for f in features]) + b) >= 1
        ).OnlyEnforceIf(E[i].Not())
    for f in features:
        model.Add(
            w[f] != 0
        ).OnlyEnforceIf(W[f])
        model.Add(
            w[f] == 0
        ).OnlyEnforceIf(W[f].Not())
        # model.AddAbsEquality(
        #     w[f],w_abs[f]
        # )

    model.Minimize(
        C * sum([E[i] for i in I])
        + sum([W[f] for f in features])
        # + sum([ w_abs[f] for f in features])
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 360
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 1
    StartTime = tm()
    status = solver.Solve(model)
    print(f'End of computation, runtime: {round(tm() - StartTime, 2)}')
    solution = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Maximum of objective function: {solver.ObjectiveValue()}")

        solution.update({f'w_{f}':solver.Value(w[f]) for f in features})
        solution.update({'b':solver.Value(b)})
    else:
        print('PROBLEM INFEASIBLE')

    return solution


if __name__ == "__main__":

    collection = [
        'delta_ailerons.arff'
    ]
    for RS in range(7, 8):
        for i in collection:
            print(f'##################### {i}-{RS} ################')
            best_acc = float('-inf')
            best_c = float('-inf')
            label_name = 'class'
            df = DataParser(i,'Classification',one_hot=True,toInt=True)
            features = list(df.columns.drop([label_name]))

            df = shuffle(df, random_state=RS)
            # print(df)
            Test_df = df.iloc[:round(len(df) * 0.2)]
            Train_df = df.iloc[len(Test_df):]

            for C in [1]:
                solution = MILP_SVM(Train_df, features, label_name, C)

                for i in solution.items():
                    print(i)
                if solution != {}:

                    Y_pred = [
                        sum([solution[f'w_{f}'] * Test_df.loc[i, f] for f in features]) + solution['b'] for i in
                        Test_df.index
                    ]

                    Y_pred = [
                        1 if i > 0 else -1 for i in Y_pred
                    ]
                    y_test = Test_df[label_name]
                    curr_acc = round(accuracy_score(y_test, Y_pred) * 100, 2)

                    print('C:',C, f'Acc: {curr_acc}%')
                    if curr_acc > best_acc:
                        best_acc = curr_acc
                        best_c = C

