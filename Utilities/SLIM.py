from gurobipy import *
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from DatabaseParser import DataParser


def MILP_SVM(df, features, label_name, C):
    I = [i for i in df.index]

    labels = df[label_name]

    M = 10 # this limits the value of the weights
    epsilon = 0.001

    m = Model('SLIM')
    m.setParam('Threads',1)
    # m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 360)

    w = m.addVars(features, lb=-M,ub=M, vtype=GRB.INTEGER, name='w')
    W = m.addVars(features, lb=0, ub=1, vtype=GRB.INTEGER, name='W')
    Wabs = m.addVars(features, lb=0, ub=M, vtype=GRB.INTEGER, name='Wabs')
    b = m.addVar(lb=-M*10,ub=M*10, vtype=GRB.INTEGER, name='b')
    E = m.addVars(I, lb=0, ub=1, vtype=GRB.INTEGER, name='E')

    m.addConstrs(
        labels[i] * (quicksum([w[f] * df.loc[i, f] for f in features]) + b) >= 1 - len(df) * E[i]
        for i in I
    )

    m.addConstrs(
        w[f] <= M * W[f] for f in features
    )

    m.addConstrs(
        w[f] >= - M * W[f] for f in features
    )

    m.addConstrs(
        w[f] <= Wabs[f] for f in features
    )

    m.addConstrs(
        w[f] >= - Wabs[f] for f in features
    )

    m.setObjective(
        C * quicksum([E[i] for i in I])
        + quicksum([W[f] for f in features])
        + epsilon * quicksum([Wabs[f] for f in features])
    )

    m.optimize()
    solution = {}
    if m.status != GRB.INFEASIBLE:
        vars = m.getVars()
        # print('VARIABLES...')
        for i in vars:
            if i.VarName[0] in ['w', 'b']:
                solution.update({i.VarName: i.X})
    return solution


if __name__ == "__main__":

    collection = [
        'biomed.arff'
    ]
    for RS in range(7, 8):
        for i in collection:
            print(f'##################### {i}-{RS} ################')
            best_acc = float('-inf')
            best_c = float('-inf')
            label_name = 'class'
            df = DataParser(i,'Classification',one_hot=True,toInt=False)
            features = list(df.columns.drop([label_name]))

            df = shuffle(df, random_state=RS)
            # print(df)
            Test_df = df.iloc[:round(len(df) * 0.2)]
            Train_df = df.iloc[len(Test_df):]

            for C in [0.1,1,10,100]:
                solution = MILP_SVM(Train_df, features, label_name, C)

                for i in solution.items():
                    print(i)

                Y_pred = [
                    sum([solution[f'w[{f}]'] * Test_df.loc[i, f] for f in features]) + solution['b'] for i in
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

