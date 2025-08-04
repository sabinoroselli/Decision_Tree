from gurobipy import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from DatabaseParser import DataParser


def MILP_SVM(df,features,label_name,C):

    I = [i for i in df.index]

    labels = df[label_name]

    m = Model('SVM')
    m.setParam('OutputFlag', 0)


    w = m.addVars(features,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='w')
    w_abs = m.addVars(features, vtype=GRB.CONTINUOUS, name='W_abs')
    b = m.addVar(lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    e = m.addVars(I,lb=0,vtype=GRB.CONTINUOUS,name='e')

    m.addConstrs(
        labels[i] * (quicksum([ w[f] * df.loc[i,f]  for f in features ]) + b) >= 1 - e[i]
        for i in I
    )

    m.addConstrs(
        w_abs[f] == (w[f])**2 for f in features
    )

    m.setObjective( C * quicksum([ e[i] for i in I ]) + quicksum([ w_abs[f] for f in features  ]) )

    m.optimize()
    solution = {}
    if m.status != GRB.INFEASIBLE:
        vars = m.getVars()
        # print('VARIABLES...')
        for i in vars:
            if i.VarName[0] in ['w','b']:
                solution.update({ i.VarName:i.X })
    return solution

if __name__ == "__main__":

    collection = [
        'california.arff'
    ]
    for RS in range(7,8):
        for i in collection:
            print(f'##################### {i}-{RS} ################')
            best_acc = float('-inf')
            best_c = float('-inf')
            label_name = 'class'
            df = DataParser(i,'Classification')
            features = list(df.columns.drop([label_name]))

            df = shuffle(df,random_state=RS)
            # print(df)
            Test_df = df.iloc[:round(len(df) * 0.2)]
            Train_df = df.iloc[len(Test_df):]

            for C in [1]:
                solution = MILP_SVM(Train_df,features,label_name,C)

                for i in solution.items():
                    print(i)

                Y_pred = [
                    sum([ solution[f'w[{f}]'] * Test_df.loc[i,f] for f in features]) + solution['b'] for i in Test_df.index
                ]

                Y_pred = [
                    1 if i > 0 else -1 for i in Y_pred
                ]
                y_test = Test_df[label_name]
                curr_acc = round(accuracy_score(y_test,Y_pred)*100,2)

                print(C, f'Acc: {curr_acc}%')
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    best_c = C

