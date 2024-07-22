from gurobipy import *
from sklearn.utils import shuffle
from Utilities.OldTrees.TreeStructure import RRSE,RAE
from DatabaseParser import DataParser


def LinReg_Gurobi(df,features,label_name):

    I = [i for i in df.index]

    labels = df[label_name]

    m = Model('LinReg')
    m.setParam('NonConvex',2)
    # m.setParam('OutputFlag', 0)
    # m.setParam('TimeLimit', 2 * 60)

    w = m.addVars(features,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='w')
    w_abs = m.addVars(features, vtype=GRB.CONTINUOUS, name='w_abs')
    b = m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name='b')
    gamma = m.addVars(I, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='gamma')  # function f(x) - y_i at node t
    gamma_abs = m.addVars(I, vtype=GRB.CONTINUOUS, name='gamma_abs')  # abs of gamma

    m.addConstrs(
        gamma[i] == quicksum([ w[f] * df.loc[i,f]  for f in features ]) + b - labels[i]
        for i in I
    )

    m.addConstrs(
        gamma_abs[i] == abs_(gamma[i]) for i in I
    )

    m.addConstrs(
        w[f] == (w_abs[f])**2 for f in features
    )

    m.setObjective( quicksum([ w[f] for f in features]) + quicksum([ gamma_abs[i] for i in I]) )

    m.optimize()
    solution = {}
    vars = m.getVars()
    for i in vars:
        if i.VarName[0] != 'g':
            solution.update({i.VarName:i.X})
    return solution

def predict(Test_df, label_name, solution):

    # split validation set into features and labels
    X_test = Test_df.drop(columns=label_name)
    Y_test = Test_df[label_name]

    X = X_test.to_dict('index')
    Y = Y_test.to_dict()

    predictions = {key:sum([ val[f] * solution[f'w[{f}]'] for f in val ]) + solution['b']
                   for key,val in X.items()}

    # for i in list(predictions):
    #     print(predictions[i],Y[i])

    RelAbsErr = RAE([Y[i] for i in Y],list(predictions.values()))
    RelRootSqErr = RRSE([Y[i] for i in Y],list(predictions.values()))
    return round(RelAbsErr,2),round(RelRootSqErr,2)

if __name__ == "__main__":

    label_name = 'class'
    file = 'vineyard'
    Splits = 0

    config = {
        'RandomSeed': 0,
        'ProbType': 'Regression',
        "ModelTree": True,
        'SplitType': 'Parallel',
        'label_name': 'class',
        'TestSize': 0.2,
        'ValSize': 0.2,
        'df_name': file,
        'Timeout': 60,  # for the single iteration (IN MINUTES)
        'Fraction': 1  # fraction
    }

    df = DataParser(f'{file}.arff', 'Regression', one_hot=True)

    df = shuffle(df, random_state=config['RandomSeed'])

    Test_df = df.iloc[:round(len(df) * config['TestSize'])]
    Train_df = df.iloc[len(Test_df):]

    features = list(Train_df.columns.drop([label_name]))


    solution = LinReg_Gurobi(Train_df,features,label_name)

    for i in solution.items():
        print(i)

    score,mae = predict(Test_df,label_name,solution)

    print(f'RelAbsErr: {score}')
    print(f'RelRootSqErr: {mae}')






