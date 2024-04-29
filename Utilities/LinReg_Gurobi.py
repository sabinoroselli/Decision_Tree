from gurobipy import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_percentage_error


def LinReg_Gurobi(df,features,label_name):

    I = [i for i in df.index]

    labels = df[label_name]

    m = Model('LinReg')
    m.setParam('NonConvex',2)
    # m.setParam('OutputFlag', 0)
    # m.setParam('TimeLimit', 2 * 60)

    w = m.addVars(features,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='w')
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

    m.setObjective( quicksum([ gamma_abs[i] for i in I]) )

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

    score = r2_score([Y[i] for i in Y],list(predictions.values()))*100
    mae = mean_absolute_percentage_error([Y[i] for i in Y],list(predictions.values()))*100
    return f'{round(score,2)}%',f'{round(mae,2)}%'

if __name__ == "__main__":

    # REAL ESTATE DATASET
    # df = pd.read_csv('real_estate/Real estate.csv', delimiter=',')
    # df.drop('No', axis=1, inplace=True)
    # label_name = 'house_price_of_unit_area'
    # BOSTON HOUSING
    df = pd.read_csv('real_estate/housing.csv',sep='\s+')
    label_name = 'MEDV'
    # SINTHETIC PIECE-WISE LINEAR
    # df = pd.read_csv('test_instances/prova', sep=',')
    # label_name = 'Y'

    df = shuffle(df,random_state = 7)

    features = list(df.columns.drop([label_name]))

    # scaler = MinMaxScaler()
    # df_scaled = scaler.fit_transform(df.to_numpy())
    # df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    # df = df_scaled

    Test_df = df.iloc[:round(len(df) * 0.3)]
    Train_df = df.iloc[len(Test_df):]

    print(Train_df)

    solution = LinReg_Gurobi(Train_df,features,label_name)

    for i in solution.items():
        print(i)

    score,mae = predict(Test_df,label_name,solution)

    print(f'R2 score: {score}%')
    print(f'MAE: {mae}')






