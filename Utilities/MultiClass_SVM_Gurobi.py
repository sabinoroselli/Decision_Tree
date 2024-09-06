from gurobipy import *
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC,LinearSVC
from DatabaseParser import DataParser


def MILP_MultiSVM(df,features,label_name,C,WW=False):

    # print(df)

    I = [i for i in df.index]

    classes = df[label_name].unique()
    # print(classes)

    LabelsPerClass = {
        c:{
            i: 1 if df.loc[i,label_name] == c else -1
            for i in I
        }
        for c in classes
    }

    # for i in the_labels.items():
    #     print(i)

    m = Model('SVM')
    m.setParam('OutputFlag', 0)


    w = m.addVars(classes,features,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='w')
    w_abs = m.addVars(classes,features, vtype=GRB.CONTINUOUS, name='W_abs')
    b = m.addVars(classes,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='b')
    e = m.addVars(classes,I,lb=0,vtype=GRB.CONTINUOUS,name='e')


    if WW:
        m.addConstrs(
            quicksum([w[df.loc[i,label_name], f] * df.loc[i, f] for f in features]) + b[df.loc[i,label_name]]
            >=
            quicksum([ w[c,f] * df.loc[i,f]  for f in features ]) + b[c] + 2 - e[c, i]
            for i in I
            for c in classes
        )
    else:
        m.addConstrs(
            LabelsPerClass[c][i] * (quicksum([ w[c,f] * df.loc[i,f]  for f in features ]) + b[c]) >= 1 - e[c,i]
            for i in I
            for c in classes
        )

    m.addConstrs(
        w_abs[c,f] == abs_(w[c,f]) for f in features for c in classes
    )

    sum_to_zero = m.addConstrs(
        quicksum([ quicksum([w[c,f] * df.loc[i,f] for f in features]) + b[c] for c in classes]) == 0
        for i in I
    )

    if WW:
        m.setObjective( C * quicksum([ e[c,i] for i in I for c in classes if c != df.loc[i,label_name] ])
                        +
                        quicksum([ w_abs[c,f] for f in features for c in classes  ]) )
    else:
        m.setObjective(C * quicksum([e[c, i] for i in I for c in classes]) + quicksum(
            [w_abs[c, f] for f in features for c in classes]))

    m.optimize()
    classifiers = {}
    if m.status != GRB.INFEASIBLE:
        vars = m.getVars()
        for i in vars:
            if i.VarName[0] in ['w','b']:
                classifiers.update({ i.VarName:i.X })
    return classifiers

if __name__ == "__main__":

    for C in [0.1,1,10,100,1000]:

        df_name = 'iris.arff'

        # data = rf.load(open(f'{df_name}'))
        # df = pd.DataFrame(data['data'])
        # df.columns = [i[0] for i in data['attributes']]
        df = DataParser(df_name,'Classification',True)

        label_name = df.columns[-1]
        features = df.drop(columns=label_name).columns
        classes = df[label_name].unique()

        df = shuffle(df, random_state=7)

        Test_df = df.iloc[:round(len(df) * 0.4)]
        Train_df = df.iloc[len(Test_df):]

        classifiers = MILP_MultiSVM(Train_df,features,label_name,C,True)

        print('Regularization Parameter:', C, end=" ")

        def predictor(classifiers,datapoint):

            scores = {
                i:sum([ classifiers[f'w[{i},{f}]'] * datapoint[f] for f in features]) + classifiers[f'b[{i}]']
                for i in classes
            }

            return max(scores, key=scores.get)

        def evaluate_set(classifiers,dataset):
            return [predictor(classifiers,datapoint) for datapoint in dataset.values()]




        # split train into features and labels
        X_train = Train_df.drop(columns=label_name)
        X_train = X_train.to_dict('index')
        Y_train = Train_df[label_name]

        # split test set into features and labels
        X_test = Test_df.drop(columns=label_name)
        X_test = X_test.to_dict('index')
        Y_test = Test_df[label_name]

        # print('     $$$$$$$$$$$$$ MILP - SVM $$$$$$$$$$$$$$$$$$$$')
        train_pred = evaluate_set(classifiers, X_train)
        print('         Train: ', round(accuracy_score(Y_train, train_pred) * 100, 2), '%',end= " ")
        test_pred = evaluate_set(classifiers, X_test)
        print('         Test: ', round(accuracy_score(Y_test, test_pred) * 100, 2), '%')

        # IN CASE I WANNA COMPARE TO AN ACTUAL (???) SVM
        # print('$$$$$$$$$$$$$ SCIKIT $$$$$$$$$$$$$$$$$$$$')
        # clf = LinearSVC(
        #     # kernel='linear',
        #     # shrinking=False,
        #     # random_state=7,
        #     C=C,
        #     penalty='l1',
        #     dual=False,
        #     # loss='hinge'
        # )
        # X = Train_df.drop([label_name], axis=1)
        # Y = Train_df[label_name]
        # clf.fit(X, Y)
        # y_pred = clf.predict(Test_df.drop([label_name], axis=1))
        # print(f'Test: {round(accuracy_score(Test_df[label_name], y_pred) * 100, 2)}%')




