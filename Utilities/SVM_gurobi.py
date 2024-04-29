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

    #### JUST SOME TESTING, PLUGGING IN THE RESULTS FROM SVM
    # m.addConstr(b == -0.0359868)
    # coeff = [0.02696192,  0.37117922,  0.65394476,  0.11534842,  0.03011364, -0.16277089,
    #  0.07986732, -0.27208812,  0.32114592,  0.23310075,  0.09940078,  0.75818051,
    #  0.62924266]
    #
    # m.addConstrs(
    #     w[i] == coeff[ind]
    #     for ind,i in enumerate(features[:-3])
    # )

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
        'heart-statlog',
        # 'sonar',
        # 'hepatitis',
        # 'breast-cancer',
        # 'ionosphere',
        # 'colic',
        # 'vote',
        # 'breast-w',
        # 'blood_transfusion',
        # 'credit-g',
        # 'kr-vs-kp',
        # 'monks_problem_2',
        # 'steel_plates_fault',
        # 'tic-tac-toe'
    ]
    for RS in range(7,8):
        for i in collection:
            print(f'##################### {i}-{RS} ################')
            best_acc = float('-inf')
            best_c = float('-inf')
            label_name = 'class'
            df = DataParser(i)
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

            # print(f'best: {best_c} ({best_acc})')

    # if False:
    #     ######## CREDIT CARD FRAUD DATASET ######################
    #     df = pd.read_csv('Credit_card_fraud/creditcard_2023.csv')
    #     df.drop(columns='id', inplace=True)
    #     label_name = 'Class'
    #     df[label_name].replace(0, -1, inplace=True)
    # ####### SYNTHETIC FRAUD DATASET ######################
    # # df = pd.read_csv('test_instances/class_prova')
    # # label_name = 'Class'
    # if False:
    #     ####### ADULT INCOME FRAUD DATASET ######################
    #     df = pd.read_csv('adult/adult.data')
    #     countries_string = ' United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'
    #     countries = countries_string.split(',')
    #     countries_dict = {i: index for index, i in enumerate(countries)}
    #
    #     sex = {' Male': 0, ' Female': 1}
    #
    #     race_string = ' White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'
    #     race = race_string.split(',')
    #     race_dict = {i: index for index, i in enumerate(race)}
    #
    #     relationship_string = ' Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'
    #     relationship = relationship_string.split(',')
    #     relationship_dict = {i: index for index, i in enumerate(relationship)}
    #
    #     occupation_string = ' Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'
    #     occupation = occupation_string.split(',')
    #     occupation_dict = {i: index for index, i in enumerate(occupation)}
    #
    #     maritalStatus_string = ' Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'
    #     maritalStatus = maritalStatus_string.split(',')
    #     maritalStatus_dict = {i: index for index, i in enumerate(maritalStatus)}
    #
    #     education_string = ' Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'
    #     education = education_string.split(',')
    #     education_dict = {i: index for index, i in enumerate(education)}
    #
    #     workclass_string = ' Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'
    #     workclass = workclass_string.split(',')
    #     workclass_dict = {i: index for index, i in enumerate(workclass)}
    #
    #     income = {' <=50K': -1, ' >50K': 1}
    #
    #     df.replace({'NativeCountry': countries_dict}, inplace=True)
    #     df.replace({'Sex': sex}, inplace=True)
    #     df.replace({'Race': race_dict}, inplace=True)
    #     df.replace({'Relationship': relationship_dict}, inplace=True)
    #     df.replace({'Occupation': occupation_dict}, inplace=True)
    #     df.replace({'MaritalStatus': maritalStatus_dict}, inplace=True)
    #     df.replace({'Education': education_dict}, inplace=True)
    #     df.replace({'Workclass': workclass_dict}, inplace=True)
    #     df.replace({'Income': income}, inplace=True)
    #
    #     index_df = df[(df['Workclass'] == ' ?') | (df['Occupation'] == ' ?') | (df['NativeCountry'] == ' ?')].index
    #     df.drop(index_df, inplace=True)
    #     label_name = 'Income'