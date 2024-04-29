import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_percentage_error,accuracy_score
from sklearn.svm import SVC,LinearSVC
from DatabaseParser import DataParser

df = DataParser('heart-statlog')
label_name = 'class'
features = list(df.columns.drop([label_name]))

RS = 7

df = shuffle(df,random_state=RS)
Test_df = df.iloc[:round(len(df) * 0.2)]
Train_df = df.iloc[len(Test_df):]

X = Train_df.drop([label_name],axis=1)
Y = Train_df[label_name]

clf = SVC(
    kernel='linear',
    shrinking=False,
    random_state=RS,
    C=1
    # penalty='l1',
    # dual=True,
    # loss='hinge'
)

clf.fit(X, Y)

print(clf.intercept_, clf.coef_)

y_pred = clf.predict(Test_df.drop([label_name],axis=1))
print(f'Acc: {round(accuracy_score(Test_df[label_name],y_pred)*100,2)}%')


######## ADULT INCOME FRAUD DATASET ######################
# df = pd.read_csv('adult/adult.data')
# countries_string = ' United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'
# countries = countries_string.split(',')
# countries_dict = {i: index for index, i in enumerate(countries)}
#
# sex = {' Male': 0, ' Female': 1}
#
# race_string = ' White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'
# race = race_string.split(',')
# race_dict = {i: index for index, i in enumerate(race)}
#
# relationship_string = ' Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'
# relationship = relationship_string.split(',')
# relationship_dict = {i: index for index, i in enumerate(relationship)}
#
# occupation_string = ' Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'
# occupation = occupation_string.split(',')
# occupation_dict = {i: index for index, i in enumerate(occupation)}
#
# maritalStatus_string = ' Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'
# maritalStatus = maritalStatus_string.split(',')
# maritalStatus_dict = {i: index for index, i in enumerate(maritalStatus)}
#
# education_string = ' Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'
# education = education_string.split(',')
# education_dict = {i: index for index, i in enumerate(education)}
#
# workclass_string = ' Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'
# workclass = workclass_string.split(',')
# workclass_dict = {i: index for index, i in enumerate(workclass)}
#
# income = {' <=50K': -1, ' >50K': 1}
#
# df.replace({'NativeCountry': countries_dict}, inplace=True)
# df.replace({'Sex': sex}, inplace=True)
# df.replace({'Race': race_dict}, inplace=True)
# df.replace({'Relationship': relationship_dict}, inplace=True)
# df.replace({'Occupation': occupation_dict}, inplace=True)
# df.replace({'MaritalStatus': maritalStatus_dict}, inplace=True)
# df.replace({'Education': education_dict}, inplace=True)
# df.replace({'Workclass': workclass_dict}, inplace=True)
# df.replace({'Income': income}, inplace=True)
#
# index_df = df[(df['Workclass'] == ' ?') | (df['Occupation'] == ' ?') | (df['NativeCountry'] == ' ?')].index
# df.drop(index_df, inplace=True)
# df.head(50)
#
# label_name = 'Income'
# features = list(df.columns.drop([label_name]))

######## CREDIT CARD FRAUD DATASET ######################
# df = pd.read_csv('Credit_card_fraud/creditcard_2023.csv')
# df.drop(columns='id', inplace=True)
# label_name = 'Class'
# df['Class'].replace(0,-1,inplace=True)

######## SYNTHETIC FRAUD DATASET ######################
# df = pd.read_csv('test_instances/class_prova')
# label_name = 'Class'

####### SCALE DOWN DATA ###########
# scaler = MinMaxScaler()
# df_scaled = df.drop(columns=label_name)
# df_scaled = scaler.fit_transform(df_scaled.to_numpy())
# df_scaled = pd.DataFrame(df_scaled, columns=features)
# df_scaled.insert(len(features), label_name, df[label_name], True)
# df = df_scaled
####### SCALE UP DATA ###########
# data = df.drop(label_name, axis=1)
# data = data*100
# data.insert(2,label_name,df[label_name],True)
# print(data)
#

