import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_percentage_error
from sklearn.svm import LinearSVR,SVR

# BOSTON HOUSING
df = pd.read_csv('real_estate/housing.csv', sep="\s+")
label_name = 'MEDV'
# SINTHETIC PIECE-WISE LINEAR
# df = pd.read_csv('test_instances/prova',sep=',')
# label_name = 'Y'

X = df.drop([label_name],axis=1)
Y = df[label_name]

df = shuffle(df,random_state = 7)

Test_df = df.iloc[:round(len(df) * 0.3)]
Train_df = df.iloc[len(Test_df):]

X_train = Train_df.drop([label_name],axis=1)
y_train = Train_df[label_name]

# clf = LinearRegression()
clf = SVR(
    kernel='linear',
    C=10,
    epsilon=0,
    shrinking=True
    # dual=True,
    # random_state=7,
    # max_iter= 1000
)
clf.fit(X_train, y_train)

print(clf.intercept_, clf.coef_)

X_test = Test_df.drop([label_name],axis=1)
y_test = Test_df[label_name]

y_pred = clf.predict(X_test)
print(f'R2 score: {round(r2_score(y_test,y_pred)*100,2)}%')
print(f'MAPE: {round(mean_absolute_percentage_error(y_test,y_pred)*100,2)}%')
