from pystreed import STreeDPiecewiseLinearRegressor as Function
from sklearn.utils import shuffle
from DatabaseParser import DataParser
from Utilities.OldTrees.TreeStructure import RRSE,RAE

RegrDataBases = [
    # 'wisconsin.arff', ###
    # 'pwLinear.arff',
    # # 'cpu.arff', ###
    # 'yacht_hydrodynamics.arff',
    # 'RAM_price.arff',
    # 'autoMpg.arff',
    # 'vineyard.arff',
    # # 'boston_corrected.arff', ###
    # 'forest_fires.arff',
    # 'meta.arff',
    # 'arsenic-female-lung.arff',
    # 'arsenic-male-lung.arff',
    # 'titanic_1.arff',
    # 'stock.arff',
    # 'Bank-Note.arff',
    # 'balloon.arff',
    # 'debutanizer.arff',
    # 'analcatdata_supreme.arff',
    # 'Long.arff',
    'KDD.arff' ###
]

# warnings.filterwarnings('error')

for file in RegrDataBases:

    df = DataParser(f'{file}','Regression',one_hot=True)
    df = shuffle(df,random_state=7)

    # print(df.head(10))

    # df.rename(columns={x:y for x,y in zip(df.columns,range(0,len(df.columns)))},inplace=True)
    Class = 'class'#len(df.columns)-1
    # df[Class] = df[Class].replace(-1,0)

    Test_df = df.iloc[:round(len(df) * 0.2)]
    Train_df = df.iloc[len(Test_df):]

    # split train into features and labels
    X_train = Train_df.drop(columns=Class)
    # X_train = X_train.to_dict('index')
    Y_train = Train_df[Class]

    # Fit the model
    model = Function(
        max_depth=2, # todo if we are using it...we need to decide on this parameter
        time_limit=3600,

    )
    try:
        model.fit(X_train, Y_train)
        # model.print_tree()

        # split test set into features and labels
        X_test = Test_df.drop(columns=Class)
        # X_test = X_test.to_dict('index')
        Y_test = Test_df[Class]

        yhat = model.predict(X_test)

        print(f'{file.split(".")[0]}   RAE: {RAE(Y_test, yhat)}, RRSE: {RRSE(Y_test,yhat)} NrLeaves: {model.get_n_leaves()}')
    except:
        print(f'{file.split(".")[0]} TIMEOUT')
