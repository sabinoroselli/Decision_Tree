from DatabaseParser import DataParser
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import arff

def compute_optimal_fraction(file,RS):

    label_name = 'class'

    df = DataParser(file, 'Classification')

    df = shuffle(df, random_state=RS)

    TestSize = 0.2
    Test_df = df.iloc[:round(len(df) * TestSize)]
    Train_df = df.iloc[len(Test_df):]

    print(f'Database initial length {len(Train_df)}')

    # split train set into features and labels
    X_train = Train_df.drop(columns=label_name)
    X_train = X_train.to_numpy()
    Y_train = Train_df[label_name]

    # split test set into features and labels
    X_test = Test_df.drop(columns=label_name)
    X_test = X_test.to_numpy()
    Y_test = Test_df[label_name]

    clf = RandomForestClassifier(n_estimators=100, random_state=RS)
    clf.fit(X_train, Y_train)
    test_pred = clf.predict(X_test)

    TresholdAcc = round(accuracy_score(Y_test, test_pred) * 100, 2) * 0.99
    CurrentAcc = round(accuracy_score(Y_test, test_pred) * 100, 2)

    print(f'Initial Accuracy: {CurrentAcc}, Threshold: {TresholdAcc}')
    counter = 1
    while CurrentAcc >= TresholdAcc:

        Train_df = Train_df.reset_index()
        Train_df = Train_df.drop(columns='index')
        Train_df = Train_df.drop(index=random.sample([i for i in range(len(Train_df)-1)],10))

        X_train = Train_df.drop(columns=label_name)
        X_train = X_train.to_numpy()
        Y_train = Train_df[label_name]

        clf.fit(X_train, Y_train)
        test_pred = clf.predict(X_test)

        CurrentAcc = round(accuracy_score(Y_test, test_pred) * 100, 2)

        print(f'Iteration: {counter}, Database Length: {len(Train_df)}, Current Accuracy: {CurrentAcc}')

        counter += 1

    return Train_df

def generate_new_df(file,df):
    attributes = [
        (j, 'NUMERIC')
        if df[j].dtypes in ['int64', 'float64']
        else (j, df[j].unique().astype(str).tolist())
        for j in df.drop(columns='class')]
    attributes.append(('class',['-1.0','1.0']))

    with open(f'ClassificationProblems/DownSampled/{file}','w',encoding='utf8') as f:
        arff.dump(
            {
                'attributes':attributes,
                'data': df.values,
                'relation': file.split('.')[0],
                'description': ''
            },
            f
        )

if __name__ == '__main__':

    collection = [
        'delta_ailerons.arff',
        # 'mushroom.arff',
        # 'mc1.arff',
        # 'mammography.arff',
        # 'california.arff'
    ]

    new_df = compute_optimal_fraction(collection[0],7)
    generate_new_df(collection[0],new_df)


