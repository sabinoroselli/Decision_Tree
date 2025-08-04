import arff as rf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from SupportFunctions import Multiplier
import os

def DataParser(name, one_hot = True,toInt = False):

    try:
        pd.set_option('future.no_silent_downcasting', True)
    except:
        pass

    with open(f'{name}','rt') as my_file:
        data = rf.load(my_file)

    df = pd.DataFrame(data['data'])
    df.columns = [i[0] for i in data['attributes'] ]

    labels = [i[0] for i in data['attributes'] if i[1] == ['FALSE','TRUE']]
    features = list(df.columns.drop([i for i in labels]))

    eliminated_cols = []
    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in df.columns:
        if df[i].nunique() <= 1:
            df = df.drop(columns=[i])
            eliminated_cols.append(i)
    features = [i for i in features if i not in eliminated_cols]

    for i in [ i for i in data['attributes'] if i[0] not in eliminated_cols]:
        # first with the features
        if i[0] in features:
            if type(i[1]) == str:
                # replace NaN with mean for Numeric value features
                mean_value = df[i[0]].mean()
                df[i[0]] = df[i[0]].fillna(value=mean_value).astype(float)
                if toInt == True:
                    # df[i[0]] = df[i[0]].round(2)  # todo comment off if you do not wan to to round
                    df.loc[:, i[0]] *= Multiplier(df[i[0]])
            else:
                if one_hot == False:
                    temp_feature_values = {i: ind for ind, i in enumerate(i[1])}
                    df[i[0]] = df[i[0]].replace(temp_feature_values).astype(float)
                else:
                    if len(i[1]) == 2: # todo for binary nominal attributes I should replace with 0/1
                        df = pd.get_dummies(df,columns=[i[0]],drop_first=True,dtype=int)
                    else:
                        df = pd.get_dummies(df, columns=[i[0]],dtype=int)
        else:
            temp_label_values = {i[1][0]:-1,
                                 i[1][1]:1}
            df[i[0]] = df[i[0]].replace(temp_label_values).astype(float)

    features = list(df.columns.drop([i for i in labels])) # update features after one-hot transformation
    df.dropna(inplace=True)

    if toInt == False:
        ##### STANDARD SCALING #######
        std_scaler = StandardScaler()

        df_scaled = df.drop(columns=labels)

        df_scaled = std_scaler.fit_transform(df_scaled.to_numpy())
        df_scaled = pd.DataFrame(df_scaled,columns=features)

        for index,i in enumerate(labels):
            df_scaled.insert(len(features) + index, i, df[i], True)
        df = df_scaled

    id_name = ['row_id']
    if any(x in df.columns for x in id_name):
        df = df.drop(columns=[id_name])

    return df,features,labels

if __name__ == "__main__":

    collection = [
        # 'yeast.arff',
        # 'emotions.arff',
        'genbase.arff',
        # 'reuters.arff'
    ]

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    for i in collection:
        df,features,labels = DataParser(i,one_hot=True,toInt=False)
        print(df.head(5))#.to_markdown())
    #########################################
