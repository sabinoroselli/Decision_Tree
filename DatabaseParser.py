import arff as rf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def DataParser(name, ProbType, one_hot = True):

    try:
        pd.set_option('future.no_silent_downcasting', True)
    except:
        pass

    label_name = 'class'

    data = rf.load(open(f'{ProbType}Problems/{name}','rt'))
    df = pd.DataFrame(data['data'])
    df.columns = [i[0] for i in data['attributes'] ]

    for i in data['attributes']:
        # first with the features
        if i[0] != label_name:
            if type(i[1]) == str:# in ['REAL','INTEGER']:
                # replace NaN with mean for RegressionProblems value features
                mean_value = df[i[0]].mean()
                df[i[0]] = df[i[0]].fillna(value=mean_value).astype(float)
            else:
                if one_hot == False:
                    temp_feature_values = {i: ind for ind, i in enumerate(i[1])}
                    df[i[0]] = df[i[0]].replace(temp_feature_values).astype(float)
                else:
                    if len(i[1]) == 2:
                        df = pd.get_dummies(df,columns=[i[0]],drop_first=True,dtype=int)
                    else:
                        df = pd.get_dummies(df, columns=[i[0]],dtype=int)
        elif i[0] == label_name and ProbType == 'Classification': # todo FOR CLASSIFICATION SCALE TARGET
            if len(i[1]) == 2:
                temp_label_values = {i[1][0]:-1,
                                     i[1][1]:1}
                df[i[0]] = df[i[0]].replace(temp_label_values).astype(float)
            else:
                temp_label_values = {i: ind for ind, i in enumerate(i[1])}
                df[i[0]] = df[i[0]].replace(temp_label_values).astype(float)


    # ELIMIATING A COLUMN IF ALL THE VALUES IN IT ARE THE SAME
    for i in df.columns:
        if df[i].nunique() == 1:
            df = df.drop(i, axis=1)


    # if abs(df.values.max()) < 0.001: # todo REMOVE WHEN SURE THIS IS NOT NEEDED
    #     print('####### SCALING DF UP BY A 100 FACTOR ##########')
    #     # If you want to scale up by 100
    #     temp_df = df.drop(columns=label_name) * 100
    #     temp_label = df[label_name]
    #     temp_df.insert(0, label_name, temp_label, True)
    #     df = temp_df
    ##### STANDARD SCALING #######
    std_scaler = StandardScaler()
    features = list(df.columns.drop([label_name]))
    df_scaled = df.drop(columns=label_name)
    df_scaled = std_scaler.fit_transform(df_scaled.to_numpy())
    df_scaled = pd.DataFrame(df_scaled,columns=features)
    df_scaled.insert(len(features), label_name, df[label_name], True)
    df = df_scaled

    return df

if __name__ == "__main__":

    #################### REGRESSION #####################
    # collection = os.listdir('RegressionProblems')
    #
    # for i in collection:
    #     df = regression_data_caller(i)
    #     print(df)
    #########################################

    #################### CLASSIFICATION #####################
    collection = [
        # 'kr-vs-kp',
    ]
    for i in collection:
        df = DataParser(i)
        print(i)
        print(df)#.to_markdown())
    #########################################
