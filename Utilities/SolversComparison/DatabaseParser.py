import arff as rf
from sklearn.preprocessing import StandardScaler
from TreeStructure import Multiplier
import pandas as pd
import os

def DataParser(name, ProbType, one_hot = True,toInt = False):

    try:
        pd.set_option('future.no_silent_downcasting', True)
    except:
        pass

    cwd = os.getcwd()
    cwd = '/'.join(cwd.split('/')[:-2])

    data = rf.load(open(f'{cwd}/{ProbType}Problems/{name}','rt'))

    label_name = data['attributes'][-1][0]

    df = pd.DataFrame(data['data'])
    df.columns = [i[0] for i in data['attributes'] ]

    eliminated_cols = []
    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in df.columns:
        if df[i].nunique() <= 1:
            df.drop(columns=[i], inplace=True)
            eliminated_cols.append(i)

    for i in [ i for i in data['attributes'] if i[0] not in eliminated_cols]:
        # first with the features
        if i[0] != label_name:
            if type(i[1]) == str:
                # replace NaN with mean for Numeric value features
                mean_value = df[i[0]].mean()
                df[i[0]] = df[i[0]].fillna(value=mean_value).astype(float)
                df[i[0]] = df[i[0]].round(2) # todo comment off if you do not wan to to round
                if toInt == True:
                    df.loc[:, i[0]] *= Multiplier(df[i[0]])
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
            # else:
            #     temp_label_values = {i: ind for ind, i in enumerate(i[1])}
            #     df[i[0]] = df[i[0]].replace(temp_label_values).astype(float)
    df.dropna(inplace=True)

    # ELIMIATING A COLUMN IF ALL THE VALUES IN IT ARE THE SAME
    for i in df.columns:
        if df[i].nunique() == 1:
            df = df.drop(i, axis=1)

    if toInt == False:
        ##### STANDARD SCALING #######
        std_scaler = StandardScaler()
        if ProbType == 'Classification':
            features = list(df.columns.drop([label_name]))
            df_scaled = df.drop(columns=label_name)
        else:
            features = list(df.columns)
            df_scaled = df
        df_scaled = std_scaler.fit_transform(df_scaled.to_numpy())
        df_scaled = pd.DataFrame(df_scaled,columns=features)
        if ProbType == 'Classification':
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
        'kr-vs-kp.arff',
    ]
    for i in collection:
        df = DataParser(i,'Classification')
        print(i)
        print(df)#.to_markdown())
    #########################################
