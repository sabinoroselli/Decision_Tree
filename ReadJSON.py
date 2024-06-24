import json
import pandas as pd

probType = 'Classification'
# probType = 'Regression'
ModelType = 'STD' # STD, MOD

modeTree = {'Classification':'LMT','Regression':'M5P'}

algorithms = [f'Parallel_{ModelType}']#,modeTree[probType],'CART','RF','SVM']

all_results = []
for i in algorithms:
    with open(f'{probType}Results/{i}.json') as json_data:
        buffer = json.load(json_data)
        if probType == 'Regression':
            if i in [f'Parallel_{ModelType}']:
                sub_buffer = {}
                for key, value in buffer.items():
                    sub_buffer.update({key: {
                        'RelAbsErr': value['Metric']['RAE'],
                        'RelRootSqErr': value['Metric']['RRSE'],
                        'Leaves': value['Leaves'],
                        'RunTimes': value['RunTimes']
                    }})
                buffer = sub_buffer
        df = pd.DataFrame.from_dict(buffer).transpose()
        if i in [f'Parallel_{ModelType}']:
            df = df.drop(columns=['RunTimes'],axis=1)
        all_results.append(df)

metrics_and_leaves = pd.concat(all_results,axis=1,join='inner')
# print(metrics_and_leaves.to_markdown())
## TODO make a table to report times
all_splits = []
with open(f'{probType}Results/Parallel_{ModelType}.json') as json_data:
    buffer = json.load(json_data)
    for key,value in buffer.items():
        sub_buffer = {key:{k:v for k,v in value['RunTimes'].items() }}
        # print(pd.DataFrame.from_dict(value['RunTimes']).transpose())
        all_splits.append(pd.DataFrame.from_dict(sub_buffer))
    runtimes = pd.concat(all_splits, axis=1, join='inner').transpose()
all_data = pd.concat([metrics_and_leaves,runtimes],axis=1,join='inner')
print(all_data.to_markdown())
