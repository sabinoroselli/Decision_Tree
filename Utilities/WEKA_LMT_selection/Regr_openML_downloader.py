import openml
import wget
import os

# the whole database on openML
datalist = openml.datasets.list_datasets(output_format="dataframe")
# # filter out all those that are NOT binary classification
datalist = datalist[
    (datalist.NumberOfClasses == 0) &
    (datalist.version == 1) &
    (datalist.NumberOfNumericFeatures <= 50) &
    (datalist.NumberOfSymbolicFeatures <= 50) &
    (datalist.NumberOfInstances >= 100) &
    (datalist.NumberOfInstances <= 30000)
    ]

datalist.sort_values('NumberOfInstances',inplace=True,ascending=True)

# let's just get the first n
# datalist = datalist.iloc[:20]

# print(datalist.head().to_markdown())
# print(datalist.to_markdown())
# print(datalist.shape)

to_keep = [
    43714,
    45718,
    43878,
    296,
    43919,
    43431,
    43926,
    45761,
    557,
    456,
    535,
    506,
    494,
    551,
    521,
    523,
    526,
    504,
    500,
    520,
    45588,
    45591,
    533,
    513,
    482,
    536,
    43050,
    43056,
    42931,
    195,
    196,
    207,
    43927,
    512,
    45586,
    43466,
    558,
    572,
    434,
    1199,
    560,
    531,
    543,
    43465,
    44152,
    42900,
    224,
    43939,
    43471,
    695,
    204,
    664,
    703,
    712,
    689,
    194,
    210,
    44052,
    45538,
    45744,
    41700,
    41928,
    534,
    43963,
    561,
    197,
    227,
    43978,
    42183,
    23516,
    42360,
    198,
    43384,
    41514,
    709,
    704,
    676,
    699,
    222,
    1099,
    42361,
    216,
    43918,
    1030,
    1027,
    44026,
    43308,
    232,
    42363,
    43440,
    1435,
    1571,
    199,
    1436,
    43785,
    40916,
    1574,
    42165,
    537,
    528,
    231,
    1097,
    43665,
    45064,
    45040,
    43442,
    44311,
    1414,
    45075,
    42364,
    8,
    545,
    42636,
    203,
    1245,
    230,
    41187,
    566,
    43747,
    43812,
    213,
    43582,
    509,
    43379,
    229,
    44028,
    209,
    663,
    40601,
    44212,
    294,
    546,
    42889,
    45074,
    45062,
    665,
    541,
    507,
    45617,
    223,
    42545,
    549,
    44029,
    23515,
    42439,
    544,
    42367,
    44269,
    42352,
    497,
    519,
    678,
    42464,
    42368,
    42369,
    503,
    287,
    191,
    42370
]

################### DOWNLOAD AARF FILES ##############
datalist.reset_index()
for index in to_keep:
# for index, row in datalist.iterrows():
    # print(index)
    dataset = openml.datasets.get_dataset(index, download_data=False)
    url = dataset.url
    name = url.split('/')[-1]
    # wget.download(url,out=f'/Users/sabinoroselli/WEKA_LMT_selection/Regr_DFs/{name}')
    print(name,len(dataset.features))#, row['NumberOfInstances'], row['NumberOfFeatures'])#


