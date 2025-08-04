import openml
import wget
import os

# the whole database on openML
datalist = openml.datasets.list_datasets(output_format="dataframe")
# # filter out all those that are NOT binary classification
datalist = datalist[
    (datalist.NumberOfClasses > 2) &
    (datalist.version == 1) &
    (datalist.NumberOfNumericFeatures <= 50) &
    (datalist.NumberOfSymbolicFeatures <= 50) &
    (datalist.NumberOfInstances >= 100) &
    (datalist.NumberOfInstances <= 10000)
    ]

datalist.sort_values('NumberOfInstances',inplace=True,ascending=True)

# let's just get the first n
# datalist = datalist.iloc[:20]

# print(datalist.head(n=100).to_markdown())
# print(datalist.to_markdown())
# print(datalist.shape)


################### DOWNLOAD AARF FILES ##############
datalist.reset_index()
# for index in to_keep:
for index, row in datalist.iterrows():
    # print(index)
    dataset = openml.datasets.get_dataset(row['did'], download_data=False)
    url = dataset.url
    name = url.split('/')[-1]
    wget.download(url,out=f'/Users/sabinoroselli/WEKA_LMT_selection/MultiClassDFs/{name}')
    print(name,row['NumberOfFeatures'],row['NumberOfNumericFeatures'])

to_keep = [
    43255,
    1455,
    41156,
    1037,
    179,
    1119,
    45051,
    44528,
    4135,
    448,
    461,
    450,
    1059,
    43051,
    44483,
    1547,
    463,
    1121,
    1460,
    44234,
    1462,
    481,
    1463,
    1464,
    13,
    15,
    45717,
    43979,
    45578,
    1447,
    40701,
    40710,
    1467,
    25,
    41538,
    40669,
    29,
    31,
    1075,
    42477,
    803,
    42882,
    37,
    40713,
    45712,
    41496,
    1471,
    151,
    43551,
    1473,
    41769,
    41721,
    40646,
    40650,
    43,
    44149,
    53,
    45023,
    55,
    43893,
    43895,
    1480,
    59,
    451,
    1048,
    1053,
    1067,
    993,
    44762,
    1448,
    3,
    1441,
    43595,
    1412,
    1120,
    310,
    1056,
    1442,
    44224,
    1450,
    40680,
    333,
    1046,
    24,
    40681,
    1071,
    43892,
    44226,
    311,
    45060,
    40706,
    1488,
    1068,
    4534,
    1489,
    1451,
    1443,
    1490,
    446,
    464,
    470,
    45077,
    1495,
    42172,
    1496,
    41160,
    1498,
    40900,
    466,
    747,
    38,
    44227,
    336,
    1504,
    43097,
    41146,
    42178,
    4329,
    40690,
    50,
    40945,
    40705,
    1507,
    56,
    1510,
    1511,
    43607,
    40693,
    43786
]

