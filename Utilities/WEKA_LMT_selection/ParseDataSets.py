import arff as rf
import os

if __name__ == "__main__":

    ClassDataBases = sorted(os.listdir('MultiClassDFs'))

    datasets = {}
    for file in ClassDataBases:

        data = rf.load(open(f'MultiClassDFs/{file}', 'rt'))

        instances = len(data['data'])
        features = len(data['attributes'])
        class_label = data['attributes'][-1][0]
        classes = data['attributes'][-1][1]

        datasets.update({file:{"inst":instances,'feats':features,'classes':len(classes)}})

    for i in dict(sorted(datasets.items(),key=lambda item:item[1]['inst'])):
        print(i.split('.')[0],datasets[i]['inst'],datasets[i]['feats'],datasets[i]['classes'])






