import matplotlib.pyplot as plt

def make_instance(num_points):
    filename = 'prova'

    first = [0.3,-5]
    second = [-0.6,6]
    third = [0.8,-7]
    fourth = [-2,3]

    container = []
    for i in range(num_points):
        if i < num_points/4:
             container.append({'x':i,'y':(first[0]*i + first[1])})
        elif i >= num_points/4 and i < 2*num_points/4:
            container.append({'x': i, 'y': (second[0] * i + second[1])})
        elif i >= 2 *num_points/4 and i < 3*num_points/4:
            container.append({'x': i, 'y': (third[0] * i + third[1])})
        else:
            container.append({'x': i, 'y': (fourth[0] * i + fourth[1])})

    x = []
    y = []
    for i in container:
        x.append(i['x'])
        y.append(i['y'])

    # plt.plot(x,y,linestyle='none',marker='o')
    # plt.show()
    with open(f'test_instances/{filename}','w+') as fn:
        fn.writelines('X,Y\n')
        for i in container:
            fn.writelines(f"{i['x']},{i['y']}\n")
    return

