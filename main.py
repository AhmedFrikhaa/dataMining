from sklearn import datasets
from itertools import cycle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

irisData = datasets.load_iris()

def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')  # cycle de couleurs
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 2], data[target == i, 3], c=c, label=label)
    pl.plot([2.5, 2.5], [0, 3])
    pl.plot([0.75,7], [0.75, 0.75])

    pl.legend()
    pl.show()


#plot_2D(irisData.data, irisData.target, irisData.target_names)
print (irisData.data)