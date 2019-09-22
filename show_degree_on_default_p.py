"""OLD"""
from __future__ import division

import sys
import os
import pickle
import json

from collections import Counter, defaultdict

from linreg import LinearRegression
from scipy import stats

import networkx as nx
import numpy as np
import matplotlib.pyplot as pplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import Reds, coolwarm
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.ndimage.filters import gaussian_filter1d


if __name__ == '__main__':
    aggregate_id = None
    ensemble = False
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Degree on default. Please provide an aggregateId to plot."
        print ''
        print ''
        ensemble = True
    elif(len(sys.argv) == 2):
        aggregate_id = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Degree on default. Please provide ONE aggregateId to plot."
        print ''
        print ''
        exit(0)

    path = 'simulation_data/deg_default/%s.bin'%aggregate_id

    counts = defaultdict(lambda: defaultdict(int))

    with open(path,'rb') as fp:
        degrees_on_default = pickle.load(fp)

    my = 0
    mx = 0
    cnt = 0
    for (d,s) in degrees_on_default:
        counts[d][s] += 1
        my = max(my,s)
        mx = max(mx,d)

    img = np.zeros((mx+1,my+1))

    for x in counts:
        for y in counts[x]:
            img[x,y] = counts[x][y]

    for i in range(my):
        img[:,i] = img[:,i]/np.max(img[:,i])


    average_degree = defaultdict(list)

    for (d,s) in degrees_on_default:
        average_degree[s].append(d)
    averages = [sum(average_degree[x])/len(average_degree[x]) for x in average_degree.keys()]
    
    xs = np.array(average_degree.keys())
    ys = np.array(averages)
    xs = xs.reshape((len(xs), 1))
    ys = ys.reshape((len(ys), 1))
    print ys.shape, xs.shape
    x = LinearRegression()
    
    y = x.fit(xs, ys)#,weight)

    r = x.coef_
    n = len(averages)
    t = (r*np.sqrt(n - 2))/(np.sqrt(1-(r**2)))
    print r, stats.t.sf(np.abs(t), n-1)*2

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Avalanche size s")
    ax.set_ylabel("Degree of first defaulting node")
    ax.scatter(average_degree.keys(), averages)
    pplot.show()


