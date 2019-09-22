"""Also old, ignore stuff with networkx"""
from __future__ import division

import sys
import os
import pickle
import json

from collections import Counter, defaultdict

from linreg import LinearRegression

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

    if ensemble:

        path = 'simulation_data/deg_default/ensemble'
        counters = []

        for fn in os.listdir(path):
            degrees_on_default = None
            filepath = "%s/%s"%(path,fn)
            with open(filepath,'rb') as fp:
                degrees_on_default = pickle.load(fp)
            c = Counter(degrees_on_default)
            counters.append(c)

        freqs = defaultdict(int)

        tf = 0
        for c in counters:
            for k in c:
                freqs[k] += c[k]
                tf += c[k]

        x = np.zeros(len(freqs))
        y = np.zeros(len(freqs))
        yy = np.zeros(len(freqs))
        mu = 0
        for i,k in enumerate(sorted(freqs)):
            x[i] = k
            y[i] = freqs[k]/len(counters)

        allfreqs = []
        for i,d in enumerate(x):
            for f in range(int(y[i])):
                allfreqs.append(d)

        print np.mean(allfreqs)
        y = y/sum(y)
        mu = sum([i*j for i,j in zip(x,y)])

        std = np.std(allfreqs)
        print mu, std



        fig = pplot.figure()
        ax = fig.add_subplot(121)
        ax.plot(x,y)
        ax = fig.add_subplot(122)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(x[1:],y[1:])
        pplot.show()

    else:
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

        xs = []
        ys = []

        for x in counts:
            for y in counts[x]:
                xs.append(x)
                ys.append(y)
        
        xs = np.array(xs)
        ys = np.array(ys)
        xs = xs.reshape((len(xs), 1))
        ys = ys.reshape((len(ys), 1))
        print ys.shape, xs.shape
        x = LinearRegression()
        y = x.fit(xs,ys)#,weight)


        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, interpolation='nearest',cmap=Reds,vmin=0,vmax=1,origin='lower')
        ax.set_title("Distribution of degree per avalanche size")
        ax.set_xlabel("Avalanche size s")
        ax.set_ylabel("Degree of first defaulting node")
        pplot.show()


