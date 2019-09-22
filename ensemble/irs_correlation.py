from __future__ import division

import sys
import os
import pickle
import json
from collections import defaultdict

import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pplot
import scipy.stats as stats
from matplotlib.cm import Reds, coolwarm


basepath = './simulation_data/large_avalanche_data'

def getconfig(aggid):
    file_path = '%s/%s/config.json'%(basepath,aggid)
    with open(file_path,'r') as fp:
        return json.load(fp)

def get_irss(aggid):
    file_path = '%s/%s/%s.bin'%(basepath,aggid,'no_irs')
    with open(file_path,'rb') as fp:
        no_irss = pickle.load(fp)
        irss_pb = pickle.load(fp)

    return no_irss, irss_pb

def irs_correlation():
    dd = defaultdict(lambda: defaultdict(int))
    m = 0
    my = 0
    mx = 0

    positives = []
    negatives = []

    for aggregate_id in os.listdir(basepath):
        with open("%s/%s/no_irs.bin"%(basepath,aggregate_id),'rb') as fp:
            no_irss = pickle.load(fp)
            irs_pb = pickle.load(fp)

            for ipb in irs_pb:
                for (x,y) in ipb:
                    dd[x][y] += 1
                    positives.append(x)
                    negatives.append(y)
                    m = max(m,dd[x][y])
                    my = max(my,y)
                    mx = max(mx,x)
            print "Done %s"%aggregate_id

    mm = max(mx,my)
    img = np.zeros((mm+1,mm+1))

    for x in dd:
        for y in dd[x]:
            img[x,y] = dd[x][y]/m

    x = np.arange(mm+1)
    y = np.arange(mm+1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([img[x,y] for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    cmp = pplot.get_cmap('Reds')
    fig = pplot.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.plot_surface(X, Y, Z,cmap=cmp,rstride=1,cstride=1,linewidth=0,antialiased=False)

    ax.set_xlabel('Positive')
    ax.set_ylabel('Negative')
    ax.set_zlabel('Fraction')


    pplot.figure()
    cax = pplot.imshow(img,interpolation='nearest',cmap=cmp, vmin=0, vmax=1,origin='lower')
    pplot.xlabel("Number of positive ends")
    pplot.ylabel("Number of negative ends")
    pplot.colorbar(cax)

    fig = pplot.figure()
    ax = fig.add_subplot(211)
    ax.hist(positives, bins=max(positives), log=True, normed=True)

    ax = fig.add_subplot(212)
    ax.hist(negatives, bins=max(negatives), log=True, normed=True)

    pplot.show()

if __name__ == '__main__':
    irs_correlation()
