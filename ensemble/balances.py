from __future__ import division

import sys
import os
import pickle
import json

from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as pplot
import scipy.stats as stats
from matplotlib.cm import Reds, coolwarm
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.ndimage.filters import gaussian_filter1d

if __name__ == '__main__':
    base = "simulation_data/large_avalanche_data/"
    balances = np.array([])
    gross_balances = np.array([])
    last_gross_balances = np.array([])


    flist = os.listdir(base)
    for d in flist:
        config = getconfig(d)
        path = "%s%s/balances.bin"%(base,d)
        with open(path,'rb') as fp:
            net = pickle.load(fp)
            gross = pickle.load(fp)

            balances = np.concatenate((balances,np.hstack(net)))
            gross_balances = np.concatenate((gross_balances,np.hstack(gross)))
            last_gross_balances = np.concatenate((last_gross_balances,gross[-1]))
        print "Done %s"%d

    sig_double_exp = np.std(gross_balances)
    dist_x = sorted(list(set(gross_balances)))

    fig = pplot.figure()
    ax = fig.add_subplot(311)
    ax.hist(balances, normed=True, bins = 100)
    ax.set_title("Net balances distribution of %d steps before avalance for x repititions")

    ax = fig.add_subplot(312)
    ax.set_title("Gross balances distribution of 500 steps before avalance for x repititions")
    ax.hist(gross_balances, normed=True, bins = 100)
    ax.plot(dist_x, stats.laplace.pdf(dist_x,0,sig_double_exp))

    ax = fig.add_subplot(313)
    ax.set_title("Gross balances distribution of last step before avalance for x repititions")
    ax.hist(last_gross_balances, normed=True, bins = 100)

    pplot.show()
