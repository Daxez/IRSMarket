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

def getconfig(aggid):
    file_path = 'simulation_data/large_avalanche_data/%s/config.json'%aggid
    with open(file_path,'r') as fp:
        return json.load(fp)

if __name__ == '__main__':
    base = "simulation_data/large_avalanche_data/"
    degrees = np.array([])
    last_degrees = np.array([])

    flist = os.listdir(base)
    for d in flist:
        config = getconfig(d)
        path = "%s%s/degrees.bin"%(base,d)
        with open(path,'rb') as fp:
            degs = pickle.load(fp)

            degrees = np.concatenate((degrees,np.hstack(degs)))
            last_degrees = np.concatenate((last_degrees,degs[-1]))
        print "Done %s"%d

    sig_double_exp = np.std(degrees)
    dist_x = sorted(list(set(degrees)))

    t_repl = (2*config['model']['max_tenure'],len(flist))

    fig = pplot.figure()
    ax = fig.add_subplot(211)
    ax.hist(degrees,normed=True,bins=np.max(degrees))
    t = "Degree distribution of %d steps before avalance for %d repititions"%t_repl
    ax.set_title(t)

    ax = fig.add_subplot(212)
    t = "Degree distribution of last step before avalance for %d repititions"%len(flist)
    ax.set_title(t)
    ax.hist(last_degrees, normed=True, bins=np.max(last_degrees))
    ax.plot(dist_x, stats.laplace.pdf(dist_x,0,sig_double_exp))

    pplot.show()
