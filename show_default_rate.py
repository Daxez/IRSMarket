"""Plot default rate i fyou have the info."""
from __future__ import division

import sys
import os
import pickle
import json
from collections import defaultdict

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pplot
import scipy.stats as stats
from matplotlib.cm import Reds, coolwarm

basepath = "simulation_data/default_rate"


if __name__ == '__main__':
    aggregate_id = None
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Degree on default. Please provide an aggregateId to plot."
        print ''
        print ''
        exit(0)
    elif(len(sys.argv) == 2):
        aggregate_id = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Degree on default. Please provide ONE aggregateId to plot."
        print ''
        print ''
        exit(0)

    filepath = "%s/%s.bin"%(basepath,aggregate_id)

    default_rates = None

    with open(filepath,'rb') as fp:
        default_rates = pickle.load(fp)


    descr = stats.describe(default_rates)


    print "Max ppcc %.4f"%stats.ppcc_max(default_rates)

    x = sorted(list(set(default_rates)))

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.hist(default_rates, normed=True, bins=100)
    ax.plot(x,stats.norm.pdf(x,descr.mean,np.sqrt(descr.variance)))
    ax.set_title("Default rate")
    ax.set_xlabel("Time between defaults")
    ax.set_ylabel("Normalized frequency")

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    stats.ppcc_plot(default_rates, -5, 5, plot=ax)

    pplot.show()
