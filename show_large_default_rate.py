"""
Waiting times/rate for larger defaults.
"""
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

basepath = "simulation_data/large_default_rate"


if __name__ == '__main__':
    aggregate_id = None
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Degree on default. Please provide an aggregateId to plot."
        print ''
        print ''
        aggregate_id = None
    elif(len(sys.argv) == 2):
        aggregate_id = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Degree on default. Please provide ONE aggregateId to plot."
        print ''
        print ''
        exit(0)

    if(aggregate_id != None):
        filepath = "%s/%s.bin"%(basepath,aggregate_id)

        default_rates = None

        with open(filepath,'rb') as fp:
            default_rates = pickle.load(fp)


        descr = stats.describe(default_rates)
        print descr

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.hist(default_rates, normed=True, bins=100)
        ax.set_title("Default rate")
        ax.set_xlabel("Time between defaults")
        ax.set_ylabel("Normalized frequency")

        pplot.show()
    
    else:
        files = os.listdir(basepath)

        data = defaultdict(list)

        for file in files:
            if (len(file) < 30): continue
            with open("%s/%s"%(basepath,file), 'rb') as fp:
                data[pickle.load(fp)].extend(pickle.load(fp))
        
        boxdata = []

        labels = sorted(data.keys())
        for no_nodes in labels:
            boxdata.append(data[no_nodes])
        
        pplot.figure()

        pplot.boxplot(boxdata, labels=labels)
        pplot.show()


