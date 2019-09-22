"""
timeseries for the amount of riks in the system.
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

basepath = "simulation_data/risk/"


if __name__ == '__main__':

    for i in [1,2,4,5]:
        filepath = "%s/type%s.bin"%(basepath,i)

        with open(filepath,'rb') as fp:
            time_series = pickle.load(fp)

        fig = pplot.figure()
        ax = fig.add_subplot(111)

        ax.plot(xrange(len(time_series)), time_series)

        ax.set_ylim([0,7000])
        ax.set_ylabel("Absolute risk")
        ax.set_xlabel("Time")

        pplot.savefig('/home/..../Programming/GitRepos/scriptie/figs/time_series/type%d.png'%i, dpi=fig.dpi)
