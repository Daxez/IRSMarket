"""Plotting degrees over time given the data and an aggreagateId"""
from __future__ import division

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as pplot
from matplotlib.cm import Reds

def getimg(data):
    dmin = np.min(data)
    dmax = np.max(data)

    return [np.histogram(a, bins=dmax, range=(dmin,dmax))[0] for a in data]

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print ''
        print ''
        print "Welcome to DegreePlotter. Please provide one aggregateId to plot."
        print ''
        print ''
        exit()

    else:
        file_path = './simulation_data/deg/%s.bin'%sys.argv[1]
        with file(file_path,'rb') as fp:
            degrees = pickle.load(fp)[:5000]
            no_irss = pickle.load(fp)[:5000]


        degree_img = getimg(degrees)
        del degrees

        irs_img = getimg(no_irss)
        del no_irss

        fig = pplot.figure()
        ax = fig.add_subplot(211)
        ax.pcolor(degree_img,cmap=Reds)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Time")

        ax = fig.add_subplot(212)
        ax.pcolor(irs_img,cmap=Reds)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Time")

        pplot.show()
