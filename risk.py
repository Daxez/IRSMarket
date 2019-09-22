"""
Plot risk data for an aggregated id
"""
from __future__ import division

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as pplot

if __name__ == '__main__':

    if len(sys.argv) != 2 or not os.path.exists('./simulation_data/risk/%s.bin'%sys.argv[1]):
        print ''
        print ''
        if(len(sys.argv) == 2):
            print "Welcome to risk. Please provide a valid aggregateId to plot."
        else:
            print "Welcome to risk. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    else:

        file_path = './simulation_data/risk/%s.bin'%sys.argv[1]
        with file(file_path,'rb') as fp:
            risk = pickle.load(fp)
            defaults = pickle.load(fp)
            no_banks = pickle.load(fp)

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        hp, = ax.plot(xrange(len(risk)),risk,'b-',label='Risk')
        ax.set_ylabel("Risk")
        ax.set_xlabel("Time")

        ax2 = ax.twinx()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel("Default cascade size")
        dp, = ax2.plot(xrange(len(defaults)),defaults,'r-', label='Default cascade size')

        #pplot.legend(handles=[hp,dp])
        pplot.show()
