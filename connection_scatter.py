"""
Scatter plot of in/out connections on precalculated data.
"""
from __future__ import division

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as pplot

if __name__ == '__main__':

    if len(sys.argv) != 2 or not os.path.exists('./simulation_data/connection_scatters/%s.bin'%sys.argv[1]):
        print ''
        print ''
        if(len(sys.argv) == 2):
            print "Welcome to connection scatters. Please provide a valid aggregateId to plot."
        else:
            print "Welcome to connection scatters. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    else:

        file_path = './simulation_data/connection_scatters/%s.bin'%sys.argv[1]
        with file(file_path,'rb') as fp:
            list_of_scatters = pickle.load(fp)

        fig = pplot.figure()
        ax = fig.add_subplot(111)

        for scatter_list in list_of_scatters:
            ax.scatter(np.array(scatter_list)[:,0],np.array(scatter_list)[:,1])

        ax.set_ylabel("Floating")
        ax.set_xlabel("Fixed")

        pplot.show()
