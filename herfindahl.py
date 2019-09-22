"""
Some brainwave about herfindahl  
"""

from __future__ import division

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as pplot


if __name__ == '__main__':

    if len(sys.argv) != 2 or not os.path.exists('./simulation_data/herfindahl/%s.bin'%sys.argv[1]):
        print ''
        print ''
        if(len(sys.argv) == 2):
            print "Welcome to Herfindahl. Please provide a valid aggregateId to plot."
        else:
            print "Welcome to Herfindahl. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    else:

        file_path = './simulation_data/herfindahl/%s.bin'%sys.argv[1]
        with file(file_path,'rb') as fp:
            herfindahl = pickle.load(fp)
            defaults = pickle.load(fp)
            no_banks = pickle.load(fp)

        print np.mean(herfindahl)

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        hp, = ax.plot(xrange(len(herfindahl)),herfindahl,'b-',label='Herfindahl')
        ap, = ax.plot(xrange(len(herfindahl)),[1/no_banks]*len(herfindahl),'g-',label='1/%d'%no_banks)

        ax2 = ax.twinx()
        ax2.yaxis.set_label_position('right')
        dp, = ax2.plot(xrange(len(defaults)),defaults,'r-', label='Largest default')

        pplot.legend(handles=[hp,ap,dp])
        pplot.show()
