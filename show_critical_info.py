"""
Attempt to plot some useful info on critical nodes (close to default) 
"""
from __future__ import division

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as pplot
from matplotlib.cm import Reds
from scipy.signal import savgol_filter

if __name__ == '__main__':

    if len(sys.argv) != 2 or not os.path.exists('./simulation_data/critical/%s.bin'%sys.argv[1]):
        print ''
        print ''
        if(len(sys.argv) == 2):
            print "Welcome to critical plot. Please provide a valid aggregateId to plot."
        else:
            print "Welcome to critical plot. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    else:

        file_path = './simulation_data/critical/%s.bin'%sys.argv[1]
        with file(file_path,'rb') as fp:
            list_of_critical = pickle.load(fp)
            avalanche_time_series = pickle.load(fp)
            giant_components = pickle.load(fp)


        xseries = len(list_of_critical)
        no_critical_banks = np.zeros(xseries)
        degree_critical_banks = np.zeros(xseries)
        no_sup_critical_banks = np.zeros(xseries)

        max_crit_degree = max([max(cc.critical_degrees) for cc in list_of_critical if len(cc.critical_degrees) > 0])
        max_cntc_degree = max([max(cc.critical_to_non_critical_degrees) for cc in list_of_critical if len(cc.critical_to_non_critical_degrees) > 0])
        img = np.zeros((10,len(no_critical_banks)))
        img_ctnc = np.zeros((10,len(no_critical_banks)))
        for (i,cc) in enumerate(list_of_critical):
            img[:,i] = np.histogram(cc.critical_degrees,range=(0,max_crit_degree))[0]
            img_ctnc[:,i] = np.histogram(cc.critical_to_non_critical_degrees,range=(0,max_cntc_degree))[0]

            no_critical_banks[i] = cc.no_critical
            no_sup_critical_banks[i] = cc.no_super_critical
            if(cc.no_critical > 0):
                degree_critical_banks[i] = sum(cc.critical_degrees)/cc.no_critical

        x = range(xseries)

        fig = pplot.figure()

        ax = fig.add_subplot(421)
        y = no_critical_banks
        ax.plot(x,y)
        y = no_sup_critical_banks
        ax.plot(x,y)

        axt = ax.twinx()
        axt.yaxis.set_label_position('right')
        axt.set_ylabel("Default cascade size")
        axt.plot(xrange(len(avalanche_time_series)), avalanche_time_series,'r-')
        ax.set_ylabel("Number of critical banks")
        ax.set_xlabel("Time")

        ax = fig.add_subplot(422)
        y = savgol_filter(no_critical_banks,51,0)
        ax.plot(x,y)
        y = savgol_filter(no_sup_critical_banks,51,0)
        ax.plot(x,y)

        axt = ax.twinx()
        axt.yaxis.set_label_position('right')
        axt.set_ylabel("Default cascade size")
        axt.plot(xrange(len(avalanche_time_series)), avalanche_time_series,'r-')

        ax.set_ylabel("Number of critical banks")
        ax.set_xlabel("Time")

        axi = fig.add_subplot(423)
        axi.pcolor(img,cmap=Reds)
        axi.set_ylabel("Degree from crit to crit")

        axi = fig.add_subplot(424)
        axi.pcolor(img_ctnc,cmap=Reds)
        axi.set_ylabel("Degree from crit to non crit")

        ax2 = fig.add_subplot(425)
        ax2.set_ylabel("Average critical degree")
        y = degree_critical_banks
        ax2.plot(x,y,'r-')

        ax2 = fig.add_subplot(426)
        ax2.set_ylabel("Average critical degree")
        y = savgol_filter(degree_critical_banks,51,0)
        ax2.plot(x,y,'r-')

        ax2 = fig.add_subplot(427)
        ax2.set_ylabel('Giant component size')
        ax2.plot(xrange(len(giant_components)),giant_components)
        non_giant_component = np.array(no_critical_banks) - np.array(giant_components)
        ax2.plot(xrange(len(giant_components)),non_giant_component)

        print len([x for x in non_giant_component if x < 0])

        pplot.show()
