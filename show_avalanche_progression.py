"""
Showing some plots for avalanche progression data (see sweep)
"""
from __future__ import division

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as pplot
from matplotlib.cm import Reds
from mpl_toolkits.mplot3d import Axes3D
from avalanche_progression import AvalancheProgression

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print ''
        print ''
        print "Welcome to JustShowMe. Please provide one aggregateId to plot."
        print ''
        print ''
        exit()

    else:
        file_path = './simulation_data/avalanche_progression/%s.bin'%sys.argv[1]
        with file(file_path,'rb') as fp:
            list_of_avalanche_progressions = pickle.load(fp)
            conf = pickle.load(fp)

        print "Number of avalanches in list: "+str(len(list_of_avalanche_progressions))

        for prgrss in list_of_avalanche_progressions:
            print "Progression size: "+str(prgrss.size)
            print "Critical nodes: ",len(prgrss.critical_nodes)


            ks = sorted(prgrss.distributions.keys(),reverse=True)
            xseries = np.arange(-20,21)
            img = np.zeros((len(ks),40))
            for distr in ks:
                if distr % 2 == 1:
                    continue

                act_dist = []
                for i in xrange(len(prgrss.distributions[distr])):
                    if(prgrss.distributions[distr][i] > 20 or prgrss.distributions[distr][i] < -20):
                        continue
                    act_dist.append(prgrss.distributions[distr][i])

                ys,xs = np.histogram(act_dist,bins=40, range=(-20,20),normed=True)

                img[distr,:] = ys

            fig = pplot.figure()
            ax = fig.add_subplot(411)
            ax.pcolor(img,cmap=Reds)

            ax2 = fig.add_subplot(412)
            aff = np.array(prgrss.caused_defaults)[:,2]
            dflts = np.array(prgrss.caused_defaults)[:,0]
            crits = np.array(prgrss.caused_defaults)[:,1]
            x = range(len(dflts))
            #ax2.bar(x,aff,color='g')
            ax2.bar(x,dflts,color='b')
            ax2.bar(x,crits,color='r')

            ax2 = fig.add_subplot(413)
            x = [i for i in range(conf['model']['no_banks']) if not i in prgrss.critical_nodes]
            y = [prgrss.no_affected[i] for i in x]
            ax2.bar(x,y)
            y = [prgrss.no_affected[i] for i in prgrss.critical_nodes]
            ax2.bar(prgrss.critical_nodes,y, color='r')

            ax2 = fig.add_subplot(414)
            y = [prgrss.no_affected[i] for i in prgrss.default_order]
            ax2.bar(xrange(len(prgrss.default_order)),y)

            pplot.show()
            raw_input("wanna continue?")
