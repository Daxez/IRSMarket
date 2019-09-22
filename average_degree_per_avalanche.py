"""
file for showing the average degree on a default (run simulation to collect data first)
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


if __name__ == '__main__':

    basepath = './simulation_data/average_degree_on_default'

    data = {}
    start_data = {}

    for file in os.listdir(basepath):
        with open('%s/%s'%(basepath, file),'rb') as fp:
            config = pickle.load(fp)
            N = config['model']['no_banks']
            ndata  = pickle.load(fp)
            if(data.has_key(N)):
                for k in ndata.keys():
                    data[N][k].extend(ndata[k])
            else:
                data[N] = ndata

            ndata = pickle.load(fp)
            if(start_data.has_key(N)):
                for k in ndata.keys():
                    start_data[N][k].extend(ndata[k])
            else:
                start_data[N] = ndata
    

    pplot.figure()
    pplot.xlabel('Avalanche Size over network size (S/N)')
    pplot.ylabel('Average degree of network')

    cmap = pplot.get_cmap('YlOrRd')
    n = len(data.keys())+1

    for (i,j) in enumerate(sorted(data.keys())):
        density_dict = data[j]

        print i
        l = '%d nodes'%j
        pplot.scatter([c/j for c in density_dict.keys()], [(sum(density_dict[k])/len(density_dict[k])) for k in density_dict.keys()], color=cmap((i+1)/n), label=l)

    #pplot.xlim((0,400))
    pplot.legend(loc='upper left')
    pplot.show()


    # First defaulting node
    for n in start_data.keys():
        avalanches = start_data[n]
        counts = defaultdict(lambda: defaultdict(int))

        max_d = 0
        min_d = n

        for s in avalanches:
            for d in avalanches[s]:
                counts[s][d] += 1
                max_d = max(max_d, d)
                
        img = np.zeros((max_d+2, max(counts.keys())+2))
        print img.shape
        for x in counts:
            for y in counts[x]:
                img[y,x] = counts[x][y]

        for i in range( max(counts.keys())+2):
            img[:,i] = img[:,i]/np.max(img[:,i])
        
        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, interpolation='nearest',cmap=pplot.get_cmap('Reds'),origin='lower')
        ax.set_title("Distribution of degree per avalanche size")
        ax.set_xlabel("Avalanche size s")
        ax.set_ylabel("Degree of first defaulting node")
        pplot.show()

        