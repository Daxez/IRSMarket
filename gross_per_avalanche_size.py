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

    basepath = './simulation_data/gross_risk_for_avalanche_size'

    data = {}

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
    

    pplot.figure()

    cmap = pplot.get_cmap('YlOrRd')
    n = len(data.keys())+1

    for (i,j) in enumerate(sorted(data.keys())):
        density_dict = data[j]

        print i
        pplot.scatter(density_dict.keys(), [sum(density_dict[k])/len(density_dict[k])/j for k in density_dict.keys()], color=cmap((i+1)/n))

    pplot.show()

