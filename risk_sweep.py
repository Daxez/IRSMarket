"""
File for some dissipation/risk graphs
"""
from __future__ import division

import time
import itertools
import os
import pickle
import copy
from operator import itemgetter
from collections import defaultdict
from uuid import uuid4

import numpy as np
from utils import Progress, is_valid_uuid
from datacontainer import DataContainer
from quick import sim

#Disble if pypy
import matplotlib.pyplot as pplot

def do_sweep():
    steps = 1000000
    save = False
    save_risk = True
    save_dist = False

    dcconfig = {
        'model':{
            'no_banks' : 500,
            'no_steps' : steps,
            'threshold' : 20,
            'sigma' : 1,
            'irs_threshold':1,
            'max_irs_value' : 6,
            'dissipation' : 0.0,
            'max_tenure' : 800
        },
        'analysis':{
            'data_to_save':['defaults']
        },
        'file_root':'./simulation_data/',
        'market_type':7
    }

    for tenure in [100,300,500,700]:
        dc = DataContainer(dcconfig,str(uuid4()),str(uuid4()))
        p = Progress(steps)
        dcconfig['model']['max_tenure'] = tenure

        s = sim(dcconfig['model'],dc,p.update,save_risk,save_dist)
        start = time.time()

        p.start()
        s.run()
        p.finish()

        print "Run took %d seconds"%(time.time()-start)

        file_path = './simulation_data/risk_swp/%s.bin'%dc.aggregate_id

        with file(file_path,'wb') as fp:
            pickle.dump(s.risk.tolist(),fp)
            pickle.dump(tenure,fp)
            pickle.dump(s.max_default_size_t.tolist(),fp)
            pickle.dump(dcconfig['model']['no_banks'],fp)

#Disable if pypy
def plot_stuff():

    percentiles = defaultdict(float)

    fig = pplot.figure()
    ax = fig.add_subplot(1,1,1)
    for (i,item) in enumerate(os.listdir("./simulation_data/_risk_swp")):
        file_path = "./simulation_data/_risk_swp/%s"%item
        with file(file_path,'rb') as fp:
            risk = pickle.load(fp)
            d = pickle.load(fp)

        percentiles[d] = max(risk)#np.percentile(risk, 80)

        lbl = "Dissipation of %f"%d
        ax.plot(xrange(len(risk)),risk,label=lbl)

    ax.set_ylabel("Risk")
    ax.set_xlabel("Time")

    pplot.legend()

    x = sorted(percentiles.keys())
    y = np.array(sorted(percentiles.iteritems(),key=itemgetter(0)))[:,1]

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Maximum risk")
    ax.set_xlabel("Dissipation")
    ax.plot(x,y)
    pplot.show()



if(__name__ == '__main__'):
    plot_stuff()
