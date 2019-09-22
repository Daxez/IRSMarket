"""
Running sweep and create some specific graphs
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

    for tenure in [200,400,600,800,1000,1200,1400]:
        for dissipation in [0,0.25,0.5,0.75]:
            for rep in range(5):
                dc = DataContainer(dcconfig,str(uuid4()),str(uuid4()))
                p = Progress(steps)
                dcconfig['model']['max_tenure'] = tenure
                dcconfig['model']['dissipation'] = dissipation

                s = sim(dcconfig['model'],dc,p.update,save_risk,save_dist)
                start = time.time()

                p.start()
                s.run()
                p.finish()

                print ""
                print "Run took %d seconds"%(time.time()-start)

                file_path = './simulation_data/rsk/%s.bin'%dc.aggregate_id

                with file(file_path,'wb') as fp:
                    pickle.dump(float(np.percentile(s.risk,95)))
                    pickle.dump(float(np.percentile(s.risk,85)))
                    pickle.dump(float(np.percentile(s.risk,50)))
                    pickle.dump(float(max(s.risk)))
                    pickle.dump(tenure,fp)
                    pickle.dump(dissipation,fp)
                    pickle.dump(dcconfig['model']['no_banks'],fp)

#Disable if pypy
def plot_stuff():

    x = [200,400,600,800,1000,1200,1400]
    x2 = [0,0.25,0.5,0.75]

    folder = "./simulation_data/rsk"
    percentiles = {}
    mr = {}

    for t in x:
        percentiles[str(t)] = {}
        mr[str(t)] = {}
        for d in x2:
            mr[str(t)][str(d)] = []
            percentiles[str(t)][str(d)] = []

    for (i,item) in enumerate(os.listdir(folder)):
        file_path = "%s/%s"%(folder,item)
        with file(file_path,'rb') as fp:
            p95 = pickle.load(fp)
            p85 = pickle.load(fp)
            p50 = pickle.load(fp)
            maxrisk = pickle.load(fp)
            tenure = pickle.load(fp)
            dissipation = pickle.load(fp)
            no_banks = pickle.load(fp)

        percentiles[str(tenure)][str(dissipation)].append(p95)
        mr[str(tenure)][str(dissipation)].append(maxrisk)


    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("95th percentile risk")
    ax.set_xlabel("Tenure")


    for d in x2:
        plotx = []
        ploty = []
        plote = []
        for t in x:
            plotx.append(t)
            ploty.append(np.mean(percentiles[str(t)][str(d)]))
            plote.append(np.std(percentiles[str(t)][str(d)]))

        lbl = "Dissipation of %1.2f"%d
        ax.errorbar(plotx,ploty,yerr=plote,label=lbl)

    pplot.legend(loc=2)

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Maximum risk")
    ax.set_xlabel("Tenure")

    x = [200,400,600,800,1000,1200,1400]
    x2 = [0,0.25,0.5,0.75]

    for d in x2:
        plotx = []
        ploty = []
        plote = []
        for t in x:
            plotx.append(t)
            ploty.append(np.mean(mr[str(t)][str(d)]))
            plote.append(np.std(mr[str(t)][str(d)]))

        lbl = "Dissipation of %1.2f"%d
        ax.errorbar(plotx,ploty,yerr=plote,label=lbl)

    pplot.legend()
    pplot.show()



if(__name__ == '__main__'):
    plot_stuff()
