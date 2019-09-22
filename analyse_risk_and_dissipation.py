"""
    Plots risk going in vs risk going out. Must have datafiles in simulation_data/risk_and_balance_dissipation/over_ten
    Do it for a specific aggregateId
    run as is
"""
from __future__ import division

import sys
import os
import pickle
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as pplot
import scipy.stats as stats

if __name__ == '__main__':
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Risk/dissipation analysis. We're now gonna make a nice graph of all data in deg_distribution"
        print ''
        print ''
        exit()
    elif(len(sys.argv) == 2):
        print "Welcome to Risk/dissipation analysis. AggregateId from sim_data/large_avalanche_data"
        aggregate_id = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Risk/dissipation analysis. Please provide ONE aggregateId to plot."
        print ''
        print ''
        exit()


    file_path = 'simulation_data/risk_and_balance_dissipation/%s.bin'%aggregate_id
    config = None
    abs_added_risk = None
    abs_dissipated_risk = None
    with open(file_path,'rb') as fp:
        config = pickle.load(fp)
        abs_added_risk = pickle.load(fp)
        abs_dissipated_risk = pickle.load(fp)

    print config
    print 'added risk mean: %.4f'%np.mean(abs_added_risk)
    print 'dissipated risk mean: %.4f'%np.mean(abs_dissipated_risk)
    print 'added risk sum: %.4f'%np.sum(abs_added_risk)
    print 'dissipated risk sum: %.4f'%np.sum(abs_dissipated_risk)
    print 'Disspation part: %.4f'%(np.mean(abs_dissipated_risk)/np.mean(abs_added_risk))

    n = config['model']['no_banks']
    t = config['model']['max_tenure']
    i = config['model']['max_irs_value']
    T = config['model']['threshold']
    fig = pplot.figure()
    fig.suptitle(('Nodes %d, tenure %d, IrsValue %d, Threshold %d')%(n,t,i,T))
    ax = fig.add_subplot(211)

    ax.plot(xrange(len(abs_added_risk)), abs_added_risk)
    ax = fig.add_subplot(212)
    ax.plot(xrange(len(abs_dissipated_risk)), abs_dissipated_risk)

    pplot.show()
