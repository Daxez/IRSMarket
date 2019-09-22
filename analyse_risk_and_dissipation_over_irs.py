"""
    Plots risk going in vs risk going out. Must have datafiles in simulation_data/risk_and_balance_dissipation/over_ten
    So sweep over your parameters first and then run this to get a graph on it. (Not the most usefull though)
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

    file_path = 'simulation_data/risk_and_balance_dissipation/over_ten'
    config = None
    abs_added_risk = None
    abs_dissipated_risk = None
    added_risk = dict()
    dissipated_risk = dict()
    for f in os.listdir(file_path):
        with open('%s/%s'%(file_path,f),'rb') as fp:
            config = pickle.load(fp)
            abs_added_risk = pickle.load(fp)
            abs_dissipated_risk = pickle.load(fp)

        added_risk[config['model']['max_tenure']] = np.mean(abs_added_risk)
        dissipated_risk[config['model']['max_tenure']] = np.mean(abs_dissipated_risk)
        #added_risk[config['model']['max_irs_value']] = np.mean(abs_added_risk)
        #dissipated_risk[config['model']['max_irs_value']] = np.mean(abs_dissipated_risk)



    fig = pplot.figure()
    fig.suptitle('Dissipated risk and added risk 200 nodes, 550 tenure threshold 10')
    ax = fig.add_subplot(111)
    keys = sorted(added_risk.keys())
    ax.plot(keys, [added_risk[k] for k in keys], label='Mean Added risk')
    ax.plot(keys, [dissipated_risk[k] for k in keys], label='Mean Dissipated risk')
    ax.set_ylabel('Mean Risk')
    ax.set_xlabel('Irs value')
    pplot.legend(loc=4)
    pplot.show()
