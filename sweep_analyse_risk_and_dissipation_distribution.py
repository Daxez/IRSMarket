"""
Old?
"""
from __future__ import division

import sys
import os
import pickle
import json
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as pplot
import scipy.stats as stats

if __name__ == '__main__':
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Risk/dissipation analysis. We're now gonna make a nice graph of all data in the sweep"
        print ''
        print ''
    elif(len(sys.argv) == 2):
        print "Welcome to Risk/dissipation analysis. No single aggregate ids "
        exit()
    else:
        print ''
        print ''
        print "Welcome to Risk/dissipation analysis. Please provide NO aggregateId to plot."
        print ''
        print ''
        exit()

    np.random.seed(None)
    #sweep values
    bank_sweep = [100]
    max_irs_value_sweep = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
    tenure_sweep = [50,100,250,400,550,700,850]
    threshold_sweep = [10]#[10,15,20,25,30]

    idDict = {}
    for i in max_irs_value_sweep:
        idDict[int(i)] = {}
        for t in tenure_sweep:
            idDict[int(i)][t] = []

    directory = 'simulation_data/risk_and_balance_dissipation/sweep_200'
    save_directory = '/home/..../Programming/GitRepos/scriptie/figs/risk_diss/200_b/distributions'

    imdata = np.zeros((len(max_irs_value_sweep),len(tenure_sweep)))

    for f in os.listdir(directory):
        config = None
        file_path = '%s/%s'%(directory,f)
        with open(file_path,'rb') as fp:
            config = pickle.load(fp)

        n = config['model']['no_banks']
        t = config['model']['max_tenure']
        i = config['model']['max_irs_value']
        T = config['model']['threshold']

        idDict[int(i)][t].append(f)

    for i in max_irs_value_sweep:
        for t in tenure_sweep:
            ratios = []
            windowSize = 500

            for f in idDict[int(i)][t]:
                file_path = '%s/%s'%(directory,f)
                with open(file_path,'rb') as fp:
                    config = pickle.load(fp)
                    abs_added_risk = pickle.load(fp)
                    abs_dissipated_risk = pickle.load(fp)


                noBins = math.ceil(len(abs_added_risk)/windowSize)
                for j in range(int(noBins)):
                    start = j*windowSize
                    end = min((j+1)*windowSize, len(abs_added_risk))
                    aar = abs_added_risk[start:end]
                    adr = abs_dissipated_risk[start:end]
                    ratios.append(np.mean(adr)/np.mean(aar))

            if True:
                fig = pplot.figure()
                ax = fig.add_subplot(111)
                #ax.set_xlim((0,2.5))
                #ax.set_ylim((0,45))
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Dissipated risk / added risk")
                ax.hist(ratios, 50)

                filename = 'dissipation_percentages_ten_%d_irs_%d.png'%(t,i)
                fl = '%s/%s'%(save_directory, filename)
                pplot.savefig(fl,bbox_inches='tight')
                pplot.close(fig)
            elif False:
                #imdata[max_irs_value_sweep.index(i)][tenure_sweep.index(t)] += np.std(ratios)
                imdata[max_irs_value_sweep.index(i)][tenure_sweep.index(t)] += np.std(ratios)
            else:
                imdata[max_irs_value_sweep.index(i)][tenure_sweep.index(t)] += np.percentile(ratios, 95)


    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("IRS Value")
    ax.set_yticks(np.arange(0,10))
    ax.set_yticklabels(np.arange(1,10))
    ax.set_xlabel("Tenure")
    ax.set_xticklabels([0,50,100,250,400,550,700,850])
    ax.set_title("Heatmap of 95th percentile in out/in ratio for 100 banks, threshold 10")

    cax = ax.imshow(imdata,origin='lower',interpolation='nearest')

    cb = fig.colorbar(cax)
    #cb.set_ticks([min_value, max_value])
    #cb.set_ticklabels([min_value, max_value])
    filename = 'percentile_95_in_out_distribution.png'
    fl = '%s/%s'%(save_directory, filename)
    pplot.savefig(fl,bbox_inches='tight')
