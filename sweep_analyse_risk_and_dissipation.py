"""
Old?
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

    analyse_steps = [100,1000,2000,5000,10000,20000,30000,40000,50000,60000]
    total_steps = 100000

    for s,noSteps in enumerate(analyse_steps):
        print "Step %d of %d"%(s, len(analyse_steps))
        mcIndices = np.random.choice(np.arange(total_steps), noSteps)

        imdata = np.zeros((len(max_irs_value_sweep),len(tenure_sweep)))
        imcount = np.zeros((len(max_irs_value_sweep),len(tenure_sweep)))
        imsamps = np.zeros((len(max_irs_value_sweep),len(tenure_sweep),5))

        directory = 'simulation_data/risk_and_balance_dissipation/sweep'
        save_directory = '/home/..../Programming/GitRepos/scriptie/figs/risk_diss/sampled'

        for f in os.listdir(directory):
            config = None
            abs_added_risk = None
            abs_dissipated_risk = None
            file_path = '%s/%s'%(directory,f)
            with open(file_path,'rb') as fp:
                config = pickle.load(fp)
                abs_added_risk = pickle.load(fp)
                abs_dissipated_risk = pickle.load(fp)

            abs_added_risk = np.array(abs_added_risk)[mcIndices]
            abs_dissipated_risk = np.array(abs_dissipated_risk)[mcIndices]
            n = config['model']['no_banks']
            t = config['model']['max_tenure']
            i = config['model']['max_irs_value']
            T = config['model']['threshold']

            imdata[max_irs_value_sweep.index(i)][tenure_sweep.index(t)] += np.mean(abs_dissipated_risk)/np.mean(abs_added_risk)
            cnt = imcount[max_irs_value_sweep.index(i)][tenure_sweep.index(t)]
            imsamps[max_irs_value_sweep.index(i)][tenure_sweep.index(t)][cnt] = np.mean(abs_dissipated_risk)/np.mean(abs_added_risk)
            imcount[max_irs_value_sweep.index(i)][tenure_sweep.index(t)] += 1

        #fig = pplot.figure()
        #subnr = 331
        #for k,i in enumerate(max_irs_value_sweep):
            #yerr = np.zeros(len(tenure_sweep))
            #for j,t in enumerate(tenure_sweep):
                #samples = imsamps[max_irs_value_sweep.index(i)][tenure_sweep.index(t)]
                #print "For %d tenure and %d irs."
                #print "Mean", np.mean(samples)
                #print "Std", np.std(samples)
                #print samples
                #print " --------------------------------"
                #yerr[j] = np.std(samples)

            #ax = fig.add_subplot(subnr+k)
            #ax.set_title("Irs value %d"%i)
            #ax.plot(tenure_sweep, imdata[:][tenure_sweep.index(t)]/5)
            #ax.set_xlabel("Tenure")
            #ax.set_ylabel("Outflux/Influx")
            #ax.fill_between(tenure_sweep,(imdata[:][tenure_sweep.index(t)]/5)-yerr, (imdata[:][tenure_sweep.index(t)]/5)+yerr)

        imdata = imdata/5.0

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel("IRS Value")
        ax.set_yticks(np.arange(0,10))
        ax.set_yticklabels(np.arange(1,10))
        ax.set_xlabel("Tenure")
        ax.set_xticklabels([0,50,100,250,400,550,700,850])
        ax.set_title("Heatmap for 100 banks, threshold 10")

        cax = ax.imshow(imdata,origin='lower',interpolation='nearest')

        cb = fig.colorbar(cax)
        #cb.set_ticks([min_value, max_value])
        #cb.set_ticklabels([min_value, max_value])
        filename = '1_dissipation_percentages_%d.png'%noSteps
        fl = '%s/%s'%(save_directory, filename)
        pplot.savefig(fl,bbox_inches='tight')
