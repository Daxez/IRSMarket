"""Checking for some distributions. Looks like itś old"""
from __future__ import division

import sys
import os
import pickle

import scipy.stats as stats
import math
import numpy as np

from data.models import *

def approximate_normal(data):
    return np.mean(data,axis=0)[0],np.std(data,axis=0)[0]

def approximate_binomial(data):
    """Data should be a list of tuples containing avalanche size s and trial size n"""
    return sum(data[:,0]/data[:,1])/len(data)

def ks_for_distributions(data):
    data = np.array(data)
    mu,sigma = approximate_normal(data)
    p = approximate_binomial(data)
    n = max(data[:,0])
    print p,n

    stats_norm = stats.kstest(data[:,0],'norm',args=(mu,sigma))
    stats_pois = stats.kstest(data[:,0],'poisson',args=(mu,))
    stats_binom = stats.kstest(data[:,0],'binom',args=(n,mu/n))

    return stats_norm, stats_pois, stats_binom

def load_data(aggregate_id):
    file_path = './simulation_data/dists/%s.bin'%aggregate_id
    with file(file_path,'rb') as fp:
        defaults = pickle.load(fp)
        no_banks = pickle.load(fp)

    return defaults,no_banks

def get_suspicious_data(other_data):
    x = sorted(other_data,key=lambda x: x[0])

    dists = [x[i+1][0] - x[i][0] for i in range(len(x)-1)]
    md = max(dists)
    if(md > 20):
        print "Extrapolating gap"
        mdi = np.where(np.array(dists) == md)[0][0]

        x1 = x[:mdi]
        x2 = x[mdi+1:]

        return x2

    raise Exception("No gap, no deal.")

if __name__ == '__main__':

    if len(sys.argv) != 2 or not os.path.exists('./simulation_data/dists/%s.bin'%sys.argv[1]):
        print ''
        print ''
        if(len(sys.argv) == 2):
            print "Welcome to risk. Please provide a valid aggregateId to plot."
        else:
            print "Welcome to risk. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    else:
        defaults,no_banks = load_data(sys.argv[1])

        data = get_suspicious_data(defaults)

        sn,sp,sb = ks_for_distributions(data)

        print "N:",len(data)
        print "Normal:",sn
        print "Binomial:",sb
        print "Poisson:",sp

        file_path = './simulation_data/dists/calc_%s.bin'%sys.argv[1]

        with file(file_path,'wb') as fp:
            pickle.dump((sn,sp,sb),fp)
