"""
network density for single aggregate
"""
from __future__ import division

import sys
import os
import pickle
import json
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as pplot
import scipy.stats as stats

#basepath = './simulation_data/large_avalanche_data'
basepath = './simulation_data/deg_distribution/irs_value_7'

def getconfig(aggid):
    file_path = '%s/%s.config'%(basepath,aggid)
    with open(file_path,'r') as fp:
        return json.load(fp)

def load_pickle(aggid,filename):
    file_path = '%s/%s.bin'%(basepath,aggid)#,filename)
    with open(file_path,'rb') as fp:
        return pickle.load(fp)

def get_degrees(aggid):
    return load_pickle(aggid,'degrees')

def get_agg_info(aggid, no):
    file_path = '%s/%s_%s.bin'%(basepath,aggid, no)
    config = None
    degrees = None
    with open(file_path,'rb') as fp:
        config = pickle.load(fp)
        degrees = pickle.load(fp)
    return config, degrees

if __name__ == '__main__':

    aggregate = False
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Network denisty analysis. We're now gonna make a nice graph of all data in deg_distribution"
        print ''
        print ''
        aggregate = True
    elif(len(sys.argv) == 2):
        print "Welcome to Network density analysis. AggregateId from sim_data/large_avalanche_data"
        aggregate_id = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Large avalanche analysis. Please provide ONE aggregateId to plot."
        print ''
        print ''
        exit()

    if(not aggregate):
        config = getconfig(aggregate_id)
        print "%d #nodes"%config['model']['no_banks']
        print "%d threshold"%config['model']['threshold']
        print "%d tenure"%config['model']['max_tenure']
        print "%d irs value"%config['model']['max_irs_value']

        degrees_over_time = get_degrees(aggregate_id)

        no_nodes = config['model']['no_banks']
        potential = (no_nodes*(no_nodes-1))/2
        densities = np.zeros(len(degrees_over_time))
        for i,degrees in enumerate(degrees_over_time):
            actual = sum(degrees)/2

            densities[i] = actual/potential

        pplot.figure()
        pplot.plot(range(len(degrees_over_time)),densities)

        pplot.figure()
        pplot.plot(range(1,10),[np.percentile(densities,i*10) for i in range(1,10)])

        pplot.show()
    else:
        aggregate_ids = []
        for filename in os.listdir(basepath):
            aggregate_ids.append(filename.split('_')[0])

        runs = defaultdict(int)

        for f in aggregate_ids:
            runs[f] += 1

        density_per_tenure = {}

        for a in runs:
            no = runs[a]
            degrees = None
            for i in range(no):
                config, run_degrees = get_agg_info(a,i)
                no_nodes = config['model']['no_banks']
                if(degrees is None):
                    degrees = run_degrees
                else:
                    degrees = np.concatenate([degrees, run_degrees])

            potential = (no_nodes*(no_nodes-1))/2
            densities = np.zeros(len(degrees))
            for i,degs in enumerate(degrees):
                actual = sum(degs)/2

                densities[i] = actual/potential

            percentile_densities = [np.percentile(densities, 25), np.percentile(densities, 50),np.percentile(densities, 75)]
            density_per_tenure[config['model']['max_tenure']] = percentile_densities

            print "Done one"

        tenures = sorted(density_per_tenure.keys())

        fig = pplot.figure()
        pplot.ylabel('Network Density')
        pplot.xlabel('Tenure')

        pplot.plot(tenures, [density_per_tenure[t][0] for t in tenures],label='25th Percentile')
        pplot.plot(tenures, [density_per_tenure[t][1] for t in tenures],label='50 Percentile')
        pplot.plot(tenures, [density_per_tenure[t][2] for t in tenures],label='75 Percentile')
        pplot.show()
