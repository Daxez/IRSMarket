"""
Create plots for degree distributions (deg_distribution)
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

#basepath = './simulation_data/deg_distribution/irs_value_7'
basepath = './simulation_data/deg_distribution'
save_directory = '/home/..../Programming/GitRepos/scriptie/figs/degree_distributions'

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

def get_agg_info(folder, aggid, no):
    file_path = '%s/%s/%s_%s.bin'%(basepath,folder, aggid, no)
    config = None
    degrees = None
    with open(file_path,'rb') as fp:
        config = pickle.load(fp)
        degrees = pickle.load(fp)
    return config, degrees

if __name__ == '__main__':

    aggregate = False
    datafilename = None
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Network density analysis. We're now gonna make a nice graph of all data in deg_distribution /irs_value_x"
        print ''
        print ''
        aggregate = True
    elif(len(sys.argv) == 2):
        print "Welcome to network density analysis aggreate, use plain"
        datafilename = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Large avalanche analysis. Please provide no aggregateids"
        print ''
        print ''
        exit()

    if(datafilename is None):
        per_irs_value = {}

        for folder in os.listdir(basepath):
            if(not folder.startswith('irs_value_')):
                continue
            irs_value = int(folder[-1])
            print irs_value

            aggregate_ids = []
            for filename in os.listdir('%s/irs_value_%s'%(basepath,irs_value)):
                aggid = filename.split('_')[0]
                if(aggid not in aggregate_ids):
                    aggregate_ids.append(aggid)

            print aggregate_ids
            degree_dist_per_tenure = {}

            for a in aggregate_ids:
                config, run_degrees = get_agg_info(folder,a,0)
                no_nodes = config['model']['no_banks']
                frequencies = defaultdict(int)
                for steps in run_degrees:
                    for node_degree in steps:
                        frequencies[int(node_degree)] += 1

                degree_dist_per_tenure[config['model']['max_tenure']] = frequencies

                per_irs_value[irs_value] = degree_dist_per_tenure

                print "Done one"

        filepath = '%s/degree_frequencies.dict'%basepath
        with file(filepath, 'wb') as fp:
            pickle.dump(per_irs_value, fp)
            print 'Saved file: %s'%filepath
    else:
        filepath = '%s/%s'%(basepath, datafilename)
        with file(filepath,'rb') as fp:
            per_irs_value = pickle.load(fp)
            print 'Loaded file: %s'%filepath


    irs_values = sorted(per_irs_value.keys())
    tenure_values = sorted(per_irs_value[irs_values[0]].keys())

    for irs_value in irs_values:
        for tenure_value in tenure_values:

            degrees = per_irs_value[irs_value][tenure_value]

            fig, ax = pplot.subplots()

            ax.set_title('Degree frequency for IRS value %d and tenure %d'%(irs_value,tenure_value))
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Degree')
            ax.set_ylim([0,20])
            ax.set_xlim([0,200])

            ax.bar(degrees.keys(), [v/100000 for v in degrees.values()])

            filename = 'run_degree_distribution_%d_irs_%d_ten.png'%(irs_value, tenure_value)
            fl = '%s/%s'%(save_directory, filename)
            pplot.savefig(fl, bbox_inches='tight')
            pplot.close()
