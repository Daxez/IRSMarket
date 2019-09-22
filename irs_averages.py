"""
old
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

# all with T = 15

basepath = './simulation_data/k'

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
        print "Welcome to IRS average analysis. We're now gonna make a nice graph of all data in k/irs_value_x"
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
            print folder
            if(not folder.startswith('irs_value_')):
                continue
            irs_value = int(float(folder.split('_')[-1]))
            print irs_value

            aggregate_ids = []
            for filename in os.listdir('%s/irs_value_%s.0'%(basepath,irs_value)):
                aggregate_ids.append(filename.split('_')[0])

            runs = defaultdict(int)

            for f in aggregate_ids:
                runs[f] += 1

            density_per_tenure = {}

            for a in runs:
                no = runs[a]
                degrees = None
                for i in range(no):
                    config, run_degrees = get_agg_info(folder,a,i)
                    no_nodes = config['model']['no_banks']
                    if(degrees is None):
                        degrees = run_degrees
                    else:
                        degrees = np.concatenate([degrees, run_degrees])

                samples = np.random.randint(10,len(degrees),1000)
                average_irss = np.zeros(len(samples))

                for i,samp in enumerate(samples):
                    degs = degrees[samp]
                    average_irss[i] = (sum(degs)/2)/no_nodes

                density_per_tenure[config['model']['max_tenure']] = average_irss
                per_irs_value[irs_value] = density_per_tenure

                print "Done one"

        filepath = '%s/aggregated.dict'%basepath
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

    print irs_values
    print tenure_values

    for term in tenure_values:

        
        values = [sum(per_irs_value[x][term])/len(per_irs_value[x][term]) for x in irs_values]
        errors = [np.std(per_irs_value[x][term]) for x in irs_values]
        fig, ax = pplot.subplots()
        ax.set_title('Average number of IRSs for a term to maturity of %d'%term)
        ax.set_ylabel('Average number of IRSs')
        ax.set_xlabel('IRS value')
        ax.set_ylim([0,500])
        ax.plot(irs_values, values, label='Measured')
        ax.plot(irs_values, [term/(v**2) for v in irs_values], label='Expected')
        pplot.legend()

        
        fl = '/home/..../Programming/GitRepos/scriptie/figs/no_irss/term_%d.png'%term

        pplot.savefig(fl, bbox_inches='tight')
