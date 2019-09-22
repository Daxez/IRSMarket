"""
Some network density graphs if you have the data (for a lot of configs).
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
                
                tenure = config['model']['max_tenure']

                potential = (no_nodes*(no_nodes-1))/2
                densities = np.zeros(len(degrees))
                for i,degs in enumerate(degrees):
                    actual = (sum(degs)/2)

                    densities[i] = actual/potential

                percentile_densities = [np.percentile(densities, 25), np.percentile(densities, 50),np.percentile(densities, 75)]
                density_per_tenure[tenure] = percentile_densities

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

    data = np.zeros((len(irs_values),len(tenure_values)))
    data2 = np.zeros((len(irs_values),len(tenure_values)))

    for i,x in enumerate(irs_values):
        for j,y in enumerate(tenure_values):
            # just get the average, which is the first object
            v = per_irs_value[x][y][1]
            data[i][j] = v * x*y
            data2[i][j] = v 

    print irs_values
    print tenure_values

    fig, ax = pplot.subplots()

    cax = ax.imshow(data, cmap=pplot.cm.Reds,
        interpolation='nearest', origin='lower')
    ax.set_title('Heatmap weighted network density')
    ax.set_ylabel('IRS value')
    ax.set_yticks(xrange(len(irs_values)))
    ax.set_yticklabels(irs_values)

    ax.set_xlabel('Tenure')
    ax.set_xticks(xrange(len(tenure_values)))
    ax.set_xticklabels(tenure_values)

    cb = fig.colorbar(cax)

    fig, ax = pplot.subplots()

    cax = ax.imshow(data2, cmap=pplot.cm.Reds,
        interpolation='nearest', origin='lower')
    ax.set_title('Heatmap network density for IRS values and term to maturity')
    ax.set_ylabel('IRS value')
    ax.set_yticks(xrange(len(irs_values)))
    ax.set_yticklabels(irs_values)

    ax.set_xlabel('Tenure')
    ax.set_xticks(xrange(len(tenure_values)))
    ax.set_xticklabels(tenure_values)

    cb = fig.colorbar(cax)

    pplot.show()

    lineplot = np.zeros(len(irs_values))
    for j,y in enumerate(irs_values):
        # just get the average, which is the first object
        v = per_irs_value[y][tenure_values[-1]][1]
        lineplot[j] = v

    lineplot_2irs = np.zeros(len(irs_values))
    for j,y in enumerate(irs_values):
        # just get the average, which is the first object
        v = per_irs_value[y][tenure_values[3]][1]
        lineplot_2irs[j] = v

    lineplot_tenure = np.zeros(len(irs_values))
    for j,y in enumerate(tenure_values):
        # just get the average, which is the first object
        v = per_irs_value[irs_values[0]][y][1]
        lineplot_tenure[j] = v

    lineplot_tenure2 = np.zeros(len(irs_values))
    for j,y in enumerate(tenure_values):
        # just get the average, which is the first object
        v = per_irs_value[irs_values[-1]][y][1]
        lineplot_tenure2[j] = v

    fig,ax = pplot.subplots()
    ax.set_title('Network densities for a term to maturity of 850')
    ax.set_ylabel('Network density')
    ax.set_xlabel('IRS value')
    ax.plot(irs_values, lineplot)
    #pplot.show()

    fig,ax = pplot.subplots()
    ax.set_title('Network densities for an IRS value of 1')
    ax.set_ylabel('Network density')
    ax.set_xlabel('Term to maturity')
    ax.plot(tenure_values, lineplot_tenure)
    pplot.show()

    exit()
    fig,ax = pplot.subplots()
    ax.set_title('Network densities for an IRS value of %d'%irs_values[-1])
    ax.set_ylabel('Network density')
    ax.set_xlabel('Term to maturity')
    ax.plot(tenure_values, lineplot_tenure2)

    fig,ax = pplot.subplots()
    ax.set_title('Network densities for a term to maturity of 850')
    ax.set_ylabel('Network density')
    ax.set_xlabel('IRS value')
    ax.plot(irs_values, lineplot_2irs)
    pplot.show()
