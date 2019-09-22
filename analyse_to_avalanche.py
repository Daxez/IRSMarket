"""
Plots some info on situations leading up to avalanches of a certain percentage of system size (configure in your run)
There are some hardcoded paths in here.
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
from matplotlib.cm import Reds, coolwarm
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.ndimage.filters import gaussian_filter1d

basepath = './simulation_data/large_avalanche_data'
def getconfig(aggid):
    file_path = '%s/%s/config.json'%(basepath,aggid)
    with open(file_path,'r') as fp:
        return json.load(fp)

def load_pickle(aggid,filename):
    file_path = '%s/%s/%s.bin'%(basepath,aggid,filename)
    with open(file_path,'rb') as fp:
        return pickle.load(fp)

def get_degrees(aggid):
    return load_pickle(aggid,'degrees')

def get_irss(aggid):
    file_path = '%s/%s/%s.bin'%(basepath,aggid,'no_irs')
    with open(file_path,'rb') as fp:
        no_irss = pickle.load(fp)
        irss_pb = pickle.load(fp)

    return no_irss, irss_pb

def get_balances(aggid):
    file_path = '%s/%s/%s.bin'%(basepath,aggid,'balances')
    with open(file_path,'rb') as fp:
        balances = pickle.load(fp)
        gross_balances = pickle.load(fp)

    return balances, gross_balances

def get_irs_changes(aggid):
    file_path = '%s/%s/%s.bin'%(basepath,aggid,'irs_data')
    with open(file_path,'rb') as fp:
        creations = pickle.load(fp)
        removals = pickle.load(fp)

    return creations, removals

def large(aggregate_id):
    config = getconfig(aggregate_id)
    irss = np.array(get_irss(aggregate_id))
    balances, gross_balances = get_balances(aggregate_id)
    degrees = np.array(get_degrees(aggregate_id))
    failed_bank_no = config['failed_bank']

    irs_val = config['model']['max_irs_value']
    threshold = config['model']['threshold']

    steps = range(len(degrees))

    bins = np.arange(-threshold-irs_val, threshold+irs_val, irs_val)
    im = np.zeros((len(balances),len(bins)-1))

    mi = int(np.max(irss))
    imirss = np.zeros((len(irss),mi))

    for i in steps:
        hist, balance_edges = np.histogram(balances[i],bins=bins)
        im[i] = hist
        hist, irs_edges = np.histogram(irss[i], bins=range(mi+1))
        imirss[i] = hist

    fig = pplot.figure()
    ax = fig.add_subplot(611)
    ax.set_title('Balance distribution')
    ax.imshow(im, interpolation='nearest', aspect='auto',vmin=0,vmax=500,
             cmap=Reds, extent=[bins[0],bins[-1],len(degrees),0])

    ax = fig.add_subplot(612)
    ax.set_title('Balance first failed bank')
    ax.plot(steps,np.array(balances)[:,failed_bank_no])
    ax.plot(steps,np.array(gross_balances)[:,failed_bank_no])

    ax = fig.add_subplot(613)
    ax.set_title('No irs and degree of failing banks and averages')
    ax.plot(steps,np.array(irss)[:,failed_bank_no])
    ax.plot(steps,np.array(degrees)[:,failed_bank_no])
    #ax.plot(steps,[np.mean(irss[x]) for x in steps])
    #ax.plot(steps,[np.mean(degrees[x]) for x in steps])

    ax = fig.add_subplot(614)
    ax.set_title('Distribution of Irss')
    ax.imshow(imirss, interpolation='nearest', aspect='auto',vmin=0,vmax=mi,cmap=Reds)

    ax = fig.add_subplot(615)
    ax.set_title('Distribution of balance at t=max')
    ax.bar(balance_edges[:-1],im[-1], width=irs_val)

    ax = fig.add_subplot(616)
    ax.set_title('Distribution of irss at t=max')
    ax.bar(irs_edges[:-1],imirss[-1])

    pplot.show()

def plot_balances_save(aggregate_id):
    config = getconfig(aggregate_id)
    irss = np.array(get_irss(aggregate_id))
    balances, gross_balances = get_balances(aggregate_id)
    degrees = np.array(get_degrees(aggregate_id))
    failed_bank_no = config['failed_bank']

    irs_val = config['model']['max_irs_value']
    threshold = config['model']['threshold']

    steps = range(len(degrees))
    bins = np.arange(-threshold-irs_val, threshold+irs_val, irs_val)
    im = np.zeros((len(balances),len(bins)-1))
    im_gross = np.zeros((len(balances),100))

    for i in steps:
        hist, balance_edges = np.histogram(balances[i],bins=bins)
        im[i] = hist
        hist, gross_edges = np.histogram(gross_balances[i], bins=100)
        im_gross[i] = hist

    fig = pplot.figure()

    ax = fig.add_subplot(211)
    ax.set_title('Distribution of balance at t=max')
    ax.bar(balance_edges[:-1],im[-1], width=irs_val)

    ax = fig.add_subplot(212)
    ax.set_title('Distribution of gross balance at t=max')
    ax.bar(gross_edges[:-1],im_gross[-1], width=irs_val)

    file_path = '%s/%s/%s.png'%(basepath,aggregate_id,'balances')
    pplot.savefig(file_path)

def balances_hist(aggregate_id):
    config = getconfig(aggregate_id)
    irss = np.array(get_irss(aggregate_id))
    balances, gross_balances = get_balances(aggregate_id)
    degrees = np.array(get_degrees(aggregate_id))
    failed_bank_no = config['failed_bank']

    irs_val = config['model']['max_irs_value']
    threshold = config['model']['threshold']

    steps = range(len(degrees))

    last_balances = sorted(np.array(balances)[-1])
    last_gross = sorted(np.array(gross_balances)[-1])

    lb_mu = np.mean(last_balances)
    lg_mu = np.mean(last_gross)

    lb_s = np.std(last_balances)
    lg_s = np.std(last_gross)

    print "Gross balances mu %.2f and sigma %.2f"%(lg_mu,lg_s)

    lb_fit = stats.norm.pdf(last_balances, lb_mu, lb_s)
    lg_fit = stats.norm.pdf(last_gross, lg_mu, lg_s)

    fig = pplot.figure()
    ax = fig.add_subplot(211)
    ax.plot(last_balances,lb_fit)
    ax.hist(last_balances,normed=True,bins=15)

    ax = fig.add_subplot(212)
    ax.plot(last_gross,lg_fit)
    ax.hist(last_gross,normed=True, bins=100)

    pplot.show()

def degree_distribution_non_failed_nodes():
    config = getconfig(aggregate_id)
    irss = np.array(get_irss(aggregate_id))
    balances, gross_balances = get_balances(aggregate_id)
    degrees = np.array(get_degrees(aggregate_id))
    failed_bank_no = config['failed_bank']
    #giant_component = load_pickle(aggregate_id, 'gc')
    defaulted_nodes = load_pickle(aggregate_id, 'defaulted')
    network = np.arrray(load_pickle(aggregate_id, 'network'))

    irs_val = config['model']['max_irs_value']
    threshold = config['model']['threshold']
    no_banks = config['model']['no_banks']

    steps = range(len(degrees))
    defnods = set(np.hstack(defaulted_nodes[:-1]))
    ndn = list(set(range(no_banks)) - defnods)
    print len(ndn)
    non_defaulted_degrees = np.array(degrees[-1])[ndn]

    fig = pplot.figure()
    #ax = fig.add_subplot(211)
    #ax.plot(steps,[len(c) for c in giant_component])

    ax = fig.add_subplot(111)
    ax.hist(non_defaulted_degrees,bins=50, normed=True)
    mu = np.mean(non_defaulted_degrees)
    std = np.std(non_defaulted_degrees)
    sndd = sorted(non_defaulted_degrees)
    ax.plot(sndd, stats.norm.pdf(sndd,mu,std),marker='o')

    pplot.show()

def balance_and_degree_scatters(aggregate_id):
    config = getconfig(aggregate_id)
    irss = np.array(get_irss(aggregate_id))
    balances, gross_balances = get_balances(aggregate_id)
    degrees = np.array(get_degrees(aggregate_id))
    failed_bank_no = config['failed_bank']
    #giant_component = load_pickle(aggregate_id, 'gc')
    defaulted_nodes = load_pickle(aggregate_id, 'defaulted')
    network = np.array(load_pickle(aggregate_id, 'network'))

    irs_val = config['model']['max_irs_value']
    threshold = config['model']['threshold']
    no_banks = config['model']['no_banks']

    steps = range(len(degrees))
    defnods = set(np.hstack(defaulted_nodes[:-1]))
    ndn = list(set(range(no_banks)) - defnods)
    print len(ndn)
    non_defaulted_degrees = np.array(degrees[-1])[ndn]

    degree_scatter_x = []
    degree_scatter_y = []

    gross_scatter_x = []
    gross_scatter_y = []

    net_scatter_x = []
    net_scatter_y = []

    sndn = sorted(ndn)

    print sndn

    for b in sndn:
        for b2 in sndn:
            if(b == b2): continue

            if(network[b,b2]>0):
                degree_scatter_x.append(degrees[-1][b])
                degree_scatter_y.append(degrees[-1][b2])

                gross_scatter_x.append(gross_balances[-1][b])
                gross_scatter_y.append(gross_balances[-1][b2])

                net_scatter_x.append(balances[-1][b])
                net_scatter_y.append(balances[-1][b2])


    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(degree_scatter_x,degree_scatter_y)
    ax.set_title('Degree scatter')
    pplot.show()

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(gross_scatter_x,gross_scatter_y)
    ax.set_title('Gross balance scatter')
    pplot.show()

    fig = pplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(net_scatter_x,net_scatter_y)
    ax.set_title('Net balance scatter')

    pplot.show()

def draw_graph(aggregate_id):
    config = getconfig(aggregate_id)
    irss = np.array(get_irss(aggregate_id)[0])
    balances, gross_balances = get_balances(aggregate_id)
    degrees = np.array(get_degrees(aggregate_id))
    failed_bank_no = config['failed_bank']
    #giant_component = load_pickle(aggregate_id, 'gc')
    defaulted_nodes = load_pickle(aggregate_id, 'defaulted')
    network = np.array(load_pickle(aggregate_id, 'network'))

    irs_val = config['model']['max_irs_value']
    threshold = config['model']['threshold']
    no_banks = config['model']['no_banks']

    steps = range(len(degrees))
    defnods = set(np.hstack(defaulted_nodes[:-1]))
    ndn = list(set(range(no_banks)) - defnods)
    print len(ndn)
    non_defaulted_degrees = np.array(degrees[-1])[ndn]


    mg = nx.DiGraph(network)

    min_gross = np.min(gross_balances)
    max_gross = np.max(gross_balances)
    ncmap = pplot.get_cmap('coolwarm')
    ncs = np.zeros(no_banks)
    for i in mg.nodes():
        ncs[i] = (gross_balances[-1][i]-min_gross)/(max_gross-min_gross)

    pos = graphviz_layout(mg, prog="fdp")
    nx.draw(mg, cmap=pplot.get_cmap('coolwarm'), vmin=0, vmax=1,node_color=ncs,pos=pos)
    pplot.show()

def plot_irs_changes(aggregate_id):
    config = getconfig(aggregate_id)
    steps = range(config['model']['no_steps'])
    degrees = get_degrees(aggregate_id)
    tenure = config['model']['max_tenure']

    creations, removals = get_irs_changes(aggregate_id)
    creations = np.array(creations)
    removals = np.array(removals)

    smoothed_creations = gaussian_filter1d(creations, tenure)
    smoothed_removals = gaussian_filter1d(removals, tenure)

    fig = pplot.figure()
    ax = fig.add_subplot(211)

    ax.plot(range(len(creations)), smoothed_creations, label="Creations (smoothed)")
    ax.plot(range(len(removals)), smoothed_removals, label="Removals (smoothed)")
    #pplot.plot(range(len(removals)),removals, label="Removals")
    #pplot.plot(range(len(removals)),creations-removals, label="Removals")

    pplot.legend()
    ax = fig.add_subplot(212)
    ax.plot(range(len(degrees)), [np.mean(d) for d in degrees])

    pplot.show()

def overall_balance_hist(aggregate_id):
    config = getconfig(aggregate_id)
    balances, gross_balances = get_balances(aggregate_id)

    fig = pplot.figure()
    ax = fig.add_subplot(411)

    ax.hist(np.hstack(balances), bins=30, normed=True)
    ax.set_title("Net Balance fequencies for whole run")

    ax = fig.add_subplot(412)

    ax.hist(np.hstack(gross_balances), bins=100, normed=True)
    ax.set_title("Gross Balance fequencies for whole run")


    ax = fig.add_subplot(413)

    ax.hist(np.hstack(balances[-10:]), bins=100, normed=True)
    ax.set_title("net Balance fequencies for last ten timesteps")

    ax = fig.add_subplot(414)

    ax.hist(np.hstack(gross_balances[-10:1]), bins=100, normed=True)
    ax.set_title("Gross Balance fequencies for last ten timesteps")

    pplot.show()

def irs_correlation(aggregate_id):
    config = getconfig(aggregate_id)
    no_irs, irs_pb = get_irss(aggregate_id)

    ipb = irs_pb[-1]

    dd = defaultdict(lambda: defaultdict(int))
    m = 0
    my = 0
    mx = 0
    for ipb in irs_pb:
        for (x,y) in ipb:
            if(x == 0 and y == 0): continue
            dd[x][y] += 1
            m = max(m,dd[x][y])
            my = max(my,y)
            mx = max(mx,x)


    img = np.zeros((mx+1,my+1))


    for x in dd:
        for y in dd[x]:
            img[x,y] = dd[x][y]/m

    cmp = pplot.get_cmap('Reds')
    pplot.figure()
    cax = pplot.imshow(img,interpolation='nearest',cmap=cmp, vmin=0, vmax=1,origin='lower')
    pplot.xlabel("Number of positive ends")
    pplot.ylabel("Number of negative ends")
    pplot.colorbar(cax)
    pplot.show()

def overall_degree_hist(aggregate_id):
    config = getconfig(aggregate_id)
    degrees = get_degrees(aggregate_id)

    fig = pplot.figure()
    ax = fig.add_subplot(211)

    ax.hist(np.hstack(degrees), bins=30, normed=True)
    ax.set_title("Degree fequencies for whole run")

    ax = fig.add_subplot(212)

    ax.hist(np.hstack(degrees[-10:]), bins=100, normed=True)
    ax.set_title("Degree fequencies for last ten timesteps")

    pplot.show()

if __name__ == '__main__':
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to Large avalanche analysis. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    if(len(sys.argv) == 2):
        aggregate_id = sys.argv[1]
    else:
        print ''
        print ''
        print "Welcome to Large avalanche analysis. Please provide ONE aggregateId to plot."
        print ''
        print ''
        exit()

    config = getconfig(aggregate_id)
    print "%d #nodes"%config['model']['no_banks']
    print "%d threshold"%config['model']['threshold']
    print "%d tenure"%config['model']['max_tenure']
    print "%d irs value"%config['model']['max_irs_value']

    #balance_and_degree_scatters(aggregate_id)
    #large(aggregate_id)
    #balances_hist(aggregate_id)
    #plot_irs_changes(aggregate_id)
    #overall_balance_hist(aggregate_id)
    #irs_correlation(aggregate_id)
    #overall_degree_hist(aggregate_id)
    draw_graph(aggregate_id)
