"""
Pretty usefull thing to plot distributions based parameter sets
"""
from __future__ import division

import argparse
import sys
import numpy as np
import matplotlib.pyplot as pplot

from data.models import *

def get_agg_ids(no_banks, threshold, irs_value, tenure):
    session = get_session()
    query = """SELECT DISTINCT aggregate_id, no_banks, threshold, max_tenure, max_irs_value
               FROM run"""
    firstDone = False
    orderby = 'ORDER BY '
    if(no_banks):
        query = '%s WHERE no_banks = %d'%(query,no_banks)
        firstDone = True

    if(threshold):
        s_and =  'AND' if firstDone else 'WHERE'
        query = '%s %s threshold = %d'%(query, s_and, threshold)
        firstDone = True

    if(irs_value):
        s_and =  'AND' if firstDone else 'WHERE'
        query = '%s %s max_irs_value = %d'%(query, s_and, irs_value)
        firstDone = True

    if(tenure):
        s_and =  'AND' if firstDone else 'WHERE'
        query = '%s %s max_tenure = %d'%(query, s_and, tenure)

    query = '%s ORDER BY no_banks, threshold, max_tenure, max_irs_value'%query
    print query
    res = session.execute(query).fetchall()
    session.close()
    return [r[0] for r in res]

def get_aggregate_dist(aggregate_id):

    session = get_session()
    run = session.query(RunModel).filter(RunModel.aggregate_id == aggregate_id).first()

    if(run == None):
        print "Could not find a run with aggregate_Id %s, now exiting"%aggregate_id
        exit(0)

    freqs = session.execute("""SELECT size, frequency
                               FROM default_aggregate
                               WHERE aggregate_id = :aggregate_id
                               ORDER BY size""",
                                  {'aggregate_id':aggregate_id}).fetchall()

    session.close()

    tot_s = sum([d[1] for d in freqs])
    x = np.array([d[0] for d in freqs])
    y = np.array([d[1] / tot_s for d in freqs])

    return x, y, run

def show_single(aggregate_id):
    x,y,run = get_aggregate_dist(aggregate_id)

    fig = pplot.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.scatter(x,y)

    pplot.show()

def show_mulitple(aggregate_ids):
    fig = pplot.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('P(S)')
    ax.set_xlabel('S')
    cmap = pplot.get_cmap('YlOrRd')
    n = len(aggregate_ids)

    for (i,aggregate_id) in enumerate(aggregate_ids):
        x,y,run = get_aggregate_dist(aggregate_id)
        strpol = (run.no_banks,run.max_tenure,run.max_irs_value,run.threshold)
        l = "%d banks, %d tenure, %d IRS Value, %d threshold"%strpol
        ax.scatter(x/run.no_banks,y, color=cmap((i+1)/n),label=l)
        #ax.scatter(x/run.no_banks,y, color=cmap((i+1)/n),label=l)
        # Shrink current axis's height by 10% on the bottom
        #ax.plot(x,y, color=cmap((i+1)/n),label=l)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width*0.75, box.height])

    # Put a legend below current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fancybox=True, shadow=True, ncol=1)

    pplot.show()


if __name__ == '__main__':
    print ''
    print ''
    print "Welcome to justshowme for params."
    print ''
    print ''

    parser = argparse.ArgumentParser()
    parser.add_argument('-N',help="Number of banks",type=int, metavar="Number")
    parser.add_argument('-i',help="Maximum value of an irs",type=int,metavar="Value")
    parser.add_argument('-te',help="Maximum tenure time",type=int,metavar="Tenure")
    parser.add_argument('-t',help="Maximum balance threshold",type=int,metavar="Threshold")

    args = parser.parse_args()

    no_banks = None
    irs_value = None
    tenure = None
    threshold = None

    print 'ARGS'
    print args
    if(args.N): no_banks = args.N
    if(args.i): irs_value = args.i
    if(args.te): tenure = args.te
    if(args.t): threshold = args.t

    aggids = get_agg_ids(no_banks, threshold, irs_value, tenure)

    if(len(aggids) > 0):
        show_mulitple(aggids)
    else:
        show_single(aggids)
