"""
Attempt to collapse some data
"""
from __future__ import division

import sys
import numpy as np
import matplotlib.pyplot as pplot
from scipy.ndimage import gaussian_filter1d
from analyse_hump import get_hump_info,get_aggregate_dist,get_hump_error

#def get_aggregate_dist(aggregate_id):
#
#    session = get_session()
#    run = session.query(RunModel).filter(RunModel.aggregate_id == aggregate_id).first()
#
#    if(run == None):
#        print "Could not find a run with aggregate_Id %s, now exiting"%aggregate_id
#        exit(0)
#
#    freqs = session.execute("""SELECT size, frequency
#                               FROM default_aggregate
#                               WHERE aggregate_id = :aggregate_id
#                               ORDER BY size""",
#                                  {'aggregate_id':aggregate_id}).fetchall()
#
#    session.close()
#
#    tot_s = sum([d[1] for d in freqs])
#    x = np.array([d[0] for d in freqs])
#    y = np.array([d[1] / tot_s for d in freqs])
#
#    return x, y, run

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
    cmap = pplot.get_cmap('YlOrRd')
    n = len(aggregate_ids)

    for (i,aggregate_id) in enumerate(aggregate_ids):
        x,y,run,raw = get_aggregate_dist(aggregate_id)
        strpol = (run.no_banks,run.max_tenure,run.max_irs_value,run.threshold)
        l = "%d banks, %d tenure, %d IRS Value, %d threshold"%strpol

        err,alpha = get_hump_info(aggregate_id)

        xn = x#/run.no_banks
        yn = gaussian_filter1d([yi*pow(xi,alpha) for xi,yi in zip(x,y)],4)
        ax.scatter(xn,yn, color=cmap((i+1)/n),label=l)

        # Shrink current axis's height by 10% on the bottom

    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width*0.75, box.height])

    # Put a legend below current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fancybox=True, shadow=True, ncol=1)

    pplot.show()


if __name__ == '__main__':
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to JustShowMe. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    if(len(sys.argv) == 2):
        aggregate_id = sys.argv[1]
        show_single(aggregate_id)
    else:
        aggregate_ids = sys.argv[1:]
        show_mulitple(aggregate_ids)
