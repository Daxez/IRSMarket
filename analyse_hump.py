"""
methods for calculating the weight in the hump
"""
from __future__ import division
import sys

from collections import defaultdict

import numpy as np
import powerlaw
import matplotlib.pyplot as pplot
from scipy.ndimage import gaussian_filter1d

from data.models import *

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

    data = np.zeros(tot_s)
    dcnt = 0
    for i in xrange(len(freqs)):
        for j in xrange(freqs[i][1]):
            data[dcnt] = freqs[i][0]
            dcnt += 1

    x = np.array([d[0] for d in freqs])
    y = np.array([d[1] / tot_s for d in freqs])

    return x, y, run, data

def estimate_pl_part(x,y,sigma=2,gap_tol=10):
    yg = gaussian_filter1d(y,sigma,mode='nearest')

    # Gap approach
    xdif = [x[i+1]-x[i] for i in range(len(x)-1)]
    mgap = max(xdif)
    xind = len(x)
    if mgap > gap_tol:
        xind = xdif.index(mgap)+1

    # Diff approach
    ydif = [yg[i]-yg[i-1] for i in range(1,len(y))]
    yind = len(yg)
    for i in range(1,len(ydif)):
        if ydif[i-1] < 0 and ydif[i] > 0:
            yind = i-1
            break

    ind = min([xind,yind])
    #print 'Xindex: %d, Yindex: %d'%(xind,yind)

    return x[:ind],y[:ind], ind, yg

def get_hump_error(x,y,raw,add_x=False):

    # estimate power law part
    x_pl,y_pl,pl_ind, yg = estimate_pl_part(x,y)

    # fit
    max_x_pl = max(x_pl)

    #print "Max X for power law: %d"%max_x_pl

    raw_pl = raw[np.where(raw <= max_x_pl)]
    f = powerlaw.Fit(raw_pl,discrete=True,xmin=1,xmax=max_x_pl)
    alpha = f.power_law.alpha

    # Get graph data
    ye = [pow(xi,-alpha) for xi in x_pl]

    humpx = x[pl_ind:]
    norm_y_error = [0]
    if len(humpx) > 0:
        py = np.array([pow(xi,-alpha) for xi in humpx])
        norm_y_error = (y[pl_ind:] - py)/py

    if(add_x):
        return sum(norm_y_error), alpha, max_x_pl
    return (len(humpx)**2)*sum(norm_y_error), alpha

def get_hump_ks_error(x,y,raw,N,add_x=False):

    # estimate power law part
    x_pl,y_pl,pl_ind, yg = estimate_pl_part(x,y)

    # fit
    max_x_pl = max(x_pl)

    #print "Max X for power law: %d"%max_x_pl

    raw_pl = raw[np.where(raw <= max_x_pl)]
    f = powerlaw.Fit(raw_pl,discrete=True,xmin=1,xmax=max_x_pl)
    alpha = f.power_law.alpha

    #dist = powerlaw.Power_Law(xmin=1, xmax=max(x), parameters=[alpha], discrete=True)
    D = powerlaw.power_law_ks_distance(raw, alpha, 1, N,  True, False)
    print D - f.D

    # Get graph data
    ye = [pow(xi,-alpha) for xi in x_pl]

    humpx = x[pl_ind:]
    norm_y_error = [0]
    approx_error = D#dist.KS(raw)
    #if len(humpx) > 0:
    #    py = np.array([pow(xi,-alpha) for xi in humpx])
    #    norm_y_error = abs(1-(y[pl_ind:]/py))
    #    approx_error = sum(norm_y_error)/len(humpx)

    if(add_x):
        return approx_error, alpha, max_x_pl
    return approx_error, alpha

def get_hump_cumulative_value(x,y):
    # estimate power law part
    x_pl,y_pl,pl_ind, yg = estimate_pl_part(x,y)

    # fit
    max_x_pl = max(x_pl)
    return sum(y[np.where(x > max_x_pl)])


def get_hump_info(aggregate_id, add_x=False):
    x,y,r,raw = get_aggregate_dist(aggregate_id)
    return get_hump_error(x,y,raw, add_x)


if __name__ == '__main__':
    if(len(sys.argv) == 1):
        print ''
        print ''
        print "Welcome to JustShowMeThaHump. Please provide an aggregateId to plot."
        print ''
        print ''
        exit()
    if(len(sys.argv) == 2):
        aggregate_ids = [sys.argv[1]]
    else:
        aggregate_ids = sys.argv[1:]

    mplm_inp = raw_input('Want to manually set the max x for power law?[y/N]')
    manual_pl_max = mplm_inp.lower() == 'y'

    if not manual_pl_max:
        # Get the figure
        fig = pplot.figure()
        ax = fig.add_subplot(111)

        # Set axes scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # colormap for different plots
        cmap = pplot.get_cmap('YlOrRd')
        n = len(aggregate_ids)


        for (i,aggregate_id) in enumerate(aggregate_ids):
            x,y,run,data = get_aggregate_dist(aggregate_id)
            err,alpha = get_hump_error(x,y,data)
            print "Error: %f, alpha: %f"%(err,alpha)

            strpol = (run.no_banks,run.max_tenure,run.max_irs_value,run.threshold)
            l = "%d banks, %d tenure, %d IRS Value, %d threshold"%strpol
            ax.scatter(x,y,color=cmap((i+1)/n),label=l)
            alpha_label = 'Alpha: %f'%alpha
            ax.plot(x,[pow(xi,-alpha) for xi in x],c=cmap((i+1)/n),label=alpha_label)

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         box.width*0.75, box.height])

        # Put a legend below current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  fancybox=True, shadow=True, ncol=1)

        pplot.show()
    else:
        agg_dict = {}
        for aggregate_id in aggregate_ids:
            x,y,run,data = get_aggregate_dist(aggregate_id)

            # Get the figure
            fig = pplot.figure()
            ax = fig.add_subplot(111)

            # Set axes scales
            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.scatter(x,y)
            pplot.show()

            max_x = int(raw_input('Give me max x for the power law. [Number]\n'))
            print max_x

            raw_pl = data[np.where(data <= max_x)]
            f = powerlaw.Fit(raw_pl,discrete=True,xmin=1,xmax=max_x)
            alpha = f.power_law.alpha

            agg_dict[aggregate_id] = (x,y,alpha,run)
            del data

        fig = pplot.figure()
        ax = fig.add_subplot(111)

        # Set axes scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # colormap for different plots
        cmap = pplot.get_cmap('YlOrRd')
        n = len(aggregate_ids)

        for i,(x,y,alpha,run) in enumerate(agg_dict.values()):
            strpol = (run.no_banks,run.max_tenure,run.max_irs_value,run.threshold)
            l = "%d banks, %d tenure, %d IRS Value, %d threshold"%strpol
            ax.scatter(x,y,color=cmap((i+1)/n),label=l)
            alpha_label = 'Alpha: %f'%alpha
            ax.plot(x,[pow(xi,-alpha) for xi in x],c=cmap((i+1)/n),label=alpha_label)

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         box.width*0.75, box.height])

        # Put a legend below current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  fancybox=True, shadow=True, ncol=1)

        pplot.show()
