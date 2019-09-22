"""
Comparing distributions (powerlaw, lognormal and exponential).
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as pplot
import powerlaw

from analyse_hump import *
from data.models import get_session

if __name__ == '__main__':
    ses = get_session()
    aggs = ses.execute('SELECT DISTINCT aggregate_id FROM run WHERE aggregate_id = \'452b894a-2aff-4d36-a58c-b15bb0219d7a\'').fetchall()
    ses.close()

    pvalues_lognormal = np.zeros(len(aggs))
    Rvalues_lognormal = np.zeros(len(aggs))

    pvalues_lognormal_pos = np.zeros(len(aggs))
    Rvalues_lognormal_pos = np.zeros(len(aggs))

    pvalues_exponential= np.zeros(len(aggs))
    Rvalues_exponential= np.zeros(len(aggs))

    for i,res in enumerate(aggs):
        aggregate_id = res[0]

        x,y,run,raw = get_aggregate_dist(aggregate_id)

        # estimate power law part
        x_pl,y_pl,pl_ind, yg = estimate_pl_part(x,y)
        print get_hump_error(x,y,raw)

        # fit
        max_x_pl = max(x_pl)
        raw_pl = raw[np.where(raw <= max_x_pl)]
        print raw_pl
        f = powerlaw.Fit(raw_pl,discrete=True,xmin=1,xmax=max_x_pl)
        ax = f.plot_pdf()
        ax.scatter(x, y)
        ax.set_xlim(1e-1,1e2)

        pplot.plot()

        R,p = f.distribution_compare('power_law','lognormal')
        pvalues_lognormal[i] = p
        Rvalues_lognormal[i] = R

        R,p = f.distribution_compare('power_law','lognormal_positive')
        pvalues_lognormal_pos[i] = p
        Rvalues_lognormal_pos[i] = R

        R,p = f.distribution_compare('power_law','exponential')
        pvalues_exponential[i] = p
        Rvalues_exponential[i] = R


    x = range(len(aggs))

    figure = pplot.figure()

    ax = figure.add_subplot(311)
    ax.plot(x,Rvalues_lognormal)
    ax.set_ylabel('R for lognormal')
    ax.set_xlabel('Runs')
    ax2 = ax.twinx()
    ax2.plot(x,pvalues_lognormal,c='r')
    ax2.set_ylabel('p-value',color='r')


    ax = figure.add_subplot(312)
    ax.plot(x,Rvalues_lognormal_pos)
    ax.set_ylabel('R for lognormal positive')
    ax.set_xlabel('Runs')
    ax2 = ax.twinx()
    ax2.plot(x,pvalues_lognormal_pos,c='r')
    ax2.set_ylabel('p-value',color='r')


    ax = figure.add_subplot(313)
    ax.plot(x,Rvalues_exponential)
    ax.set_ylabel('R for Exponential')
    ax.set_xlabel('Runs')
    ax2 = ax.twinx()
    ax2.plot(x,pvalues_exponential,c='r')
    ax2.set_ylabel('p-value',color='r')

    pplot.show()
