"""Old file to create a scatter plot of distribution types"""
from __future__ import division

import math
import pickle
import time

import numpy as np
import matplotlib.pyplot as pplot

from sqlalchemy import func
from sqlalchemy.orm import load_only
from data.models import *

types = [1,2,3,4,5]
markers = ['o','v','^','>','<']

if __name__ == '__main__':

    session = get_session()
    runs = session.query(RunModel).all()
    agg_types = session.query(AggregateType).all()

    session.close()

    fig = pplot.figure()
    fig.canvas.set_window_title('Type for Tenure and Dissipation for 500, 100 and 50 nodes')
    ax = fig.add_subplot(311)
    ax.set_title('Scatter')

    for t in types:
        pnts_ids = [a.aggregate_id for a in agg_types if a.type_id == t]
        pnts = [a for a in runs if a.aggregate_id in pnts_ids and a.no_banks == 500]
        x = [a.max_tenure for a in pnts]
        y = [a.dissipation for a in pnts]

        print x,y,len(x),len(y)
        lbl = 'Type %d'%t
        ax.scatter(x,y, marker=markers[t-1],label=lbl)

    pplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    ax = fig.add_subplot(312)
    for t in types:
        pnts_ids = [a.aggregate_id for a in agg_types if a.type_id == t]
        pnts = [a for a in runs if a.aggregate_id in pnts_ids and a.no_banks == 100]
        x = [a.max_tenure for a in pnts]
        y = [a.dissipation for a in pnts]

        print x,y,len(x),len(y)
        lbl = 'Type %d'%t
        ax.scatter(x,y, marker=markers[t-1],label=lbl)

    ax = fig.add_subplot(313)
    for t in types:
        pnts_ids = [a.aggregate_id for a in agg_types if a.type_id == t]
        pnts = [a for a in runs if a.aggregate_id in pnts_ids and a.no_banks == 50]
        x = [a.max_tenure for a in pnts]
        y = [a.dissipation for a in pnts]

        print x,y,len(x),len(y)
        lbl = 'Type %d'%t
        ax.scatter(x,y, marker=markers[t-1],label=lbl)

    pplot.show()
