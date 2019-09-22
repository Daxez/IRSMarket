"""File for generating heatmaps
Started off as something generic, but copying became easier at a certain point.
"""
from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import desc

from data.models import *


def get_runs(aggregate_id,session):
    return session.query(RunModel)\
                  .filter(RunModel.aggregate_id == aggregate_id)\
                  .all()

def get_defaults(run_id,session):
    return session.query(DefaultModel)\
                  .filter(DefaultModel.run_id == run_id)\
                  .all()

def get_largest_default(run_id,session):
    return session.query(DefaultModel)\
                  .filter(DefaultModel.run_id == run_id)\
                  .order_by(desc(DefaultModel.size))\
                  .first()

possible_columns = ['sigma','max_irs_value','irs_threshold','max_tenure']

def get_columns(runs):
    x = None
    xs = None
    y = None
    ys = None

    for c in possible_columns:
        colset = set([getattr(r,c) for r in runs])
        if len(colset) > 1:
            if( x == None):
                x = c
                xs = list(colset)
            else:
                y = c
                ys = list(colset)
                break

    if(x == None or y == None):
        raise "No sweep aggregate"

    return (x,xs,y,ys)

def heat_map(data,title,x,y,xset,yset,cbset,path,cols,showplot=False):
    fig, ax = plt.subplots()

    cax = ax.imshow(data, cmap=plt.cm.Oranges, interpolation='nearest')

    ax.set_title(title)

    ax.set_xticks(np.arange(len(yset)), minor=False)
    ax.set_yticks(np.arange(len(xset)), minor=False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticklabels(yset, minor=False)
    ax.set_yticklabels(xset, minor=False)
    ax.set_xlabel(y)
    ax.set_ylabel(x)

    cb = fig.colorbar(cax, ticks=cbset)
    #cb.set_ticks([t/cbset[-1] for t in cbset])
    #cb.set_ticklabels(cbset)

    save_addition = title+"_"+cols
    if showplot:
        plt.show()
    else:
        plt.savefig(path%save_addition)
    plt.close()

if __name__ == '__main__':
    root_path = './sweep/HeatMaps/'
    types = ['','Default','Shuffle','Sorted']
    session = get_session()
    aggregate_ids = ['08fad362-a240-4c7e-a985-9b1e4bd84fb1',
                     '0ea58a50-01a5-4ff5-b327-4d447b81fc29',
                     '200ef28d-429e-49dc-9183-898c0a815ffb',
                     '3c71f8a5-dd59-45b8-af2b-fb5a4446fb32',
                     '499b6a98-a996-494b-9c75-03963d523602',
                     '561106f1-1c4a-4cd1-892f-1bb734075182',
                     '925cf690-c25b-48ca-b7f8-245abbd8943a',
                     'ba4ab7ec-055c-4c46-ba6a-1d28f0fabe4c',
                     'bcf2db62-e775-4f9c-a119-281841aeccba',
                     'cd783fb7-a1df-4828-8e7e-deea83ca879c',
                     'ddff966f-f4e9-412c-89cc-718d0abf2ba8',
                     'edb32939-fa55-4724-a53f-0ae627f5029d']

    for aggregate_id in aggregate_ids:
        print 'Running heatmap for aggregate %s'%aggregate_id

        runs = get_runs(aggregate_id,session)

        (x,xset,y,yset) = get_columns(runs)

        hm_default = np.zeros((len(xset),len(yset)))
        hm_noswaps = np.zeros((len(xset),len(yset)))
        max_swapno = max(runs,key=lambda x: x.swaps).swaps

        for run in runs:
            xv = xset.index(getattr(run,x))
            yv = yset.index(getattr(run,y))
            largest_default = get_largest_default(run.run_id,session)
            if largest_default != None:
                hm_default[xv,yv] = (largest_default.size/run.no_banks)
            else:
                hm_default[xv,yv] = 0

            hm_noswaps[xv,yv] = run.swaps/max_swapno

        print x,y
        print xset,yset
        print hm_default

        path = root_path+types[runs[0].market_type]+'/%s.png'

        heat_map(hm_default,"Defaults",x,y,xset,yset,range(20),path,x+'__'+y)
        heat_map(hm_noswaps,"Number_of_swaps",x,y,xset,yset,[0,0.5*max_swapno,max_swapno],path,x+'__'+y)
