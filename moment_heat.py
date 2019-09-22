"""
Heatmap of the moments for different configurations
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs
import scipy.stats as stats

from data.models import *
from analyse_hump import get_hump_error, get_aggregate_dist

def get_aggregate_id(b,t,irst,ten):
    query = """SELECT aggregate_id
               FROM run
               WHERE threshold = :threshold and no_banks = :no_banks
                     and max_irs_value = :irs_val and max_tenure = :ten
               LIMIT 1"""
    session = get_session()
    agg_id = session.execute(query,{'threshold':t,
                           'no_banks':b,
                           'irs_val':irst,
                           'ten':ten}).first()[0]
    session.close()
    return agg_id

def get_type(aggregate_id):
    query = """SELECT type_id FROM aggregate_type WHERE aggregate_id = :agg """

    session = get_session()
    t = session.execute(query,{'agg':aggregate_id}).first()[0]
    session.close()
    return t

def show_irs_vs_tenure(irs_threshold,tenure,no_banks,threshold,moment):
    a = raw_input("Do you want to see IRS value vs tenure?[Y/n]")
    if(a != 'n'):
        for t in threshold:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(tenure)))
                for i,irst in enumerate(irs_threshold):
                    for j,ten in enumerate(tenure):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        x,y,r,raw = get_aggregate_dist(agg_id)
                        m = stats.moment(raw,moment)
                        im_data[i,j] = np.log(m)

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Tenure")
                ax.set_xticklabels([0,50,100,250,400,550,700,850])
                ax.set_title("Heatmap for %d banks, threshold %d"%(b,t))

                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                vmin=np.min(im_data),vmax=np.max(im_data),cmap=clrs.Reds)
                cb = fig.colorbar(cax)
                pplot.show()

                try:
                    a = raw_input("Do you want to continue [Y/n]")
                    if(a == 'n'):
                        return
                except KeyboardInterrupt:
                    exit(0)

def show_irs_vs_threshold(irs_threshold,tenure,no_banks,threshold,moment):
    a = raw_input("Do you want to see IRS value vs threshold?[Y/n]")
    if(a != 'n'):
        for ten in tenure:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(threshold)))
                for i,irst in enumerate(irs_threshold):
                    for j,t in enumerate(threshold):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        x,y,r,raw = get_aggregate_dist(agg_id)
                        m = stats.moment(raw,moment)
                        im_data[i,j] = np.log(m)

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Threshold")
                ax.set_xticklabels([0,10,15,20,25,30])
                ax.set_title("Heatmap for %d banks, tenure %d"%(b,ten))

                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                vmin=np.min(im_data),vmax=np.max(im_data),cmap=clrs.Reds)
                cb = fig.colorbar(cax, ticks=[np.min(im_data), np.max(im_data)])
                pplot.show()


                try:
                    a = raw_input("Do you want to continue [Y/n]")
                    if(a == 'n'):
                        return
                except KeyboardInterrupt:
                    exit(0)

if __name__ == '__main__':

    irs_threshold = np.arange(1,9)
    tenure = [50,100,250,400,550,700,850]
    no_banks = [100,200]#np.arange(100,600,100)
    threshold = np.arange(10,35,5)
    moment = 2

    show_irs_vs_tenure(irs_threshold,tenure,no_banks,threshold,moment)
    show_irs_vs_threshold(irs_threshold,tenure,no_banks,threshold,moment)
