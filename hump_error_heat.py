"""
Generating heatmaps for the hump error.
"""
from __future__ import division
import sys

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs

from data.models import *
from analyse_hump import get_hump_error, get_aggregate_dist, get_hump_ks_error, get_hump_cumulative_value

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

def get_color(error,mx):
    return clrs.get_cmap('Reds')(error/mx)
    if error < 20000:
        e = (error-100)/(20000-100)
        return clrs.get_cmap('Greens')(e)
    elif error < 150000:
        e = (error-20000)/(150000-20000)
        return clrs.get_cmap('Blues')(e)
    else:
        e = (error-150000)/(mx-150000)
        return clrs.get_cmap('Reds')(e)

def get_color_image(im):
    xlen,ylen = im.shape
    cim = np.ones((xlen,ylen,4))
    mx = np.max(im)
    mn = np.min(im)

    for i in xrange(xlen):
        for j in xrange(ylen):
            if(im[i,j] < 100): continue
            cim[i,j] = get_color(im[i,j],mx)
    return cim

def normalize_data(im):
    for i in range(3):
        rng = im[im[:,:,i]>1]
        if(len(rng)>0):
            mx = np.max(rng)
            mn = np.min(rng)
            rng = (rng-mn)/(mx-mn)
            im[im[:,:,i]>1] = rng

    return im

def show_irs_vs_tenure(irs_threshold,tenure,no_banks,threshold,autocontinue=False):
    if not autocontinue:
        a = raw_input("Do you want to see IRS value vs tenure?[Y/n]")
    if(autocontinue or a != 'n'):
        for t in threshold:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(tenure)))
                for i,irst in enumerate(irs_threshold):
                    for j,ten in enumerate(tenure):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        x,y,r,raw = get_aggregate_dist(agg_id)
                        #err,alp = get_hump_ks_error(x,y,raw,b)
                        err = get_hump_cumulative_value(x,y)
                        im_data[i,j] = err

                #im = get_color_image(im_data)

                min_value = np.min(im_data)
                max_value = 0.05#np.max(im_data)

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Tenure")
                ax.set_xticklabels([0,50,100,250,400,550,700,850])
                ax.set_title("Heatmap for %d banks, threshold %d"%(b,t))

                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                    vmin=min_value,vmax=max_value)

                cb = fig.colorbar(cax)
                cb.set_ticks([min_value, max_value])
                cb.set_ticklabels([min_value, max_value])
                cb.set_label('P(x > Last powerlaw point)', rotation=270)

                #pplot.show()
                filename = 'irs_ten_%d_banks_%d_threshold.png'%(b,t)
                path = '/home/..../Programming/GitRepos/scriptie/figs/heatmaps/'
                fl = '%s%s'%(path,filename)
                pplot.savefig(fl,bbox_inches='tight')
                pplot.close(fig)

                if not autocontinue:
                    try:
                        a = raw_input("Do you want to continue [Y/n]")
                        if(a == 'n'):
                            return
                    except KeyboardInterrupt:
                        exit(0)

def show_irs_vs_threshold(irs_threshold,tenure,no_banks,threshold,autocontinue = False):
    if not autocontinue:
        a = raw_input("Do you want to see IRS value vs threshold?[Y/n]")
    if(autocontinue or a != 'n'):
        for ten in tenure:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(threshold)))
                for i,irst in enumerate(irs_threshold):
                    for j,t in enumerate(threshold):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        x,y,r,raw = get_aggregate_dist(agg_id)
                        #err,alp = get_hump_ks_error(x,y,raw,b)
                        err = get_hump_cumulative_value(x,y)
                        im_data[i,j] = err

                #im = get_color_image(im_data)
                min_value = np.min(im_data)
                max_value = 0.05#np.max(im_data)

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Threshold")
                ax.set_xticklabels([0,10,15,20,25,30])
                ax.set_title("Heatmap for %d banks, tenure %d"%(b,ten))

                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                    vmin=min_value,vmax=max_value)
                cb = fig.colorbar(cax)
                cb.set_ticks([min_value, max_value])
                cb.set_ticklabels([min_value, max_value])
                cb.set_label('P(x > Last powerlaw point)', rotation=270)

                filename = 'irs_thresh_%d_banks_%d_tenure.png'%(b,ten)
                path = '/home/..../Programming/GitRepos/scriptie/figs/heatmaps/'
                fl = '%s%s'%(path,filename)
                pplot.savefig(fl,bbox_inches='tight')

                if not autocontinue:
                    try:
                        a = raw_input("Do you want to continue [Y/n]")
                        if(a == 'n'):
                            return
                    except KeyboardInterrupt:
                        exit(0)

if __name__ == '__main__':
    autocontinue = False

    if(len(sys.argv) == 2):
        autocontinue = sys.argv[1] == '-a'

    irs_threshold = np.arange(1,9)
    tenure = [50,100,250,400,550,700,850]
    no_banks = [100,200]#np.arange(100,600,100)
    threshold = np.arange(10,35,5)

    show_irs_vs_tenure(irs_threshold,tenure,no_banks,threshold, autocontinue)
    show_irs_vs_threshold(irs_threshold,tenure,no_banks,threshold, autocontinue)
