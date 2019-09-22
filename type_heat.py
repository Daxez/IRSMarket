"""
Heatmap of the distribution types.
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs

from data.models import get_session
from analyse_hump import get_hump_error, get_aggregate_dist

def get_aggregate_id(b,t,irst,ten):
    print b,t,irst,ten
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

def show_irs_vs_tenure(irs_threshold,tenure,no_banks,threshold):
    a = raw_input("Do you want to see IRS value vs tenure?[Y/n]")
    if(a != 'n'):
        for t in threshold:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(tenure)))
                for i,irst in enumerate(irs_threshold):
                    for j,ten in enumerate(tenure):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        tpe = get_type(agg_id)
                        im_data[i,j] = tpe

                #im = get_color_image(im_data)

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Term to maturity")
                ax.set_xticklabels([0,50,100,250,400,550,700,850])
                ax.set_title("Heatmap for %d banks, term to maturity %d"%(b,t))

                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                    vmin=1, vmax=5)

                cb = fig.colorbar(cax)
                cb.set_ticks([1,2,3,4,5])
                cb.set_ticklabels([1,2,3,4,5])
                cb.set_label('Distribution type', rotation=270, labelpad=15)

                path = '/home/..../typeimages/type_irs_tenure_%d_banks_%d_threshold.png'%(b,t)
                fig.savefig(path,bbox_inches='tight')
                pplot.close()


if __name__ == '__main__':

    irs_threshold = np.arange(1,9)
    tenure = [50,100,250,400,550,700,850]
    no_banks = [100,200]#np.arange(100,600,100)
    threshold = np.arange(10,35,5)

    show_irs_vs_tenure(irs_threshold,tenure,no_banks,threshold)
