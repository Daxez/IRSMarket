"""
Another powerlaw exponent heatmap (I think alpha heat is more recent.)
"""

from __future__ import division

from itertools import product
import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs
from sklearn import svm

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

def run_svm(threshold, no_banks,irs_threshold,tenure):
    for t in threshold:
        for b in no_banks:

            points = np.dstack(np.meshgrid(irs_threshold, tenure)).reshape(-1, 2)
            alpha_class = np.zeros(len(points))
            for i,x in enumerate(points):
                irst = x[0]
                ten = x[1]
                agg_id = get_aggregate_id(b,t,irst,ten)
                x,y,r,raw = get_aggregate_dist(agg_id)
                _,alp = get_hump_error(x,y,raw)

                if alp <= 3:
                    alpha_class[i] = 1
                else:
                    alpha_class[i] = 0
            svc = svm.SVC(kernel='poly').fit(points,alpha_class)

            xx, yy = np.meshgrid(np.arange(min(irs_threshold),max(irs_threshold), 0.02),
                                 np.arange(min(tenure),max(tenure), 0.02))

            Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            pplot.figure()
            pplot.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
            pplot.scatter(points[:,0],points[:,1], c=alpha_class)

            pplot.show()




if __name__ == '__main__':

    irs_threshold = np.arange(1,9)
    tenure = [50,100,250,400,550,700,850]
    no_banks = [100, 200]#np.arange(100,600,100)
    threshold = np.arange(10,35,5)

    a = raw_input("Do you want to see IRS value vs tenure?[Y/n]")
    if(a != 'n'):
        for t in threshold:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(tenure)))
                for i,irst in enumerate(irs_threshold):
                    for j,ten in enumerate(tenure):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        x,y,r,raw = get_aggregate_dist(agg_id)
                        err,alp = get_hump_error(x,y,raw)
                        im_data[i,j] = alp

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Tenure")
                ax.set_xticklabels([0,50,100,250,400,550,700,850])
                ax.set_title("Heatmap for %d banks, threshold %d:Alpha"%(b,t))

                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                    vmin=np.min(im_data),vmax=np.max(im_data))

                pplot.colorbar(cax)
                pplot.show()

                try:
                    a = raw_input("Do you want to continue [Y/n]")
                    if(a == 'n'):
                        exit(0)
                except KeyboardInterrupt:
                    exit(0)

    a = raw_input("Do you want to see IRS value vs threshold?[Y/n]")
    if(a != 'n'):
        for ten in tenure:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(threshold)))
                for i,irst in enumerate(irs_threshold):
                    for j,t in enumerate(threshold):
                        agg_id = get_aggregate_id(b,t,irst,ten)
                        x,y,r,raw = get_aggregate_dist(agg_id)
                        err,alp = get_hump_error(x,y,raw)
                        im_data[i,j] = alp

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(0,9))
                ax.set_xlabel("Threshold")
                ax.set_xticklabels([0,10,15,20,25,30])
                ax.set_title("Heatmap for %d banks, tenure %d: Alpha"%(b,ten))


                cax = ax.imshow(im_data,origin='lower',interpolation='nearest',
                    vmin=np.min(im_data),vmax=np.max(im_data))

                pplot.colorbar(cax)
                pplot.show()

                try:
                    a = raw_input("Do you want to continue [Y/n]")
                    if(a == 'n'):
                        exit(0)
                except KeyboardInterrupt:
                    exit(0)
