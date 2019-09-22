"""Really specific thing for plotting some approximations of the distribution?"""
from __future__ import division

import sys
import numpy as np
import matplotlib.pyplot as pplot

from approx_dist import get_aggregate_dist, distribution
from data.models import get_session

if __name__ == '__main__':
    #aggregate_ids = [#'166409b2-4009-4705-9d98-5f61a1e21a2a',
    #                 'a96a44fc-7dcb-4c96-b15f-eb57ef03b0e8',
    #                 '166c6f0d-2085-4f95-a718-7445fd7ac0d2',
    #                 '07f6350c-b92e-4088-b76e-5f0f3f6cf58c',
    #                 'fba13f2e-d46a-47fc-b0a6-2b3b44302776']
    aggregate_ids = ['1292221b-3dbb-4ac8-a021-2d33c5bb4f38',
                     'c79cabcd-5ba1-44dd-9924-51a0a0d04fd8',
                     'b658fb0f-c06c-463f-942f-f9aa7e2e1b9f',
                     'eb32d8e1-cac5-48cc-9a64-77c8bafc1e68',
                     'b4fbcad9-f350-4fd1-92b6-5ef1b82b58a8',
                     '11980c19-e24e-4186-94ca-218e65c17899',
                     '380312f4-d7d3-415e-a629-dfe9395df5d9',
                     '1a34d7af-4d37-4b95-98d7-c75f8546af07']

    fig = pplot.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')
    cmap = pplot.get_cmap('YlOrRd')
    n = len(aggregate_ids)
    session = get_session()

    for (i,aggregate_id) in enumerate(aggregate_ids):
        aid,w,alpha,mu,sigma,x0,m,s,wsum,max_tenure,irs_threshold,dissipation,no_banks = session.execute("""
                    SELECT DISTINCT run.max_tenure, run.irs_threshold, run.dissipation, run.no_banks
                    FROM `aggregate_distribution`
                    WHERE run.aggregate_id = :aggId
                    """,{'aggId':aggregate_id}).fetchall()[0]

        x,y,ap,no_banks = get_aggregate_dist(aggregate_id)
        lbl = "Type %d"%(i+1)
        print x, w, alpha, mu, sigma
        if(i < 3):
            ax.plot(x,distribution(x,w,alpha,mu,sigma), color=cmap((i+1)/n),label=lbl)
        else:
            print x
            x1 = x[:9]
            x2 = x[9:]
            ax.plot(x1,distribution(x1,w,alpha,mu,sigma), color=cmap((i+1)/n),label=lbl)
            ax.plot(x2,distribution(x2,w,alpha,mu,sigma), color=cmap((i+1)/n))

    ax.set_ylabel("p(Cascade size)")
    ax.set_xlabel("Cascade size")
    pplot.legend(loc=3)
    pplot.show()
