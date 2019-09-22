"""
Also something for moments.
"""

from __future__ import division

from collections import defaultdict

import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as pplot

from analyse_hump import get_aggregate_dist
from data.models import *

def moments_by_type(tpe):
    query = """SELECT aggregate_type.aggregate_id FROM aggregate_type
               INNER JOIN run ON run.aggregate_id = aggregate_type.aggregate_id
               WHERE type_id = %d and run.no_banks = 200
               GROUP BY aggregate_type.aggregate_id"""%tpe

    session = get_session()
    aggregate_ids = session.execute(query).fetchall()
    session.close()

    moments = np.zeros((len(aggregate_ids)))

    for i,aggregate_id in enumerate(aggregate_ids):
        _,_,_,d = get_aggregate_dist(aggregate_id[0])
        moments[i] = stats.moment(d,3)
        print moments[i], aggregate_id

    return moments

if __name__ == '__main__':

    p = 611

    fig = pplot.figure()

    y = np.zeros(5)
    for i in range(1,6):
        print "Starting with type %d"%i
        ax = fig.add_subplot(p+i-1)
        m = moments_by_type(i)
        y[i-1] = np.mean(m)
        ax.hist(m,bins=200)
        ax.set_title("4th moment for distributions with type %d"%i)

    ax = fig.add_subplot(616)
    ax.plot(range(1,6), y)

    pplot.show()
