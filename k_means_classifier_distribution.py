"""
Kmean clasifier on the distribution
"""
from __future__ import division
import sys

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs

from data.models import *
from analyse_hump import get_hump_error, get_aggregate_dist, get_hump_ks_error, get_hump_cumulative_value
from sklearn.cluster import KMeans


def get_aggregate_ids(no_banks):
    query = """SELECT DISTINCT aggregate_id, max_tenure, max_irs_value, threshold
               FROM run
               WHERE no_banks = :no_banks AND threshold < 25
               """
    session = get_session()
    agg_ids = session.execute(
        query, {'no_banks': no_banks}).fetchall()
    session.close()
    return agg_ids


def get_frequencies(no_banks, aggregate_id):
    session = get_session()
    freqs = session.execute("""SELECT size, frequency
                               FROM default_aggregate
                               WHERE aggregate_id = :aggregate_id
                               ORDER BY size""",
                            {'aggregate_id': aggregate_id}).fetchall()

    session.close()

    y = np.zeros(no_banks)

    for row in freqs:
        y[row[0]-1] = row[1]

    return y/sum(y)


if __name__ == '__main__':

    irs_threshold = np.arange(1, 9)
    tenure = [50, 100, 250, 400, 550, 700, 850]
    no_banks = 100
    thresholds = [10, 15, 20]
    aggregate_ids = get_aggregate_ids(no_banks)

    data = []
    datapoints = {}
    for t in thresholds:
        datapoints[t] = {}

        for tn in tenure:
            datapoints[t][tn] = {}

    for (i, (aggregate_id, ten, irs_val, thresh)) in enumerate(aggregate_ids):
        vec = get_frequencies(no_banks, aggregate_id)
        data.append(vec)
        datapoints[thresh][ten][irs_val] = i

    kmeans = KMeans(4, init='random').fit_predict(data)


    for t in thresholds:
        im_data = np.zeros((len(irs_threshold), len(tenure)))
        for i, irst in enumerate(irs_threshold):
            for j, ten in enumerate(tenure):
                im_data[i, j] = kmeans[datapoints[t][ten][irst]]

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel("IRS Value")
        ax.set_yticklabels(irs_threshold)
        ax.set_xlabel("Term to maturity")
        ax.set_xticklabels(tenure)
        ax.set_title("Type for %d banks, threshold %d" % (no_banks, t))

        cax = ax.imshow(im_data, origin='lower', interpolation='nearest')

        cb = fig.colorbar(cax)
        cb.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cb.set_ticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cb.set_label('Alpha', rotation=270)
        # pplot.show()

        filename = 'alpha_irs_ten_%d_banks_%d_threshold.png' % (no_banks, t)
        path = '/home/..../Programming/GitRepos/scriptie/figs/heatmaps/types/'
        fl = '%s%s' % (path, filename)

        pplot.savefig(fl, bbox_inches='tight')
        pplot.close(fig)
