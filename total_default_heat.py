"""
Yet another file to plot an heatmap of the total number of defaults.
"""
from __future__ import division
import sys

import numpy as np
import matplotlib.pyplot as pplot

from data.models import get_session

def get_no_defaults( no_banks, threshold, irs_value, tenure):
    query = """SELECT r.aggregate_id, max_tenure, max_irs_value, threshold, sum(da.frequency)
               FROM run as r
               INNER JOIN default_aggregate as da ON da.aggregate_id = r.aggregate_id
               WHERE no_banks = :no_banks AND threshold  = :threshold
               AND max_irs_value = :max_irs_value AND max_tenure = :max_tenure
               GROUP BY r.aggregate_id, r.max_irs_value, r.max_tenure, r.threshold
               """
    session = get_session()
    info = session.execute(
        query, {
            'no_banks': no_banks,
            'max_tenure': tenure,
            'threshold': threshold,
            'max_irs_value': irs_value,
        }).first()
    session.close()
    return info


if __name__ == '__main__':

    irs_threshold = np.arange(1, 9)
    tenure = [50, 100, 250, 400, 550, 700, 850]
    no_banks = 100
    thresholds = [10, 15, 20]

    # from the db
    min_value = 231160
    max_value = 7799440

    for t in thresholds:
        im_data = np.zeros((len(irs_threshold), len(tenure)))
        for i, irst in enumerate(irs_threshold):
            for j, ten in enumerate(tenure):
                info = get_no_defaults(no_banks, t, irst, ten)
                im_data[i, j] = info[4]

        fig = pplot.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel("IRS Value")
        ax.set_yticklabels(np.arange(0,9))
        ax.set_xlabel("Term to maturity")
        ax.set_xticklabels([0,50,100,250,400,550,700,850])
        ax.set_title("Type for %d banks, threshold %d" % (no_banks, t))

        cax = ax.imshow(im_data, origin='lower', interpolation='nearest', vmin=min_value, vmax=max_value)

        cb = fig.colorbar(cax)
        cb.set_ticks([min_value, max_value])
        cb.set_ticklabels(["%.1E"%min_value, "%.1E"%max_value])
        cb.set_label('Number of defaults', rotation=270)

        filename = 'no_defaults_irs_ten_%d_banks_%d_threshold.png' % (no_banks, t)
        path = '/home/..../Programming/GitRepos/scriptie/figs/heatmaps/defaults/'
        fl = '%s%s' % (path, filename)

        pplot.savefig(fl)
        pplot.close(fig)
