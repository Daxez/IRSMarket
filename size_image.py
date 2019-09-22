"""
Heatmap where the most defaults happen (looks like that anyway)
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as pplot
import matplotlib.cm as clrs

from data.models import *

if __name__ == '__main__':

    part_of_graph = 0.7
    query = """SELECT
      run.`aggregate_id`,
      `no_steps`,
      `no_banks`,
      `sigma`,
      `irs_threshold`,
      `max_irs_value`,
      `max_tenure`,
      `threshold`,
      SUM(default_aggregate.frequency),
      avg(default_aggregate.size),
      da2.totfreq,
      SUM(default_aggregate.frequency)/da2.totfreq
    FROM
      `run`
      INNER JOIN default_aggregate
      	ON run.aggregate_id = default_aggregate.aggregate_id
      INNER JOIN (
      	SELECT da.aggregate_id as aggregate_id, SUM(da.frequency) as totfreq
        FROM default_aggregate as da
        GROUP BY da.aggregate_id
      ) as da2 ON da2.aggregate_id = run.aggregate_id
    WHERE
       no_steps = 200000 AND default_aggregate.size > %f*(run.no_banks)
     GROUP BY
    run.`aggregate_id`,
      run.`no_steps`,
      run.`no_banks`,
      run.`sigma`,
      run.`irs_threshold`,
      run.`max_irs_value`,
      run.`max_tenure`,
      run.`threshold`,
      da2.totfreq"""%part_of_graph

    session = get_session()
    data = session.execute(query).fetchall()

    irs_threshold = np.arange(2,12,2)
    tenure = np.arange(400,900,100)
    no_banks = np.arange(100,600,100)
    threshold = np.arange(10,60,10)

    a = raw_input("Do you want to see IRS value vs tenure?[Y/n]")
    if(a != 'n'):
        for t in threshold:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(tenure)))
                for d in data:
                    if d[2] != b or d[7] != t: continue

                    x = np.where(irs_threshold == int(d[5]))[0][0]
                    y = np.where(tenure == d[6])[0][0]

                    im_data[x,y] = d[11]

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(2,12,2))
                ax.set_xlabel("Tenure")
                ax.set_xticklabels(np.arange(400,900,100))
                ax.set_title("Heatmap for %d banks, threshold %d"%(b,t))

                ax.pcolor(im_data,cmap=clrs.Reds,vmin=np.min(im_data),vmax=np.max(im_data))
                pplot.show()

                try:
                    a = raw_input("Do you want to continue [Y/n]")
                    if(a == 'n'):
                        exit(0)
                except KeyboardInterrupt:
                    exit(0)

    a = raw_input("Do you want to see IRS value vs threshold?[Y/n]")
    if(a != 'n'):
        for t in tenure:
            for b in no_banks:
                im_data = np.zeros((len(irs_threshold),len(tenure)))
                for d in data:
                    if d[2] != b or d[6] != t: continue

                    x = np.where(irs_threshold == int(d[5]))[0][0]
                    y = np.where(threshold == d[7])[0][0]

                    im_data[x,y] = d[11]

                fig = pplot.figure()
                ax = fig.add_subplot(111)
                ax.set_ylabel("IRS Value")
                ax.set_yticklabels(np.arange(2,12,2))
                ax.set_xlabel("Threshold")
                ax.set_xticklabels(np.arange(10,60,10))
                ax.set_title("Heatmap for %d banks, tenure %d"%(b,t))

                ax.pcolor(im_data,cmap=clrs.Reds,vmin=np.min(im_data),vmax=np.max(im_data))
                pplot.show()

                try:
                    a = raw_input("Do you want to continue [Y/n]")
                    if(a == 'n'):
                        exit(0)
                except KeyboardInterrupt:
                    exit(0)
