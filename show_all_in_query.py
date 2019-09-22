"""Not used anymore"""
from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as pplot

from show import show_single
from analyse_hump import get_hump_info
from data.models import *
from utils import Progress

def manually_type():
    query = """SELECT DISTINCT run.aggregate_id FROM `run`
               LEFT OUTER JOIN aggregate_type
                    ON run.aggregate_id = aggregate_type.aggregate_id
               WHERE dissipation < 0.01 and aggregate_type.type_id IS NULL"""

    session = get_session()

    aggregate_ids = session.execute(query).fetchall()
    session.close()

    for aggid in aggregate_ids:
        aggregate_id = aggid[0]

        show_single(aggregate_id)
        idinput = raw_input('What type?\n')
        type_id = int(idinput)

        session = get_session()
        session.bulk_save_objects([AggregateType(aggregate_id,type_id)])
        session.commit()
        session.close()

def get_hump_error_per_type():
    query = """SELECT DISTINCT run.aggregate_id, aggregate_type.type_id FROM `run`
               LEFT OUTER JOIN aggregate_type
                    ON run.aggregate_id = aggregate_type.aggregate_id
               WHERE dissipation < 0.01"""

    session = get_session()

    aggregate_ids = session.execute(query).fetchall()
    session.close()

    errors = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }
    p = Progress(len(aggregate_ids))
    p.start()
    cnt = 0
    for aggregate_id,type_id in aggregate_ids:
        err,alpha = get_hump_info(aggregate_id)
        errors[type_id].append(err)
        cnt += 1
        p.update(cnt)
    p.finish()

    x = errors.keys()
    y = [np.mean(yi) for yi in errors.values()]
    yerr = [np.std(yi) for yi in errors.values()]

    ymax = [np.max(yi) for yi in errors.values()]
    ymin = [np.min(yi) for yi in errors.values()]

    with file('./simulation_data/type_hump_error_ranges.bin','wb') as fp:
        pickle.dump(errors,fp)

    fig = pplot.figure()
    ax = fig.add_subplot(311)

    ax.bar(x,y,yerr=yerr,color='b')
    ax.set_ylabel("Avergage power law error")
    ax.set_xlabel("Type")

    ax = fig.add_subplot(312)
    ax.set_ylabel("Maximum power law error")
    ax.set_xlabel("Type")
    ax.bar(x,ymax, color='r')

    ax = fig.add_subplot(313)
    ax.set_ylabel("Minimum power law error")
    ax.set_xlabel("Type")
    ax.bar(x,ymin, color='g')

    pplot.show()



if __name__ == '__main__':

    manually_type()
