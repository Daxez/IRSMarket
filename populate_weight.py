"""Adding value to the humpweight table for all experiments"""
from __future__ import division

from collections import defaultdict

from data.models import AggregateHumpWeight, get_session
from analyse_hump import get_aggregate_dist, get_hump_error, get_hump_cumulative_value
from utils import Progress

def get_aggregate_ids():
    query = """SELECT DISTINCT aggregate_id
               FROM run
               WHERE no_banks >= 200
               """
    session = get_session()
    agg_ids = session.execute(
        query).fetchall()
    session.close()
    return [a[0] for a in agg_ids]


def get_weight(aggregate_id):
    x, y, _, raw = get_aggregate_dist(aggregate_id)
    _, alp = get_hump_error(x, y, raw)
    cumv = get_hump_cumulative_value(x,y)
    return (aggregate_id, alp, cumv)

if __name__ == "__main__":

    aggregateids = get_aggregate_ids()

    rows = []
    p = Progress(len(aggregateids))
    p.start()
    for (i,aggid) in enumerate(aggregateids):
        rows.append(AggregateHumpWeight(*get_weight(aggid)))
        p.update(i)
    
    p.finish()
    
    session = get_session()
    session.bulk_save_objects(rows)
    session.commit()
    session.close()


