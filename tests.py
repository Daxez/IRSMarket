"""Yes, I actually validated the first implementation using somekind of unittest thing."""
from __future__ import division

import logging
import matplotlib.pyplot as pplot

from market import ShuffleIRSMarket
from bank import Bank
from irs import IRS
from datacontainer import DataContainer
from math import fabs
from uuid import uuid4


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    c = {'sigma' : 1,
    'no_banks' : 30,
    'no_steps' : 200,
    'irs_threshold' : 15,
    'max_irs_value' : 20,
    'max_tenure' : 80,
    'no_sims' : 1,
    'threshold' : 4}

    mkt = ShuffleIRSMarket(c,
                DataContainer({'file_root':'./test','model':{'no_steps':1}},str(uuid4()),str(uuid4())),
                None)


    banks = mkt.banks
    b1 = banks[0]
    b1.__balance__ = -5
    b2 = banks[1]
    b2.__balance__ = 5
    b3 = banks[2]
    b3.__balance__ = 3

    b1.set_dirty()
    b2.set_dirty()
    b3.set_dirty()

    logging.debug(("Bank 1 balance: %f"%b1.balance))
    logging.debug(("Bank 2 balance: %f"%b2.balance))

    if(b1.balance != -5): raise Exception("Balance is kaput")
    if(b2.balance != 5): raise Exception("Balance is kaput")

    irs = IRS(5,10,0)
    mkt.add_edge(b2,b1,irs)

    b1.set_dirty()
    b2.set_dirty()

    if(b1.balance != 0): raise Exception("Balance/IRS is kaput")
    if(b2.balance != 0): raise Exception("Balance/IRS is kaput")

    logging.debug("Bank 1 balance after IRS: %f"%b1.balance)
    logging.debug("Bank 2 balance after IRS: %f"%b2.balance)

    ne = len(mkt.get_edges())
    logging.debug("%d swaps in the system"% ne)

    if(ne != 1): raise Exception("No swaps in system not correct")

    for i in range(10):
        irs.time_step()

    ne = len(mkt.get_edges())
    logging.debug("%d swaps in the system"% ne)

    if(ne != 0): raise Exception("No swaps in system not correct")

    logging.debug("Bank 1 balance after IRS removal: %f"%b1.balance)
    logging.debug("Bank 2 balance after IRS removal: %f"%b2.balance)

    if(b1.balance != -5): raise Exception("Balance is kaput")
    if(b2.balance != 5): raise Exception("Balance is kaput")

    irs = IRS(0,10,0)
    mkt.add_edge(b2,b1,irs)
    b1.set_dirty()
    b2.set_dirty()

    irs2 = IRS(0,10,0)
    mkt.add_edge(b2,b3,irs2)
    b2.set_dirty()
    b3.set_dirty()

    b1.check_balance()

    ne = len(mkt.get_edges())
    logging.debug("%d swaps in the system"% ne)

    mkt.create_swap(b2,b3)
    mkt.create_swap(b2,b3)
    ne = len(mkt.get_edges())
    logging.debug("%d swaps in the system"% ne)
