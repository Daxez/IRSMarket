"""Bank classes: classes for bank nodes

    Old model.
"""

from __future__ import division

import math
import logging
from collections import defaultdict
import numpy as np
import basegraphs as bg

from uuid import uuid4

class Bank(bg.Node):
    """
    Bank wrapper
    """

    def __init__(self, balance=0, sigma=1,seed=None):
        super(Bank,self).__init__()
        self.__balance__ = balance
        self.__previous_balance__ = self.__balance__
        self.__dirty__ = False
        self.sigma = sigma
        self.seed = seed
        self.degree = 0
        if(self.seed == None):
            # Max seed value
            self.seed = np.random.randint(4294967295)
        self.generator = np.random.RandomState(self.seed)
        self.irs_wt = [0]

    def register_failed_irs(self):
        self.market.datacontainer.register_failed_irs(self.name)

    def register_success_irs(self):
        self.market.datacontainer.register_success_irs(self.name)

    def time_step(self):
        e = self.generator.normal(0,self.sigma)
        self.__balance__ += e
        self.__previous_balance__ += e

    def check_balance(self):
        if(self.check_for_default()):
            self.default(str(uuid4()),0)

    def register_balance(self):
        self.market.datacontainer.register_gross(self.name,self.__balance__)
        self.market.datacontainer.register_net(self.name,self.balance)
        self.market.datacontainer.register_degree(self.name,self.degree)

    def set_dirty(self):
        self.__dirty__ = True

    @property
    def balance(self):
        if(not self.__dirty__):
            return self.__previous_balance__

        nds = defaultdict(int)

        current_balance = self.__balance__
        for irs in self.get_in():
            current_balance += irs.float_value
            nds[irs.fixed.name] += 1

        for irs in self.get_out():
            current_balance += irs.fix_value
            nds[irs.floating.name] += 1

        self.degree = len(nds.keys())
        self.__previous_balance__ = current_balance
        self.__dirty__ = False
        return current_balance

    @property
    def market(self):
        return self.graph

    def check_for_default(self):
        return math.fabs(self.balance) > self.market.threshold

    def activate(self,a_id=None, depth=0):
        if(a_id == None):
            a_id = str(uuid4())

        if(self.check_for_default()):
            self.default(a_id,depth)

    def default(self,a_id,depth):

        #(fixs,floats) = self.get_edges()
        nodes_to_activate = {}
        for irs in self.get_in():
            nodes_to_activate[irs.fixed.name] = irs.fixed
            irs.remove()

        for irs in self.get_out():
            nodes_to_activate[irs.floating.name] = irs.floating
            irs.remove()

        self.market.datacontainer.register_default(a_id,
                                               self.name,
                                               self.balance,
                                               self.market.time,
                                               len(nodes_to_activate),
                                               depth)

        self.__balance__ = 0
        self.__previous_balance__ = 0
        self.__dirty__ = False

        for nd in nodes_to_activate.values():
            nd.activate(a_id,depth+1)

if __name__ == '__main__':
    b = Bank()
    print b.balance
    b.time_step()
