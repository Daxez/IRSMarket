"""DataContainer: collecting data and updating them to the database, this one is still used"""
# Author: Dexter Drupsteen
from __future__ import division

import datetime
import logging
import os
import pickle
from uuid import uuid4

from collections import defaultdict

from data.models import RunModel,BankModel,SwapModel,DefaultModel,BankDefaultModel,get_session
from utils import Progress

# Run (id, aggregate_id, timestamp, number of steps, number of banks, sigma, irs_threshold, max_irs_value, tenure)
# Bank (Name, Run id)
# Swap (Name, value, bank name floating end, bank name fixed end, created step, tenure)
# Default (Id, banks involved, timestep)

class DataContainer:

    def __init__(self,config,run_id,aggregate_id=None):
        self.run_id = run_id
        self.aggregate_id = aggregate_id
        self.time_stamp = datetime.datetime.now()
        self.config = config
        self.steps = self.config['model']['no_steps']

        # Data
        self.seed = 0
        self.swaps = defaultdict()
        self.banks = defaultdict()
        self.defaults = defaultdict(int)
        self.default_participants = defaultdict(list)
        self.gross = defaultdict(list)
        self.net = defaultdict(list)
        self.degree = defaultdict(list)
        self.irs_tries = defaultdict(list)

        #settings
        self.root_path = config['file_root']+self.aggregate_id+'/'+self.run_id

    def register_seed(self,seed):
        if('seed' in self.config['analysis']['data_to_save']):
            self.seed = seed

    def register_swap(self,sid,float_id,fix_id,value,start,end,tenure):
        if('swaps' in self.config['analysis']['data_to_save']):
            self.swaps[sid] = (float_id,fix_id,value,start,end,tenure)

    def register_gross(self,bid,val):
        if('balances' in self.config['analysis']['data_to_save']):
            self.gross[bid].append(val)

    def register_net(self,bid,val):
        if('balances' in self.config['analysis']['data_to_save']):
            self.net[bid].append(val)

    def register_degree(self,bid,val):
        if('degree' in self.config['analysis']['data_to_save']):
            self.degree[bid].append(val)

    def register_bank(self,bid,seed):
        self.banks[bid] = seed

    def register_failed_irs(self,bid):
        if('irs_tries' in self.config['analysis']['data_to_save']):
            if(len(self.irs_tries[bid]) == 0):
                self.irs_tries[bid].append(2)

            self.irs_tries[bid][-1] += 1

    def register_success_irs(self,bid):
        if('irs_tries' in self.config['analysis']['data_to_save']):
            self.irs_tries[bid].append(1)

    def register_default_simple(self,size):
        self.defaults[size] += 1

    def save_data(self):
        self.save_run()
        self.save_banks()
        self.save_defaults()

    def save_run(self):
        ses = get_session()

        avgswaps = len(self.swaps)/(self.config['model']['no_steps']*self.config['model']['no_banks'])

        run = RunModel(self.run_id,
                       self.aggregate_id,
                       self.config['model']['no_steps'],
                       self.config['model']['no_banks'],
                       self.config['model']['sigma'],
                       self.config['model']['irs_threshold'],
                       self.config['model']['max_irs_value'],
                       self.config['model']['max_tenure'],
                       self.config['model']['threshold'],
                       self.time_stamp, self.seed,
                       self.config['market_type'],
                       avgswaps,len(self.swaps),
                       self.config['model']['dissipation'])

        ses.bulk_save_objects([run])

        ses.commit()
        ses.close()

    def save_banks(self):
        bnks = []
        if(len(self.banks)> 0):
            for bank in self.banks:
                b = BankModel(bank,self.run_id,self.banks[bank])
                bnks.append(b)

            ses = get_session()
            ses.bulk_save_objects(bnks)
            ses.commit()
            ses.close()

    def save_swaps(self):
        swp_objs = []
        for sid in self.swaps:
            (float_id,fix_id,value,start,end,tenure) = self.swaps[sid]
            swp_objs.append(SwapModel(sid,value,float_id,fix_id,start,end,tenure,self.run_id))

        prev = 0
        for x in range(0,len(swp_objs),40000):
            ses = get_session()
            ses.bulk_save_objects(swp_objs[prev:x])
            ses.commit()
            ses.close()
            prev = x

        if(len(swp_objs[prev:]) > 0):
            ses = get_session()
            ses.bulk_save_objects(swp_objs[x:])
            ses.commit()
            ses.close()

    def save_defaults(self):
        print "Saving avalanche data"
        query = """INSERT INTO default_aggregate (aggregate_id, frequency, size)
           VALUES (:aggregate_id, :freq, :size)
           ON DUPLICATE KEY UPDATE frequency = frequency + :freq"""

        session = get_session()

        for dkey in self.defaults:
            session.execute(query,{'aggregate_id': self.aggregate_id,
                                   'freq': self.defaults[dkey],
                                   'size': dkey})

        session.commit()
        session.close()

    def save_default_participants(self):
        dps = []
        for did in self.defaults:
            for (bank_id,balance,root) in self.default_participants[did]:
                dps.append(BankDefaultModel(did,bank_id,balance,root,self.run_id))

        prev = 0
        for x in range(0,len(dps),40000):
            ses = get_session()
            ses.bulk_save_objects(dps[prev:x])
            prev = x
            ses.commit()
            ses.close

        if(len(dps[prev:]) > 0):
            ses = get_session()
            ses.bulk_save_objects(dps[x:])
            ses.commit()
            ses.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.swaps
        del self.defaults
        del self.default_participants
        del self.gross
        del self.net
        del self.degree
