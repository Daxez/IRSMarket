"""OLD: File for running multiple simulations"""
from __future__ import division

import logging
import gc
import time
import json
import os
import matplotlib.pyplot as pplot

from uuid import uuid4

from market import *
from utils import Progress
from datacontainer import DataContainer
from analysis import Analysis
from aggregate import Aggregate
from quick import sim


class Simulation:

    def __init__(self,config,aggregate_id = None,name=""):
        self.aggregate_id = aggregate_id
        if(aggregate_id == None):
            self.aggregate_id = str(uuid4())
        self.config = config
        self.no_sims = config['simulation']['repeat']
        self.save_data = config['simulation']['save_data']
        self.root_path = config['file_root']+self.aggregate_id
        self.name = name
        if(len(name) > 0):
            self.root_path = config['file_root']+name+'_'+self.aggregate_id
        self.market_type = config['market_type']

        self.analysis_to_file = self.config['analysis']['save_to_file']
        self.analysis_methods = self.config['analysis']['methods']

        self.aggregate_to_file = self.config['aggregate']['save_to_file']
        self.aggregate_methods = self.config['aggregate']['methods']

        if(self.analysis_to_file or self.save_data):
            if(not os.path.exists(self.root_path)):
                os.makedirs(self.root_path)
            with open(self.root_path+'/current.config','w') as fp:
                json.dump(config,fp,indent=4)

    def do_analysis(self,datacontainer,run_id):
        anal_root = self.root_path+'/'+run_id
        if(self.no_sims == 1):
            anal_root = self.root_path
        an = Analysis(anal_root,self.analysis_to_file,self.name)
        for method in self.analysis_methods:
            getattr(an,method)(datacontainer)

    def do_aggregate(self,datacontainer,run_id):
        ag = Aggregate(self.root_path,self.aggregate_id,self.aggregate_to_file,self.name)
        for meth in self.aggregate_methods:
            getattr(ag,meth)()

    def run(self):
        agstart = time.time()
        for i in xrange(self.no_sims):
            logging.info("Going for simulation %d"%(i+1))
            gc.collect()
            run_id = str(uuid4())

            with DataContainer(self.config,run_id,self.aggregate_id) as dc:
                p = Progress(self.config['model']['no_steps'])

                model_class = None
                if(self.market_type == 1):
                    logging.info("Using default Market")
                    model_class = Market
                elif(self.market_type == 2):
                    logging.info("Using ShuffleIRSMarket")
                    model_class = ShuffleIRSMarket
                elif(self.market_type == 3):
                    logging.info("Using SortedIRSMarket")
                    model_class = SortedIRSMarket
                elif(self.market_type == 4):
                    logging.info("Using RandomSortedIRSMarket")
                    model_class = SortedRandomIRSMarket
                elif(self.market_type == 5):
                    logging.info("Using RandomShuffleIRSMarket")
                    model_class = ShuffleRandomIRSMarket
                elif(self.market_type == 6):
                    logging.info("Using ConstantRandomShuffleIRSMarket")
                    model_class = ConstShuffleIRSMarket
                elif(self.market_type == 7):
                    logging.info("Using quick CRS-IRS-Mkt")
                    model_class = sim
                else:
                    raise "No such market type"

                p.start()
                start = time.time()
                with model_class(self.config['model'],dc,p.update) as m:
                    m.run()

                t = time.time()-start
                p.finish()

                print ""
                logging.info("Run took %f seconds"%t)

                if(self.config['analysis']['do_analysis']):
                    start = time.time()
                    self.do_analysis(dc,run_id)
                    t = time.time()-start
                    logging.info("Analysis took %f seconds"%t)

                if(self.save_data):
                    start = time.time()
                    dc.save_data()
                    t = time.time()-start
                    logging.info("Saving data took %f seconds"%t)

            gc.collect()
            print ""
            print ""

        gc.collect()
        dt = (time.time() - agstart) / 60
        logging.info("Experiment took %f minutes"%dt)

        if(self.config['aggregate']['do_aggregate'] and self.save_data):
            start = time.time()
            self.do_aggregate(dc,run_id)
            logging.info('Aggregation took %f seconds'%(time.time()-start))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sigma = 1
    no_banks = 50
    no_steps = 20000
    irs_threshold = 15
    max_irs_value = 20
    max_tenure = 80
    no_sims = 1
    threshold = 50

    simm = Simulation(threshold,sigma,no_banks,no_steps,irs_threshold,max_irs_value,max_tenure,no_sims)
    simm.run(True)
