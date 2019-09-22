"""
If you want to run simulations (more than one) THIS is the file to go.
Run: pypy quick_sweep. Set the file to the right configuration (see config_sim) to see what you can track
After the if(__name__)==main thing, you can set your configuration ranges, number of steps etc.
"""
from __future__ import division

import time
import itertools
import os
import pickle
import copy
from datetime import datetime, timedelta
from collections import defaultdict
from uuid import uuid4

import numpy as np
from utils import Progress, is_valid_uuid
from datacontainer import DataContainer
from quick import sim

def config_sim(s):
    s.save_avalanche_progression = False
    s.save_risk_avalanche_time_series = False
    s.collect_critical_info = False
    s.save_giant_component = False
    s.save_avalanche_tree = False
    s.save_degree_distribution = False
    s.avalanche_tree_file_path = './simulation_data/trees/%s/'%s.dc.aggregate_id
    s.save_degree_on_default = False
    s.save_default_rate = False
    s.save_time_between_large_events = False
    s.save_abs_risk_and_dissipation = False
    s.save_gross_risk_for_avalanche_size = False#True
    s.save_density_for_avalanche_size = False#True
    s.save_average_degree_on_default = False

    if(s.save_density_for_avalanche_size):
        s.density_per_avalanche_size = defaultdict(list)
    if(s.save_gross_risk_for_avalanche_size):
        s.gross_risk_per_avalanche_size = defaultdict(list)

    if(s.save_giant_component):
        s.giant_components = np.zeros(s.no_steps)
    if(s.save_degree_distribution):
        s.degrees = np.zeros((steps,s.no_banks))
        s.no_irs = np.zeros((steps,s.no_banks))
    if(s.save_abs_risk_and_dissipation):
        s.balance_dissipation = np.zeros(steps)
        s.abs_added_risk = np.zeros(steps)

    if(s.save_average_degree_on_default):
        s.average_degree_on_default = defaultdict(list)
        s.degree_on_default = defaultdict(list)


if(__name__ == '__main__'):
    steps = 100000

    dcconfig = {
        'model':{
            'no_banks' : 100,
            'no_steps' : steps,
            'threshold' : 10,
            'sigma' : 1,
            'irs_threshold':-2,
            'max_irs_value' : 1,
            'dissipation' : 0.0,
            'max_tenure' : 550
        },
        'analysis':{
            'data_to_save':['defaults']
        },
        'file_root':'./simulation_data/',
        'market_type':7
    }

    save = True

    #bank_sweep = [100]
    #bank_sweep = [200]
    #bank_sweep = [300]
    #bank_sweep = [400]
    #bank_sweep = [500, 600, 700, 800]
    #bank_sweep = [900, 1000, 2000]
    #bank_sweep = [3000,4000]
    bank_sweep = [300, 400, 500, 1000, 2000]
    max_irs_value_sweep = [7.0]
    tenure_sweep = [250]
    threshold_sweep = [10]#[10,15,20,25,30]

    params = itertools.product(bank_sweep,max_irs_value_sweep,tenure_sweep,threshold_sweep)

    no_reps = 5
    nosims = no_reps*len(bank_sweep)*len(max_irs_value_sweep)*len(tenure_sweep)*len(threshold_sweep)
    cnt = 1

    txt = "Going to do %d sims\nContinue?[y/N]"%nosims
    res = raw_input(txt)
    if(res.lower() == 'n' or res.lower() == 'no' or len(res) == 0):
        exit(0)

    save_risk = False

    aggstart = datetime.now()
    for (i,(no_banks,max_irs_value,tenure,threshold)) in enumerate(params):
        print "Starting with b %d,%f,%d,%f"%(no_banks,max_irs_value,tenure,threshold)

        cur_config = copy.deepcopy(dcconfig)

        cur_config['model']['no_banks'] = no_banks
        cur_config['model']['max_irs_value'] = max_irs_value
        cur_config['model']['max_tenure'] = tenure
        cur_config['model']['threshold'] = threshold

        aggregate_id = str(uuid4())
        for i in xrange(no_reps):
            print "Startin run %d of %d"%(cnt,nosims)

            run_id = str(uuid4())
            p = Progress(steps)
            start = time.time()

            with DataContainer(cur_config, run_id, aggregate_id) as dc:
                with sim(cur_config['model'],dc,p.update,save_risk,False) as cursim:
                    config_sim(cursim)
                    p.start()
                    cursim.run()
                    p.finish()

                    if(save):
                        dc.save_defaults()
                        dc.save_run()

                    if cursim.save_degree_distribution:
                        directory = './simulation_data/k/irs_value_%s'%max_irs_value
                        file_path = '%s/%s_%s.bin'%(directory,dc.aggregate_id,i)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        with file(file_path,'wb') as fp:
                            pickle.dump(cur_config,fp)
                            pickle.dump(cursim.no_irs.tolist(),fp)
                    if cursim.save_abs_risk_and_dissipation:
                        file_path = './simulation_data/risk_and_balance_dissipation/sweep_200/%s.bin'%(dc.run_id)
                        with file(file_path,'wb') as fp:
                            pickle.dump(cur_config,fp)
                            pickle.dump(cursim.abs_added_risk.tolist(),fp)
                            pickle.dump(cursim.balance_dissipation.tolist(),fp)
                    if save_risk:
                        file_path = './simulation_data/risk/%s.bin'%(dc.aggregate_id)
                        with file(file_path,'wb') as fp:
                            pickle.dump(cur_config,fp)
                            pickle.dump(cursim.risk.tolist(),fp)
                    
                    if cursim.save_time_between_large_events:
                        file_path = './simulation_data/large_default_rate/%s_%s.bin'%(dc.aggregate_id,i)
                        with file(file_path,'wb') as fp:
                            pickle.dump(no_banks,fp)
                            pickle.dump(cursim.tble,fp)

                    if cursim.save_density_for_avalanche_size:
                        file_path = './simulation_data/density_for_avalanche_size'
                        abs_file_path = "%s/%s_%s.bin"%(file_path, dc.aggregate_id,i)
                        if(not os.path.exists(file_path)):
                            os.makedirs(file_path)

                        with file(abs_file_path, 'wb') as fp:
                            pickle.dump(cur_config,fp)
                            pickle.dump(cursim.density_per_avalanche_size,fp)

                    if cursim.save_gross_risk_for_avalanche_size:
                        file_path = './simulation_data/gross_risk_for_avalanche_size'
                        abs_file_path = "%s/%s_%s.bin"%(file_path, dc.aggregate_id,i)
                        if(not os.path.exists(file_path)):
                            os.makedirs(file_path)

                        with file(abs_file_path, 'wb') as fp:
                            pickle.dump(cur_config,fp)
                            pickle.dump(cursim.gross_risk_per_avalanche_size,fp)

                    if cursim.save_average_degree_on_default:
                        file_path = './simulation_data/average_degree_on_default'
                        abs_file_path = "%s/%s_%s.bin"%(file_path, dc.aggregate_id,i)
                        if(not os.path.exists(file_path)):
                            os.makedirs(file_path)

                        with file(abs_file_path, 'wb') as fp:
                            pickle.dump(cur_config,fp)
                            pickle.dump(cursim.average_degree_on_default,fp)
                            pickle.dump(cursim.degree_on_default,fp)
                    


            cnt += 1
            print "Took %d seconds"%(time.time()-start)
            expected_remain = (datetime.now() - aggstart) * int((nosims-cnt)/cnt)
            eta = datetime.now()+expected_remain
            print "Probably lasts till %s"%eta.isoformat()
