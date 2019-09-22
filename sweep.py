"""Old: Script for performing a sweep"""
import copy
import logging
import uuid
import numpy as np
import matplotlib.pyplot as pplot

from simulator import Simulation
from config import default_configuration,import_configuration
from utils import Progress
from aggregate import Aggregate
from heatmap import heat_map

def get_fresh_config(test = False):
    config = copy.deepcopy(default_configuration)

    config['model']['no_banks'] = 20
    config['model']['threshold'] = 20

    config['model']['no_steps'] = 500000
    if(test):
        config['model']['no_steps'] = 500

    config['market_type'] = 2

    config['simulation']['save_data'] = True
    if(test):
        config['simulation']['save_data'] = False

    config['simulation']['repeat'] = 10
    if(test):
        config['simulation']['repeat'] = 1
    config['aggregate']['do_aggregate'] = False
    config['analysis']['methods'] = []
    config['analysis']['do_analysis'] = False
    config['analysis']['data_to_save'] = ['defaults']
    config['file_root'] = "./sweep/exponentv2/"

    if(test):
        config['file_root'] = "./sweep/test/"

    return config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    config = get_fresh_config()
    orig_file_root = config['file_root']
    config['file_root'] += 'hm/'

    sigma_sweep = range(1,11)
    max_tenure = range(10,200,20)
    threshold = range(30,130,10)

    results = np.zeros((5,5))

    irs_thrs = [2,4,6,8,10]
    irs_max = [12,14,16,18,20]

    if(False):
        print "Going to sweep over Maximum and threshold IRS values"

        for (i,irs_ths) in enumerate(irs_thrs):
            for (j,irs_mx) in enumerate(irs_max):
                config['model']['irs_threshold'] = irs_ths
                config['model']['max_irs_value'] = irs_mx

                aggregate_id = str(uuid.uuid4())
                sim = Simulation(config,aggregate_id)
                sim.run()
                a = Aggregate(config['file_root'],aggregate_id,True)

                results[i,j] = a.default_distribution()

        print results
        path = orig_file_root+"%s.png"
        cbmax = int(np.max(results)+0.5)
        #cbmax = int(5.2+0.5)
        heat_map(results,"Exponent heatmap",'IrsThreshold','MaxIrsValue',irs_thrs,irs_max,range(0,cbmax),path,"IrsThs_IrsMax")

        min_alpha = np.min(results[np.where(results > 0)])



        (xr,yr) = np.where(results == min_alpha)
        i = xr[0]
        j = yr[0]
        irs_ths = irs_thrs[i]
        irs_mx = irs_max[j]

    irs_ths = irs_thrs[4]
    irs_mx = irs_max[3]

    config = get_fresh_config()
    orig_file_root = config['file_root']
    config['file_root'] += 'th/'
    config['model']['irs_threshold'] = irs_ths
    config['model']['max_irs_value'] = irs_mx

    tenrng = range(10,110,5)
    results = np.zeros(20)
    print "Going to sweep over the tenure time with the minimal threshold and values"
    
    for (i,ten) in enumerate(tenrng):
        config['model']['max_tenure'] = ten

        aggregate_id = str(uuid.uuid4())
        sim = Simulation(config,aggregate_id)
        sim.run()
        a = Aggregate(config['file_root'],aggregate_id,True)

        results[i] = a.default_distribution()

    print results

    fig = pplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(tenrng,results)
    ax.set_title("Exponent value over tenure rates")
    pplot.savefig(orig_file_root+'tenure.png')
    pplot.close()


    print "Going to check the statistical stability of the values"

    (xr,) = np.where(results == np.min(results[np.where(results > 0)]))
    ten = tenrng[xr[0]]

    config = get_fresh_config()
    orig_file_root = config['file_root']
    config['file_root'] += 'sts/'
    config['model']['irs_threshold'] = irs_ths
    config['model']['max_irs_value'] = irs_mx
    config['model']['max_tenure'] = ten

    results = np.zeros(20)
    for i in xrange(20):
        aggregate_id = str(uuid.uuid4())
        sim = Simulation(config,aggregate_id)
        sim.run()
        a = Aggregate(config['file_root'],aggregate_id,True)

        results[i] = a.default_distribution()

    print results
    print "Mean: %f, Standard Deviation: %f"%(np.mean(results),np.std(results,dtype=np.float64))
