"""Configuration file
Bas for a configuration. usually do it manually now.
"""
# Author: Dexter Drupsteen

import json
import copy

class Configuration(dict):
    def __init__ (self, *args, **kwargs):
        super(Configuration,self).__init__(*args,**kwargs)
        self.__d

default_configuration = {
    'model':{
        'sigma': 1,
        'no_banks':50,
        'no_steps':10000,
        'irs_threshold': 15,
        'max_irs_value': 20,
        'max_tenure': 80,
        'threshold': 50,
        'dissipation':0.0 
    },
    'simulation':{
        'save_data': True,
        'repeat': 1
    },
    'analysis': {
        'do_analysis': True,
        'methods': ['no_swaps_through_time',
                    'default_histogram',
                    'balance_through_time',
                    'links_at_defaults',
                    'irs_tries_distribution',
                    'average_degree'],
        'data_to_save':['seed',
                        'swaps',
                        'balances',
                        'degree',
                        'defaults',
                        'irs_tries',
                        'default_members'],
        'save_to_file': True
    },
    'aggregate': {
        'do_aggregate': True,
        'methods': ['average_activity',
                    'waiting_time_distribution',
                    'default_distribution',
                    'default_scatter'],
        'save_to_file': True
    },
    'logging_level':20,
    'file_root': './simulation_data/',
    'market_type': 7
}

def replace_in_dirs(source,target):
    for k in target:
        if(isinstance(target[k],dict) and k in source):
            tmp = replace_in_dirs(source[k],target[k])
            source[k] = tmp
        else:
            source[k] = target[k]
    return source

def import_configuration(path):
    config = copy.deepcopy(default_configuration)
    tmp = {}

    if path == None:
        return config

    with open(path,'r') as fp:
        tmp = json.load(fp)

    return replace_in_dirs(config,tmp)

if __name__ == '__main__':
    with open('configuration/default.config','w') as fp:
        json.dump(default_configuration,fp,indent=4)
