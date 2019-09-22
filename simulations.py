"""OLD Main module for starting simulations"""
from __future__ import division

import argparse
import os
import logging
import json
import pprint

from simulator import Simulation
from config import import_configuration

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help="The configuration file to use",
                        metavar="config")
    parser.add_argument('--sigma',help="Internal sigma of the bank", type=float,metavar="sigma")
    parser.add_argument('--banks',help="Number of banks",type=int, metavar="Number")
    parser.add_argument('--steps',help="Number of steps",type=int, metavar="Number")
    parser.add_argument('--irs-threshold',help="Minimum balance value to start irss",type=float,
                        metavar="Threshold")
    parser.add_argument('--max-irs',help="Maximum value of an irs",type=float,metavar="Value")
    parser.add_argument('--max-tenure',help="Maximum tenure time",type=float,metavar="Tenure")
    parser.add_argument('--threshold',help="Maximum balance threshold",type=float,metavar="Threshold")
    parser.add_argument('--save',help="Save the data to the database",choices=[0,1],type=int,metavar="Save")
    parser.add_argument('--repeat',help="Repeat the experiment for aggregation",type=int,metavar="Number")
    parser.add_argument('--save-figs',help="Saves the figures or plots",choices=[0,1],type=int,metavar="Save figs")
    parser.add_argument('--logging',help="Logging level (10 debug 20 info 30 warn 40 error)",
                        type=int,choices=[10,20,30,40],metavar="Level")
    parser.add_argument('--aggregate-id',help="Set the aggregate ID to add simulations to the aggregate id",metavar="Aggregate ID")
    parser.add_argument('--market-type',help="Set market type to use",metavar="Market type",choices=[1,2,3,4,5,6,7],type=int)
    parser.add_argument('--name',help="Name of the aggregate",metavar="Name",type=str,default="")
    parser.add_argument('--no-confirm',help="Don't confirm the settings",action="store_true")

    args = parser.parse_args()

    path = args.config
    if(path != None):
        if(not os.path.exists(path)):
            if( not os.path.exists('./configuration/'+path)):
                err = "Could not find configuration file at %s or %s"%(path,'./configuration/'+path)
                raise Exception(err)
            else:
                path = './configuration/'+path

    config = import_configuration(path)

    if(args.sigma):
        config['model']['sigma'] = args.sigma
    if(args.banks):
        config['model']['no_banks'] = args.banks
    if(args.steps):
        config['model']['no_steps'] = args.steps
    if(args.irs_threshold):
        config['model']['irs_threshold'] = args.irs_threshold
    if(args.max_irs):
        config['model']['max_irs_value'] = args.max_irs
    if(args.max_tenure):
        config['model']['max_tenure'] = args.max_tenure
    if(args.threshold):
        config['model']['threshold'] = args.threshold
    if(args.save != None):
        if(args.save == 0):
            config['simulation']['save_data'] = False
        else:
            config['simulation']['save_data'] = True
    if(args.repeat):
        config['simulation']['repeat'] = args.repeat
    if(args.save_figs != None):
        if(args.save_figs == 0):
            config['analysis']['save_to_file'] = False
        else:
            config['analysis']['save_to_file'] = True
    if(args.logging):
        config['logging_level'] = args.logging
    if(args.market_type):
        config['market_type'] = args.market_type

    print "Going to run with the following configuration:"
    pprint.pprint(config)

    if not args.no_confirm:
        answer = raw_input('Do you want to continue? [Y/n]\r\n')

        if(answer == 'n'):
            exit(0)
    else:
        print "No confirm detected, continuing."

    logging.basicConfig(level=config["logging_level"])

    sim = Simulation(config,args.aggregate_id,args.name)
    sim.run()
