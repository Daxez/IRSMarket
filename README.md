# Interest rate swap market simulation

This is (most of) the code used to create my master thesis on Self-organized criticality in an Interest rate swap market. 
There are a lot of files but there are a few that contain the core of running the simulations.

## Purpose

The main purpose of this project is to run a simulation of _N_ nodes that have a balance _B_ that the nodes want to keep balanced (at 0) while that balance is perturbed by a random normal variable every timestep. The hedging occurs if two nodes with an oposite balance go into contract with eachother for a certain amount of time (an IRS). Their balance is then corrected with a IRS value. When a node reaches a threshold _T_ they default. IRSs are removed on default and this allows the default to spread.

## How to use this code

Before starting I suggest setting up a virtualenv for the project and installing the right dependencies using the _freeze.pip_ file (it's quite specific). Also, if you want to save the data set up an mysql database, add your connection info to _data/models.py_ line 189 (not all models are applicable anymore) and run that to set up the database. It uses `sqlalchemy` to add data to the database and plain sql queries in the code to retrieve. 

Most of the data that is collected is outputed to _.bin_ files using `pickle`. We do this because for running the main simulation we use *pypy* to gain some more speed. Not all libraries were implemented when we started the project, and updating did not seem useful, so it still works in that way. After simulations are run using pypy you can use the meriad of graphing scripts that are in there (some more useful than others) to plot the data (using actual python and matplotlib).

Some of the graphing methods are implemented two times. Most of the graphing methods are copied from one file to another. Implementing dynamic graphing for just this projected seemed a bit of a stretch, so copying saved some time. 

These files can be run using pypy and are basically the core of the simulation (all the other stuff is either old or plotting tools)
  * `quick.py`
    contains the main simulation and a lot of methods that administer certain properties during the simulation (connectedness, risk, node information, avalance progression info, etc.) The whole file seems a bit daunting but 90% of the actual simulation is in the default and the run methods of the simulation class.
  * `quick_sweep.py`
    is a file that you can use for running multiple simulations for different configurations. An elaborate number of bools is set in the first few lines of the main entry point of the file, to control what you are going to administer during the simulations.

  All other files are either old or plotting something. Some plots that I used:

  * `hump_error_heat.py` Heatmap of the weight in the hump of a default cascade distribution (uses analyse_hump.py)
  * `alpha_boxplot.py` boxplots for the alpha of the power law part
  * `alpha_heat.py` heatmap for alpha
  * `show.py shows` the distributions for aggregate_ids
  * `show_by_params.py` shows distributions based on some configurations (also on ranges)
  * `time_series.py` to show risk time series (absolute total risk in the system)
  * `type_heat.py` creates a heatmap of types if you filled those in the database

  A lot of the other graphing scripts were used to gain insight of course, but it has been a while and I am not sure if they all still work.

  In the folder _cpp_ there is a C++ implementation of the model that is way faster than the python implementation. I do not guarantee it is correct (I have not used it in my Thesis, but it was fun to see what it did anyway). It outputs a json format of all avalance sizes and their frequency:

  ```
  {
      '1': 2001,
      '2': 1904,
      '34': 12
  }
  ```

  It can be further optimized by seperating random number generation and the simulation from eachother, or by generating random numbers in some background tasks in parallel.

  There is no make file, but a simple gcc compile command should do the trick.  

## Notes

  I have tried to clean it up as much as possible (deleting some stuff that was really old), but due to the "I am only writing this code for me" nature of this project it has all become a bit tied and I don't really see what can be deleted anymore. Probably some files like _bank.py_ _irs.py_ _market.py_ _test.py_ _app.py_ are really obsolete (first implementation of the model using a way too object oriented approach for the problem).

  There are some absolute paths in the files (e.g. /home/..../Programming) that you want to get rid of, or replace according to your own folder structure. I removed the user part of the paths for obvious reasons.

  There are some aggregate_ids hardcoded in the files that don't have any meaning in your runs.

## Outputs to the database:

  The main tables for the database are the _run_ table and the _default\_aggregate_ table. The run table contains a run Id and an aggregated id (if you decide you wnat several runs on the same configuration and their frequencies added in the default aggregate table). The _default\_aggregate_ table contains the size of a default cascade and its frequencies. Obsolete are:
  * bank
  * bank_default
  * swap
  
  The following tables are filled out using scripts but not the simulation (e.g. _populate\_weight.py_):
  * aggregatehumpWeight
  * aggregate_distribution
  * aggregate_powerlaw
  * aggregate_type