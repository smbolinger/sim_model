# -----------------------------------------------------------------------------
#  SETTINGS 
# -----------------------------------------------------------------------------

from datetime import datetime
import csv
import itertools
import numpy as np
import os
import getopt
from pathlib import Path
import sys
from typing import Dict, Generator
import yaml
# from datsim import config
from getClass import Config
from paramLists import staticPar, parLists, parLists2, plTest, plTest2, plTestFlood, plDebug

dtime = datetime.today().strftime('%d %b %Y @ %H:%M')
print("\n\n<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
print(f"\n <> <> <> <> <> <> <> <> datsim.py - {dtime} <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
print("\n[*] [*] [*] [*] [*] [*] settings [*] [*] [*] [*] [*] [*] [*] [*] [*] ")

try:
    opts,args = getopt.gnu_getopt(sys.argv[1:],"ht:d",["Help", "Test", "Debug"])
except getopt.error as err:
    print(str(err))
    sys.exit(2)

def load_config(fpath, debug=False):
    with open(fpath, "r") as cfg:
        my_conf = yaml.safe_load(cfg) # if debug: print(">=> config:\n", my_conf)
    my_conf = Config(**my_conf)
    if debug: print("\t>=> config, converted to class:", my_conf)
    return my_conf

test=""
debug=False
for arg, val in opts:
    if arg in ("-h", "--Help"):
        print("\n-------------------------------------------------------------------------------------------------------",
               "\nsimdata_vect.py usage:\n\n",
              "[-t --Test] Choose testing mode w/ smaller # of nests and reduced # of params OR used fixed probs:\n",
              "\t\t1.'norm' - moderate values; 2.'storm' - extremes of storm values;\n",
              "\t\t3.'fixed' - use fixed values; 4.'fixedtest' - test of fixed; 5.'no' (default)\n\n",
              "[-d --Debug-general] Turn on simple/broad debugging statements? (Default:False)\n\n",
              "\n-------------------------------------------------------------------------------------------------------"
                )
        sys.exit()
    elif arg in ("-t", "--Type"):
        test=val
    elif arg in ("-d", "--Debug-general"):
        debug = True

if test:
    config=load_config("/home/wodehouse/Projects/sim_model/test-config.yaml")
    print("\t|>Config-TEST mode:", config.testing, end=" ")
else:
    print("\t|>no args passed; using default config", end=" ")
    config = load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)

if debug:
    config.debug = True
    print("\t|>Config-debug:", config.debug, end=" ")

#--- OTHER SETTINGS ------------------------------------------------------------
if config.useWin:
    config.likeDir = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output"
    config.stormInit = "C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv" 
    config.fnUnique   = False

rng = np.random.default_rng(seed=config.rngSeed)
# if config.testing == "norm":
if test == "":
    pLists = parLists # don't need to update any settings if not testing?
    print("\t|>not testing; using full param lists")
elif test == "norm":
    # config.nreps=400 # print("changed config values:",config.debug, config.nreps)
    config.nreps=1 # print("changed config values:",config.debug, config.nreps)
    pLists = plTest
    # global debug 
    debug = True
    lf_suffix = "-test"
    print("\t|>using test values. global debug = ", debug)
elif test=="storm":
    config.nreps=10
    config.debugFlood=True
    config.debugObs=True
    pLists = plTestFlood
    debug = True
    lf_suffix = "-flood"
    print("\t|>using storm test values. global debug = ", debug)
elif test=="debug":
    print("\t|>CHECK THE DEBUG VALUES!!")
    config.nreps=10
    debug=True
    pLists=plDebug
    lf_suffix="-debug"
# elif config.testing=="fixed":
#     pLists=parLists2
#     lf_suffix="-fixed"
# elif config.testing=="fixedtest":
#     config.nreps=50
#     pLists=plTest2
#     lf_suffix="-fixed-test"
else:
    pLists = parLists # don't need to update any settings if not testing?
    print("\t|>testing val invalid; using full param lists")

# print("\n\t|>output directory:", config.likeDir)

#---- NEST MODEL PARAMETERS: ------------------------------------------------
#region-----------------------------------------------------------------------
# NOTE the main problem is that my fate-masking variable (storm activity) also 
#      leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
# staticPar = {'nruns': 1,
# These are the values that are passed to the Params class

# initDat=init_from_csv(storm_init) # this will evaluate after storm_init has been changed for wsl
