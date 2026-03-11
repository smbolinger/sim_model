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
from paramLists import staticPar, parLists, plNoStorm,plNSTest, parLists2, plTest, plTest2, plTestFlood, plDebug

dtime = datetime.today().strftime('%d %b %Y @ %H:%M')
atype=""
debug=False
use_pwrong=False

# print("\n\n<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
print("\n\n+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
print(" + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
print(f"\n <> <> <> <> <> <> <> <> datsim.py - {dtime} <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
print("\n[*] [*] [*] [*] [*] [*] settings [*] [*] [*] [*] [*] [*] [*] [*] [*] ")

try:
  opts,args = getopt.gnu_getopt(sys.argv[1:],"ht:d",["Help", "Type", "Debug"])
except getopt.error as err:
  print(str(err))
  sys.exit(2)

def load_config(fpath, debug=False):
  with open(fpath, "r") as cfg:
      my_conf = yaml.safe_load(cfg) # if debug: print(">=> config:\n", my_conf)
  my_conf = Config(**my_conf)
  if debug: print("\t>=> config, converted to class:", my_conf)
  return my_conf

for arg, val in opts:
  if arg in ("-h", "--Help"):
      print("\n-------------------------------------------------------------------------------------------------------",
             "\nsimdata_vect.py usage:\n\n",
            "[-t --Type] Choose mode w/ smaller # of nests and reduced # of params OR used fixed probs:\n",
            # "\t\t1.'norm' - moderate values; 2.'storm' - extremes of storm values;\n",
            "\t\t1.'nostorm'-no storms; 2.'test'-moderate values; 3.'storm'-extreme storm values;\n",
            "\t\t3.'fixed'-fixed values; 4.'fixedtest'-test fixed; 5.'no' (default); 6.'nstest'-test no storm\n\n",
            "[-d --Debug-general] Turn on simple/broad debugging statements? (Default:False)\n\n",
            "\n-------------------------------------------------------------------------------------------------------"
              )
      sys.exit()
  elif arg in ("-t", "--Type"):
      atype=val
  elif arg in ("-d", "--Debug-general"):
      debug = True

tests = ['test', 'storm', 'fixedtest', 'nstest']
if atype in tests:
  config=load_config("/home/wodehouse/Projects/sim_model/test-config.yaml")
  print("\t|>Config-TEST mode:", config.testing, end=" ")
else:
  print("\t|>using default config", end=" ")
  config = load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)

if debug:
  # config.debug = True
  config.debug = 2
  print("\t|>Config-debug (using default val):", config.debug, end=" ")

#--- OTHER SETTINGS ------------------------------------------------------------
if config.useWin:
  config.likeDir = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output"
  config.stormInit = "C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv" 
  config.fnUnique   = False

rng = np.random.default_rng(seed=config.rngSeed)
# if config.testing == "norm":
if atype == "":
  pLists = parLists # don't need to update any settings if not testing?
  print("\t|>not testing; using full param lists",end="")
elif atype == "norm":
  # config.nreps=400 # print("changed config values:",config.debug, config.nreps)
  config.nreps=1 # print("changed config values:",config.debug, config.nreps)
  pLists = plTest
  # global debug 
  debug = True
  lf_suffix = "-test"
  print("\t|>using test values. global debug = ", debug,end="")
elif atype=="storm":
  config.nreps=10
  config.debugFlood=True
  config.debugObs=True
  pLists = plTestFlood
  debug = True
  lf_suffix = "-flood"
  print("\t|>using storm test values. global debug = ", debug,end="")
elif atype=="debug":
  print("\t|>CHECK THE DEBUG VALUES!!",end="")
  config.nreps=10
  debug=True
  pLists=plDebug
  lf_suffix="-debug"
# elif config.testing=="fixed":
#   pLists=parLists2
#   lf_suffix="-fixed"
# elif config.testing=="fixedtest":
#   config.nreps=50
#   pLists=plTest2
#   lf_suffix="-fixed-test"
elif atype=="nstest":
  print("\t|>no storms-TEST", end="")
  pLists=plNSTest
  lf_suffix="-nostorm-test"
elif atype=="nostorm":
  print("\t|>no storms", end=" ")
  pLists=plNoStorm
  lf_suffix="-nostorm"
else:
  pLists = parLists # don't need to update any settings if not testing?
  print("\t|>testing val invalid; using full param lists")

if use_pwrong == False:
  print("\t|>not using pWrong")
  pLists["pWrong"]=[0]
  # del pLists["pWrong"]
else:
  print("\t|>using pWrong")

# print("\n\t|>output directory:", config.likeDir)

#---- NEST MODEL PARAMETERS: ------------------------------------------------
#region-----------------------------------------------------------------------
# NOTE the main problem is that my fate-masking variable (storm activity) also 
#    leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
# staticPar = {'nruns': 1,
# These are the values that are passed to the Params class

# initDat=init_from_csv(storm_init) # this will evaluate after storm_init has been changed for wsl
