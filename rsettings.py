# -----------------------------------------------------------------------------
#  SETTINGS 
# -----------------------------------------------------------------------------

from datetime import datetime
import csv
import itertools
import numpy as np
import pickle
import os
import getopt
from pathlib import Path
import sys
from typing import Dict, Generator
import yaml
# from datsim import config
from helpers import mk_outdir, mk_param_list_list
from getClass import Config
from paramLists import (
  staticPar,
  plSubset,
  plDefault,
  parLists,
  plNoStorm,
  plNSTest,
  plControl,
  # parLists2,
  plTest, 
  plTest2,
  plTestFlood,
  plDebug,
  plSmall,
  plSupp,
  plTestRange,
)
# print(type(staticPar))

now_long  = datetime.today().strftime('%m%d%Y_%H%M%S')
now_short  = datetime.today().strftime('%Y%m%d')
# mess = "" #+> message to be printed at beginning, explaining 
#           #+> purpose of test or hwatever

# atype=""
# atype_r = os.environ.get('atypeR')
atype = os.environ.get('atypeR')
debug=False
use_pwrong=False
nWeeks = 2
initFromFile = True
stormFromFile = True
np.set_printoptions(precision=3) # NOTE this doesn't work outside np arrays?

try:
  opts,args = getopt.gnu_getopt(sys.argv[1:],"ht:do:",["Help", "Type", "Debug","Options"])
except getopt.error as err:
  print(str(err))
  sys.exit(2)

def load_config(fpath, ctype="default", debug=False):
  """
    load config from fpath (my_conf)
    ctype = config type ('full', 'test', 'default')

    will create Config from my_conf[ctype]
  """
  with open(fpath, "r") as cfg:
      my_conf = yaml.safe_load(cfg) # if debug: print(">=> config:\n", my_conf)
  my_conf = Config(**my_conf[ctype]) ## select correct config set
  # if debug: print(f"\t\t>=> config, type {atype}; converted to class:", my_conf)
  if debug: print(f"\t\t>=> config, type {ctype}; converted to class:", my_conf)
  return my_conf

# atype="default" ## can be changed with CL args, below
optval="none"
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
  elif arg in ("-o", "--Options"):
    optval=val

# tests = ['debug', 'control','testing', 'storm', 'fixedtest', 'nstest']

# +> ANALYSIS TYPE - GROUPS:
tests = ['norm','range', 'test2','debug', 'small','small2','xtrastorm', 'fixedtest', 'nstest']
fullList  = ['nostorm', 'full', 'supp', 'logexp','subset']
ctrlList = ['control']

#test2 is control vals; range is extremes at either end of param vals
# ctrlList = ['control', 'test2']

def choose_config(atype):
  if atype in tests:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml", "test")
    print("\t\t|>Config-TEST mode:", config.testing, end=" ")
# elif atype == "full":
  elif atype in ctrlList:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml", "ctrl")
  elif atype in fullList:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml", "full")
    print("\t\t|>Config-FULL mode", end=" ")
# elif atype == ""
  else:
    print("\t\t|>using default config", end=" ")
    # config = load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)
    config = load_config("/home/wodehouse/Projects/sim_model/config.yaml")
  return config

# if debug:
#   # config.debug = True
#   config.debug = 2
#   print("\t\t|>Config-debug (using default val):", config.debug, end=" ")
#
# #--- OTHER SETTINGS ------------------------------------------------------------
# if config.useWin:
#   config.likeDir = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output"
#   config.stormInit = "C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv" 
#   config.fnUnique   = False
#

# def other_settings(config, pLists):
  # print(f"\t|>{config.rngSeed=}", end=" ")
  # rng = np.random.default_rng(seed=config.rngSeed)
  #
  # print(f"\t|>{config.optimizer=}", end=" ")
  # odir  = mk_outdir(now_short, con=config)
  # paramsArray = mk_param_list_list(parList=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False)
  # pArrList = mk_param_list_list(parList=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False, listRet=True)

def choose_parlist(atype, config):
  match atype:
    case "full":
      pLists = parLists
      print("\t\t|>not testing; using full param lists",end=" ")
    case "control":
      print("\t\t|>full w/control values", end=" ")
      pLists=plControl
      config.stormFate = 2
    case "predict":
      print("\t\t|> full WITH PREDICTIONS")
      config.predSave="mean"
      config.stormFate=2
      pLists=plSubset
    case "nostorm":
      print("\t\t|>run with no storms")
      pLists=plNoStorm
    case "nstest":
      print("\t\t|>run with no storms")
      pLists=plNSTest
    case "supp":
      print("\t\t|>run with supplemental param sets")
      pLists=plSupp
    case "norm":
      pLists = plTest
      # debug = True
      print("\t\t|>using test values.")
    case "test2":
      pLists=plTest2
      # initFromFile=False
      print("\t\t|>testing with control vals")
    case "small":
      print("\t\t|>small set of values", end=" ")
      pLists=plSmall
      # initFromFile=False
      # staticPar['brDays'] = 25
    ## WHY DOES THIS BREAK EVERYTHING??? EVEN AFTER YOU STOP UING IT??
    # case "small2":
    #   print("\t\t|>small set of vals & num breeding days & shorter inc time")
    #   pLists=plSmall
    #   staticPar['brDays'] = 25
    #   pLists['stormFrq'] = [0]
    #   pLists['hatchTime'] = [10,12]
    #   initFromFile=False
    case _:
      pLists = plDefault # don't need to update any settings if not testing?
      print("\t\t|>no type provided; using default param lists", end=" ")

  if config.hTime==0:
    pLists["hatchTime"] = [16]
    print(f"\t\t\t|>!OVERRIDE - using {pLists["hatchTime"]} as hatch time", end=" ")
  if config.hTime==1:
    pLists["hatchTime"] = [20]
    print(f"\t\t\t|>!OVERRIDE - using {pLists["hatchTime"]} as hatch time", end=" ")
  if config.hTime==2:
    pLists["hatchTime"] = [28]
    print(f"\t\t\t|>!OVERRIDE - using {pLists["hatchTime"]} as hatch time", end=" ")
  else:
    print(f"\t\t\t|> using {pLists["hatchTime"]} as hatch time", end=" ")
  # if config.hTime==3:
  if config.stormFate==0:
    pLists["stormFate"] = [False]
    print(f"\t\t>> override - using {pLists["stormFate"]} as storm fate", end=" ")
  elif config.stormFate==1:
    pLists["stormFate"] = [True]
    # print("using '2' as storm fate")
    print(f"\t\t>> override - using {pLists["stormFate"]} as storm fate", end=" ")
  else:
    print(f"\t\t\t|> using {pLists["stormFate"]} as storm fate", end=" ")
  if config.mcType==0:
    pLists['propMC'] = list(np.linspace(0.0,0.5,11))
    print(f"\t\t\t|> ***OVERRIDE to change propMC: using {pLists["propMC"]=} & {pLists["propUnk"]=} ", end=" ")
  elif config.mcType==1:
    pLists['propUnk'] = list(np.linspace(0.0,0.5,11))
    print(f"\t\t\t|> ***OVERRIDE to change propUnk: using {pLists["propMC"]=} & {pLists["propUnk"]=} ", end=" ")
  else:
    print(f"\t\t\t|> using {pLists["propMC"]=} & {pLists["propUnk"]=} ", end=" ")

  return pLists
if atype in ["small", "test2"]:
  initFromFile=False
  stormFromFile=False

# def edit_config(pLists, config):
  # if config.stormFate==0:
  #   pLists["stormFate"] = [False]
  #   print(f"\t\t>> override - using {pLists["stormFate"]} as storm fate", end=" ")
  # elif config.stormFate==1:
  #   pLists["stormFate"] = [True]
  #   # print("using '2' as storm fate")
  #   print(f"\t\t>> override - using {pLists["stormFate"]} as storm fate", end=" ")
  # else:
  #   print(f"\t\t\t|> using {pLists["stormFate"]} as storm fate", end=" ")

def print_settings(config, atype, paramsArray, scriptName = "lexp.R"):
  print("\n\t[*] [*] [*] [*] [*] [*] settings [*] [*] [*] [*] [*] [*] [*] [*] [*] ")
  print(f"|>{atype=}", end=" ")
  print(f"\t|>{config.rngSeed=}", end=" ")
  print(f"\t|>{config.optimizer=}", end=" ")
  print(f"\t|>{initFromFile=}")

  print(f"|>saving nest data? {config.saveNData}", end=" ")
  print(f"|>saving prediction data? {config.predSave}", end=" ")
  print(f"|>saving model coefficients? {config.coefSave}", end=" ")
  print(f"|>saving nest obs matrix? {config.obsSave}")
  print(
        f"\n|>|>|>{len(paramsArray)} param sets x {config.nreps} reps ="
        f" {len(paramsArray)*config.nreps} total rows"
        )


# print(f"{atype=}\t\t{initFromFile=}")
# NOTE needs to be saved in dir for specific script instance
# with open(configFile, "wb") as f:
#   pickle.dump(config,f)
config = choose_config(atype)
pLists = choose_parlist(atype, config)
odir  = mk_outdir(now_short, con=config)
paramsArray = mk_param_list_list(parL=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False)
pArrList = mk_param_list_list(parL=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False, listRet=True)
if config.debug>=2:
  print("|>|>param sets:")
  print(pArrList)
rng = np.random.default_rng(seed=config.rngSeed)

# print(f"\t|>{config.rngSeed=}", end=" ")
# print(f"\t|>{config.optimizer=}", end=" ")
# dtime = datetime.today().strftime('%d %b %Y @ %H:%M')
# # print("\n\n<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
# print("\n\n+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
# print(" + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
# # print(f"\n <> <> <> <> <> <> <> <> datsim.py - {dtime} <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
# print(f"\n <> <> <> <> <> <> <> <> {scriptName} - {dtime} <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
# print("\n\t[*] [*] [*] [*] [*] [*] settings [*] [*] [*] [*] [*] [*] [*] [*] [*] ")
#
# print(f"|>saving nest data? {config.saveNData}", end=" ")
# print(f"|>saving prediction data? {config.predSave}", end=" ")
# print(f"|>saving model coefficients? {config.coefSave}", end=" ")
# print(f"|>saving nest obs matrix? {config.obsSave}")
# print(
#       f"\n\t|>|>|>{len(paramsArray)} param sets x {config.nreps} reps ="
#       f" {len(paramsArray)*config.nreps} total rows"
#       )
#


# def set_settings(atype, config, debug=0):
#   """
#   """

# if use_pwrong == False:
#   print("\t\t|>not using pWrong", end=" ")
#   pLists["pWrong"]=[0]
#   # del pLists["pWrong"]
# else:
#   print("\t\t|>using pWrong", end=" ")

# print("\n\t|>output directory:", config.likeDir)



# print(
#     f"\n\t\t<>DEBUG VALUES<>",
#     f"\t {[]}"
#     )

# if optval == "saveobs":
#   print("\t\tNOT saving model output; saving nest obs matrix")
#   config.predSave = "none"
#   config.coefSave = "none"
#   config.obsSave = True

# if config.testing == "norm":
# if atype == "":
#   pLists = plDefault # don't need to update any settings if not testing?
#   print("\t\t|> no type provided; using default param lists", end=" ")
# elif atype == "full":
#   # config.nreps=400 # print("changed config values:",config.debug, config.nreps)
#   # config.nreps=1 # print("changed config values:",config.debug, config.nreps)
#   pLists = parLists
#   print("\t\t|>not testing; using full param lists",end=" ")
#   # global debug 
#   lf_suffix = "-full"
#   print("\t\t|>using test values. global debug = ", debug,end="")
# elif atype == "logexp":
#   pLists = parLists
#   print("\t\t|>not testing; using full param lists w/ logistic exposure",end=" ")
#   lf_suffix="logexp"
# elif atype == "norm":
#   # config.nreps=400 # print("changed config values:",config.debug, config.nreps)
#   # config.nreps=1 # print("changed config values:",config.debug, config.nreps)
#   pLists = plTest
#   # global debug 
#   debug = True
#   lf_suffix = "-test"
#   print("\t\t|>using test values. global debug = ", debug,end="")
# elif atype=="subset":
#   pLists = plSubset
#   print("\t\t|> using small subset of params")
# elif atype=="range":
#   pLists = plTestRange
#   lf_suffix="testd"
#   print("\t\t|> using daily observations")
# elif atype=="xtrastorm":
#   # config.nreps=10
#   # config.debugFlood=True
#   # config.debugObs=True
#   pLists = plTestFlood
#   # debug = True
#   lf_suffix = "-flood"
#   print("\t\t|>using storm test values. global debug = ", debug,end="")
# elif atype=="debug":
#   print("\t\t|>CHECK THE DEBUG VALUES!!",end="")
#   # config.nreps=10
#   # debug=True
#   pLists=plDebug
#   lf_suffix="-debug"
# # elif config.testing=="fixed":
# #   pLists=parLists2
# #   lf_suffix="-fixed"
# elif atype=="test2":
#   # config.nreps=50
#   pLists=plTest2
#   print("\t\t|> testing with control vals")
#   lf_suffix="-ctrl-test"
# elif atype=="nstest":
#   print("\t\t|>no storms-TEST", end="")
#   pLists=plNSTest
#   lf_suffix="-nostorm-test"
#
# elif atype=="nostorm":
#   print("\t\t|>no storms", end=" ")
#   pLists=plNoStorm
#   lf_suffix="-nostorm"
#
# elif atype=="control":
#   print("\t\t|>control values", end=" ")
#   # pLists=plTest2
#   pLists=plControl
#   lf_suffix="-control"
#
# elif atype=="small2": # breeding days = 20
#   print("\t\t|> small set of vals & num breeding days & shorter inc time")
#   pLists=plSmall
#   staticPar['brDays'] = 25
#   pLists['stormFrq'] = [0]
#   pLists['hatchTime'] = [10,12]
#   initFromFile=False
#
#   # pLists['stormFrq'] = [0]
#   lf_suffix="-small2"
#
# elif atype=="small":
#   print("\t\t|>small set of values", end=" ")
#   pLists=plSmall
#   lf_suffix="-small"
#
# elif atype=="supp":
#   print("\t\t|> run with supplemental param sets")
#   pLists=plSupp
#   lf_suffix="-supp"
# else:
#   # pLists = plDefault # <> don't need to update any settings if not testing?
#   # pLists = plDefault # EX: don't need to update any settings if not testing?
#   pLists = plDefault # ~ don't need to update any settings if not testing?
#   print("\t\t|>testing val invalid; using minimal param set")
#
#
#---- NEST MODEL PARAMETERS: ------------------------------------------------
#region-----------------------------------------------------------------------
# NOTE problem: fate-masking variable (storm activity) also leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
# staticPar = {'nruns': 1,
# These are the values that are passed to the Params class

# initDat=init_from_csv(storm_init) # this will evaluate after storm_init has been changed for wsl

