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
import time
from typing import Dict, Generator
import yaml
# from datsim import config
from helpers import mk_outdir, mk_param_list_list,print,sprob_from_csv,init_from_csv
from getClass import Config
from print_func import dfPrint
from paramLists import (
  staticPar,
  plSubset,
  plDefault,
  parLists,
  plNoStorm,
  plNSTest,
  plControl,
  plCtlStorm,
  # parLists2,
  plTest, 
  plTest2,
  plCtlTest,
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
print(f"rsettings: {atype=}")
# mcType = os.environ.get('mcTypeR')
# debug=False
# use_pwrong=False
# nWeeks = 2
initFromFile = True
stormFromFile = True
# np.set_printoptions(precision=7) # NOTE this doesn't work outside np arrays?
printSettings = os.environ.get('printset')
otherVal = os.environ.get('otherVal')
# if debug: print(f"\t\t>>{printSettings=}")

# print("> run getopt -> ", end=" ")
try:
  opts,args = getopt.gnu_getopt(sys.argv[1:],"ht:do:",["Help", "Type", "Debug","Options"])
except getopt.error as err:
  print(str(err))
  sys.exit(2)
# print(f"{opts=} {args=}")

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
  if debug: msg=f"\t\t>=> config, type {ctype}; converted to class: {my_conf}"
  # return [my_conf,msg]
  return my_conf

# atype="default" ## can be changed with CL args, below
optval="none"
if len(opts)>0:
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

# tests = ['norm','range', 'test2','debug', 'small','small2','xtrastorm', 'fixedtest', 'nstest']
# +> ANALYSIS TYPE - GROUPS:
tests = ['norm','range','test2','small','small2','xtrastorm','test100',
         'testmaxmin','testctrl','fixedtest','nstest1','nstest2','nostormtest',
         'test500','setn','snrange']
fullList = ['nostorm','full','supp','subset',]
ctrlList = ['control','ctlstorm']
rangeList = ['range','snrange']
 ## don't necessarily have to have different param set for each atype
 ## can just set some values in the function

#test2 is control vals; range is extremes at either end of param vals
# ctrlList = ['control', 'test2']

def choose_config(atype,debug=False):
  if atype in tests:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml","test")
    msg = f"\t\t|>Config-TEST mode <{config.testing=}>"
# elif atype == "full":
  elif atype in ctrlList:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml","ctrl")
    msg = f"\t\t|>Config-control <{config.testing=}>"
  elif atype in fullList:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml","full")
    msg = f"\t\t|>Config-full <{config.testing=}>"
    # msg=("\t\t|>Config-FULL mode", end=" ")
# elif atype == ""
  else:
    # msg=("\t\t|>using default config", end=" ")
    # config = load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)
    config = load_config("/home/wodehouse/Projects/sim_model/config.yaml")
    msg = f"\t\t|>Config-default <{config.testing=}>"
  return config, msg

def choose_parlist(atype, config,debug=False):
  match atype:
    case "ctlstorm":
      pLists = plCtlStorm
      msg="\t\t|>not testing; using **control-storm** param lists"
    case "full10rep":
      pLists = parLists
      msg="\t\t|>not testing; using full param lists"
    case "full":
      pLists = parLists
      msg="\t\t|>not testing; using full param lists"
    case "control":
      msg="\t\t|>full w/control values"
      pLists=plControl
      config.stormFate = 2
      config.rangeVar = 1 ## probSurv is range of values
    case "predict":
      msg="\t\t|> full WITH PREDICTIONS"
      config.predSave="mean"
      config.stormFate=2
      pLists=plSubset
    case "nostorm":
      msg=("\t\t|>run with no storms")
      pLists=plNoStorm
    case "nostormtest":
      msg=("\t\t|>run with no storms")
      pLists=plNSTest
    case "nstest1":
      msg=("\t\t|>run with no storms")
      pLists=plNSTest
      # pLists['propMC'] = list(np.linspace(0.0,0.4,9)),
      #NOTE: why does this not work here if it works below??
      # pLists['propMC'] = list(np.linspace(0.0,0.5,11))
      # msg=msg+f"\t\t\t|> ***OVERRIDE: using {pLists["propMC"]=} & {pLists["propUnk"]=} "
    case "nstest2":
      msg=("\t\t|>run with no storms")
      pLists=plNSTest
      # pLists['propUnk'] = list(np.linspace(0.0,0.4,9)),
      # pLists['propUnk'] = list(np.linspace(0.0,0.5,11))
      # msg=msg+f"\t\t\t|> ***OVERRIDE: using {pLists["propMC"]=} & {pLists["propUnk"]=} "
    case "supp":
      msg=("\t\t|>run with supplemental param sets")
      # pLists=plSupp
      pLists=parLists
    case "norm":
      pLists = plTest
      # debug = True
      msg=("\t\t|>using test values.")
    case "test100":
      pLists = plTest # debug = True
      msg=("\t\t|>using test values.")
    case "test200":
      pLists = plTest # debug = True
      msg=("\t\t|>using test values.")
    case "test500":
      pLists = plTest # debug = True
      msg=("\t\t|>using test values.")
    case "setn":
      pLists = plTest # debug = True
      msg=("\t\t|>using test values.")
    case "snrange":
      pLists = plTest # debug = True
      msg=("\t\t|>using test values.")
    case "testmaxmin":
      pLists = plTest # debug = True
      msg=("\t\t|>using test values.")
    case "test2":
      pLists=plTest2
      # initFromFile=False
      msg=("\t\t|>testing w/chosen values")
    case "testctrl":
      pLists=plCtlTest
      # initFromFile=False
      msg=("\t\t|>testing with control vals")
    case "range":
      msg="\t\t|>range of values"
      pLists=plTestRange
    case "small":
      msg="\t\t|>small set of values"
      pLists=plSmall
    case "small2":
      msg="\t\t|>small number of nests, no evidence decay"
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
      msg="\t\t|>no type provided; using default param lists"

  # if atype=="range" and config.rangeVar==1:
  if (atype=="testmaxmin"):
    # pLists['probSurv'] = [0.96,0.98]
    pLists['stormFrq'] = [0, 4]
    pLists['obsFreq'] = [3,7]
    # pLists['hatchTime'] = [16,28]
    pLists['stormFate'] = [True,False]
    pLists['numNests'] = [250,500]

    msg= msg + f"\t\t\t|> using max and min values for some params"

  # if  atype=="range" and config.rangeVar==1:
  if (atype=="control" or atype=="testctrl"):
    # pLists['probSurv'] = np.round(np.linspace(0.88,0.99, 12),2).tolist() 
    pLists['probSurv'] = np.round(np.linspace(0.89,0.98, 10),2).tolist() 
    # pLists['discProb'] = [1.0]
    # pLists['decayRate'] = [0.0, 0.08]
    msg= msg + f"\t\t\t|> using a range of values for probSurv"

  # if  atype=="range" and config.rangeVar==1:
  if  atype in rangeList and config.rangeVar==1:
    # pLists['probSurv'] = np.round(np.linspace(0.88,0.99, 12),2).tolist() 
    pLists['probSurv'] = np.round(np.linspace(0.9,0.99, 10),2).tolist() 
    # pLists['probSurv'] = np.round(np.linspace(0.91,0.99, 9),2).tolist() 
    msg= msg + f"\t\t\t|> using a range of values for probSurv"
  elif atype in rangeList and config.rangeVar==2:
    print(list(np.linspace(0.02,0.2, 19)))
    # pLists['decayRate'] = list(np.linspace(0.02,0.2, 19)),  ## this one is class 'tuple'; propMC isn't??
    pLists['decayRate'] = np.round(np.linspace(0.02,0.2, 19), 2).tolist()  ## this one is class 'tuple'; propMC isn't??
    print(f"{type(pLists['decayRate'])=}")
    msg= msg + f"\t\t\t|> using a range of values for decayRate"
  elif atype in rangeList and config.rangeVar==3:
    pLists['discProb'] = np.round(np.linspace(0.7,1.0, 6),2).tolist() 
    msg= msg + f"\t\t\t|> using a range of values for discProb"
    
  elif atype in rangeList and config.rangeVar==4:
    pLists['stormFrq'] = [0,1,2,3,4,5]
    msg= msg + f"\t\t\t|> using a range of values for stormFrq"
  elif atype in rangeList and config.rangeVar==5:
    # pLists['obsFreq'] = [1,2,3,4,5,6,7] ## obs int = 1 too many obs for MCMC
    pLists['obsFreq'] = [3,4,5,6,7] ## obs int = 1 too many obs for MCMC
    msg= msg + f"\t\t\t|> using a range of values for obsFreq"
  if atype=="supp":
    # pLists["decayRate"] = [0.1,0.12]
    # pLists["stormFrq"] = [5]
    pLists["stormDur"] = [3]
  if config.hTime==1:
    pLists["hatchTime"] = [16]
    msg= msg + f"\t\t\t|>!OVERRIDE - using {pLists["hatchTime"]} as hatch time"
  elif config.hTime==2:
    pLists["hatchTime"] = [20]
    msg= msg+f"\t\t\t|>!OVERRIDE - using {pLists["hatchTime"]} as hatch time"
  elif config.hTime==3:
    pLists["hatchTime"] = [28]
    msg= msg+f"\t\t\t|>!OVERRIDE - using {pLists["hatchTime"]} as hatch time"
  else:
    msg= msg+f"\t\t\t|> using {pLists["hatchTime"]} as hatch time"
  # if config.hTime==3:
  if config.stormFate==1:
    pLists["stormFate"] = [False]
    pLists["stormUnk"] = [True]
    msg= msg+f"\t\t>> override - using {pLists["stormFate"]} as storm fate"
  elif config.stormFate==2:
    pLists["stormFate"] = [True]
    pLists["stormUnk"] = [True]
    # print("using '2' as storm fate")
    msg=msg+f"\t\t>> override - using {pLists["stormFate"]} as storm fate"
  elif config.stormFate==3:
    pLists["stormFate"] = [False]
    # pLists["stormUnk"] = [True]
    pLists["stormUnk"] = [False]
    # print("using '2' as storm fate")
    # msg=msg+f"\n\t>> marking nests w/long final int unknown, but not storm final int"
    msg=msg+f"\t\t>> override - using {pLists["stormFate"]} as storm fate"
  else:
    msg=msg+f"\t\t\t|> using {pLists["stormFate"]} as storm fate"
  if config.numNests==1:
    # pLists['propMC'] = list(np.linspace(0.0,0.5,11))
    pLists['numNests'] = [250]
    print(f"{type(pLists['numNests'])=}")
    msg=msg+f"\t\t\t|> ***OVERRIDE to change numNests: using {pLists["numNests"]=}"
  elif config.numNests==2:
    # pLists['propUnk'] = list(np.linspace(0.0,0.5,11))
    pLists['numNests'] = [500]
    msg=msg+f"\t\t\t|> ***OVERRIDE to change numNests: using {pLists["numNests"]=}"
  elif config.numNests==3:
    # pLists['propUnk'] = list(np.linspace(0.0,0.5,11))
    pLists['numNests'] = [100]
    msg=msg+f"\t\t\t|> ***OVERRIDE to change numNests: using {pLists["numNests"]=}"
  elif config.numNests==4:
    # pLists['propUnk'] = list(np.linspace(0.0,0.5,11))
    pLists['numNests'] = [50]
    msg=msg+f"\t\t\t|> ***OVERRIDE to change numNests: using {pLists["numNests"]=}"
  else:
    msg=msg+f"\t\t\t|> using {pLists["numNests"]=}"

  if config.mcType==1:
    # pLists['propMC'] = list(np.linspace(0.0,0.5,11))
    # pLists['propMC'] = list(np.round(np.linspace(0.0,0.7,15),2))
    pLists['propMC'] = list(np.round(np.linspace(0.1,0.7,7),2))
    # pLists['discProb'] = [1.0]
    pLists['decayRate'] = [0.0]
    pLists['discProb'] = [1.0, 0.8]
    print(f"{type(pLists['propMC'])=}")
    msg=msg+f"\t\t\t|> ***OVERRIDE to change propMC: using {pLists["propMC"]=} & {pLists["propUnk"]=} "

  elif config.mcType==2:
    # pLists['propUnk'] = list(np.linspace(0.0,0.5,11))
    # pLists['propUnk'] = list(np.round(np.linspace(0.0,0.7,15),2))
    pLists['propUnk'] = list(np.round(np.linspace(0.1,0.7,7),2))
    pLists['decayRate'] = [0.0]
    pLists['discProb'] = [1.0, 0.8]
    # pLists['discProb'] = [1.0]
    msg=msg+f"\t\t\t|> ***OVERRIDE to change propUnk: using {pLists["propMC"]=} & {pLists["propUnk"]=} "

  else:
    msg=msg+f"\t\t\t|> using {pLists["propMC"]=} & {pLists["propUnk"]=} "

  # if config.other=="high_sdecay":
  if otherVal=="high_sdecay":
    pLists['decayStorm'] = [0.19]
    # pLists['hatchTime'] = [20]
    # pLists['probSurv'] = [0.966]
    msg=msg+f"\t\t\t|> using {pLists['decayStorm']=}"
  # if config.other=="setn":
  if otherVal=="setn":
    pLists['stormFrq'] = [0]
    pLists['hatchTime'] = [20]
    # pLists['probSurv'] = [0.966]
    msg=msg+f"\t\t\t|> preset nest fates - using {pLists['stormFrq']=}"
  ## this might work better:
  # if atype == nstest:

  # if atype=="small2":

  if atype in ["small", "testctrl"]:
    initFromFile=False
    stormFromFile=False


  if debug: print(msg)
  # NOTE could you collect messages and then print the msg so far on error?
  # NOTE that way, you can turn off printing in one place
  # return pLists
  # return [pLists, (msg)]
  #NOTE need to return as tuple
  return pLists, msg

def print_settings(config, atype, initFromFile, paramsArray, pListOut, confMsg):
  print("\n\t[*] [*] [*] [*] [*] [*] settings [*] [*] [*] [*] [*] [*] [*] [*] [*] \n")
  print(confMsg, end=" ")
  print(f"\t|>{atype=}", end=" ")
  print(f"\t|>{config.rngSeed=}", end=" ")
  print(f"\t|>{config.optimizer=}", end=" ")
  print(f"\t|>{initFromFile=}")
  # if atype in tests:
  #   print(f"\t\t|>Config-TEST mode <{config.testing=}>", end=" ")
  # elif atype in ctrlList:
  #   print(f"\t\t|>Config-control <{config.testing=}>", end=" ")
  # elif atype in fullList:
  #   print(f"\t\t|>Config-FULL <{config.testing=}>", end=" ")
  # else:
  #   print("\t\t|>Config-default <{config.testing=}>", end=" ")
  # print(pListOut[0])
  # print(pListOut[1])

  print(f"\t\t|>SAVING: nest data? {config.saveNData}", end=" ")
  print(f">prediction data? {config.predSave}", end=" ")
  print(f">model coefficients? {config.coefSave}", end=" ")
  print(f">nest obs matrix? {config.obsSave}")
  # print(
  #       f"\n|>|>|>{len(paramsArray)} param sets x {config.nreps} reps ="
  #       f" {len(paramsArray)*config.nreps} total rows"
  #       )
  # if config.debug>=5:
  #   print("\t\t|>|>param sets:")
  #   print(pArrList)


# print(f"{atype=}\t\t{initFromFile=}")
# NOTE needs to be saved in dir for specific script instance
# with open(configFile, "wb") as f:
#   pickle.dump(config,f)
configOut = choose_config(atype)
config = configOut[0]
debug = config.debug
confMsg = configOut[1]
if atype=="nstest1":
  config.nreps=100
  config.mcType=1
if atype=="nstest2":
  config.nreps=100
  config.mcType=2
if atype=="full10rep":
  config.nreps=10
# if atype=="test2":
#   config.nreps=100
if atype=="test100":
  config.nreps=100
if atype=="test200":
  config.nreps=200
if atype=="test500":
  config.nreps=500
if atype=="small2":
  config.debugLL=4
# if atype=="setn":
if atype in ["setn","snrange"]:
  config.other="setn"

# if atype=="smalln":
print(f"{config=}")


pListOut = choose_parlist(atype, config, debug=False)
pLists = pListOut[0]
# print(f"{pListOut=}")
pListMsg = pListOut[1]
fullMsg = confMsg + pListMsg
# print(f"{pListMsg=}")
odir  = mk_outdir(now_short, con=config)
# print(f"{odir=}")
# paramsArray = mk_param_list_list(parL=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=config.debug)
# rng = np.random.default_rng(seed=config.rngSeed)
# stormDat      = sprob_from_csv(config.stormInit,debug=True)
# initDat      = init_from_csv(config.stormInit,debug=True)
# config.mcType = int(mcType)

coniInit = [2,11,9,4,22,18,14,7,11,2,6,2,20]
coniWeek = np.arange(4,16,1)
# leteInit = [4,74,67,48,51,33,42,41,34,28,36,10,7,96] ## 96=NAs
# leteWeek = np.arange(3,16,1)
leteInit = [4,74,67,48,51,33,42,41,34,28,36,10,7]
leteWeek = np.arange(3,15,1)
wiplInit = [1,7,18,7,8,6,1,5,11,3,1,6]
wiplWeek = np.arange(1,12,1)
# inits = [1,7,22,83,86,63,56,60,71,58,42,39,38,16,9]
# inits = [1,7,25,94,80,62,57,63,67,56,39,41,35,16,8]
# weeks = np.arange(1,16,1)
inits=leteInit
weeks=np.arange(0,13,1)
# print("\tusing LETE init dates")
fullMsg = fullMsg + "\n\t\t\t>> using LETE init dates"

initProb = inits / np.sum(inits) # make them into probabilities again
time.sleep(3) ## pause before printing
# init_weeks = np.arange(14,29,1)
# weekStart = (init_weeks * 7) - 90 # why minus 90?
initWeek = weeks * 7
initWeek = initWeek.astype(int)
initDat = [initProb, initWeek]
# if debug>=1: print(f"{type(paramsArray)=} ; {type(staticPar)=}")

# stormProb = [0.006,0.019,0.044,0.025,0.069,0.044,0.050,0.044,0.025,0.038,
# stormProb = [0.019,0.044,0.025,0.069,0.044,0.050,0.044,0.025,0.038,
#              0.050,0.057,0.031,0.069,0.025]
# stormNum = [2,2,2,3,5,8,7,4,7,5,3,14,5,5,5]
# stormNum = [1,2,2,3,5,8,7,4,7,5,3,14,5,5,5]
# stormNum = [2,2,1,4,5,3,3,4,4,5,5,1,9,5,4,16]
# stormNum = [0,1,2,1,4,5,3,3,4,4,5,5,2,9,5,2]
stormNum = [1,4,5,3,3,4,4,5,5,2,9,4,2]
# stormNum = [1,2,3,5,8,7,4,7,5,3,14,5,5,5]
             # 0.050,0.057,0.031,0.069,0.025,0.069,0.038,0.082,0.031,0.038,
             # 0.063,0.050,0.031]
stormProb = stormNum/np.sum(stormNum)
# dfPrint(np.array([stormProb]), names=list(np.arange(23)))
# stormWeek = np.arange(23)
# stormWeek = np.arange(1,18,1)
# stormWeek = np.arange(2,17,1)
# sitormWeek = np.arange(0,16,1)
stormWeek = np.arange(1,14,1)
weekStart = stormWeek*7
stormDat = [stormProb, weekStart]
paramsArray = mk_param_list_list(parL=pLists,pStatic=staticPar, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=True)
pArrList = mk_param_list_list(parL=pLists,pStatic=staticPar, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False, listRet=True)

if config.nreps>20 and config.nreps<51:
  # print("nreps>20")
  if config.debug>=3: config.debug=1
  ## NOTE now these should all be capped at the 'debug' value, which can also be changed w/arg
  # if config.debugObs>=2: config.debugObs=1
  # if config.debugNests>=2: config.debugNests=1
  # if config.debugDSR>=2: config.debugDSR=1
  # if config.debugLogEx>=2: config.debugLogEx=1
  # if config.debugLL>=2: config.debugLL=1
  # if config.debugFlood>=2: config.debugFlood=1
  # if config.debugSummary>=2: config.debugSummary=1
elif config.nreps>50:
  # print("nreps>50")
  config.debug=0
  # config.debugObs=0
  # config.debugNests=0
  # config.debugDSR=0
  # config.debugLogEx=0
  # config.debugLL=0
  # config.debugFlood=0
  # config.debugSummary=0

# print(config)
vary=""
# if atype=="range":
# if atype=="range" or atype=="test2":
# if atype in ["range", "test2","snrange"]:
if atype in ["range", "testctrl","snrange"]:
  config.nreps=100
  config.debug=0
  # config.debugNests=0
  # config.debugLogEx=0
  # config.debugDSR=0
  # config.debugObs=0
  print(f"*** OVERRIDE atype=range; {config.debug=} {config.debugNests=} {config.debugObs=} {config.nreps=}")
  print(f"{type(pLists)=}{pLists=}")
  vary = [k for k,v in pLists.items() if len(v)>2]
  print("vary the levels of ", vary)

if printSettings == 'TRUE':
  # print_settings(config, atype, initFromFile, paramsArray, pListMsg, confMsg )
  if debug>=0: print(f"\n\t\tinitProb by week & week start day [{len(initProb)=}]:")
  if debug>=0: dfPrint(np.array([initProb]), names=list(initWeek))
  if debug>=0: print(f"\t\tstormProb by week & week start day [{len(stormProb)=}]:")
  if debug>=0: dfPrint(np.array([stormProb]), names=list(weekStart))
  print_settings(config, atype, initFromFile, paramsArray, pListMsg, fullMsg )
  

if __name__ == "__main__":
  # print_settings(config, atype, initFromFile, paramsArray, pListMsg, confMsg )
  print_settings(config, atype, initFromFile, paramsArray, pListMsg, fullMsg )

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
