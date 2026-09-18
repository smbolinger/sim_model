from datetime import datetime
import time
import csv
import functools
import itertools
import numpy as np
import pandas as pd
import os
import traceback
import warnings
import sys

from pathlib import Path
# import matplotlib.pyplot as plt
import pprint
import sys
from typing import Dict, Generator
import yaml
# from datsim import config
from getClass import Config, Params
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
# from rsettings import tests, fullList, ctrlList, rangeList
# from settings import config, rng
now = datetime.today().strftime('%H%M%S')
# debug = config.debug
# NOTE: maybe make an indent print function for strings? instead of typing \t all the time
# NOTE: I *THINK* maybe this should be functions that don't require any of my 
# NOTE:     other scripts (to avoid circular referencing or whatever)

## flush print buffer immediately to std.out (so not delayed)
print = functools.partial(print, flush=True)
# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS
# -----------------------------------------------------------------------------
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

def choose_config(atype,tests,ctrlList,fullList,debug=False):
  if atype in tests:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml","test")
    msg = f"\t\t|>Config-TEST mode <{config.testing=}>"
  elif atype in ctrlList:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml","ctrl")
    msg = f"\t\t|>Config-control <{config.testing=}>"
  elif atype in fullList:
    config=load_config("/home/wodehouse/Projects/sim_model/config.yaml","full")
    msg = f"\t\t|>Config-full <{config.testing=}>"
  else:
    config = load_config("/home/wodehouse/Projects/sim_model/config.yaml")
    msg = f"\t\t|>Config-default <{config.testing=}>"

  if atype in ["setn","snrange"]:
    config.other="setn"
    msg = msg + f"preset fate numbers: {config.other=}"
  if atype=="control":
      config.stormFate = 2
      config.rangeVar = 1 ## probSurv is range of values
  if atype=="nstest1":
    config.nreps=100
    config.mcType=1
    msg = msg + f"test no storm: {config.mcType=} {config.nreps=}"
  if atype=="nstest2":
    config.nreps=100
    config.mcType=2
    msg = msg + f"test no storm: {config.mcType=} {config.nreps=}"
  if config.nreps>20 and config.nreps<51:
    if config.debug>=3: config.debug=2
    msg = msg + f"! set debug val lower {config.debug=}"
  elif config.nreps>50:
    config.debug=0
    msg = msg + f"! set debug val lower {config.debug=}"
  if atype in ["range", "testctrl","snrange"]:
    config.nreps=100
    config.debug=0
    msg = msg + f"! set debug val lower {config.debug=}{config.nreps=}"
  return config, msg

def choose_parlist(atype, config,otherVal=[],debug=False):
  pLists = {} 

  match atype:
    case "full":     pLists = parLists
    case "ctlstorm": pLists = plCtlStorm
    case "control":  pLists=plControl
    case "nostorm":  pLists=plNoStorm
    case "nstest1":  pLists=plNSTest
    case "nstest2":  pLists=plNSTest
    case "norm":     pLists = plTest
    case "setn":     pLists = plTest # debug = True
    case "snrange":  pLists = plTest # debug = True
    case "test2":    pLists=plTest2
    case "testctrl": pLists=plCtlTest
    case "range":    pLists=plTestRange
    case "small":    pLists=plSmall
    case "small2":   pLists=plSmall
    case _:          pLists = plDefault # don't need to update any settings if not testing?

  msg = f"\tPARAM LISTS = \n\t\t{pLists}"
  rangeList = ['range','snrange']
  # if  atype=="range" and config.rangeVar==1:
  if (atype=="control" or atype=="testctrl"):
    pLists['probSurv'] = np.round(np.linspace(0.89,0.98, 10),2).tolist() 
    msg= msg + f"\t\t\t|> using a range of values for probSurv"

  if  atype in rangeList and config.rangeVar==1:
    pLists['probSurv'] = np.round(np.linspace(0.9,0.99, 10),2).tolist() 
    msg= msg + f"\t\t\t|> using a range of values for probSurv"
  elif atype in rangeList and config.rangeVar==2:
    print(list(np.linspace(0.02,0.2, 19)))
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
    pLists['propMC'] = list(np.round(np.linspace(0.1,0.7,7),2))
    pLists['decayRate'] = [0.0]
    pLists['discProb'] = [1.0, 0.8]
    print(f"{type(pLists['propMC'])=}")
    msg=msg+f"\t\t\t|> ***OVERRIDE to change propMC: using {pLists["propMC"]=} & {pLists["propUnk"]=} "
  elif config.mcType==2:
    pLists['propUnk'] = list(np.round(np.linspace(0.1,0.7,7),2))
    pLists['decayRate'] = [0.0]
    pLists['discProb'] = [1.0, 0.8]
    msg=msg+f"\t\t\t|> ***OVERRIDE to change propUnk: using {pLists["propMC"]=} & {pLists["propUnk"]=} "
  else:
    msg=msg+f"\t\t\t|> using {pLists["propMC"]=} & {pLists["propUnk"]=} "

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

  vary = [k for k,v in pLists.items() if len(v)>2]
  if debug: print(msg)
  # if debug: print("\t>> param lists:", pLists)
  # NOTE could you collect messages and then print the msg so far on error?
  # NOTE that way, you can turn off printing in one place
  # return pLists
  # return [pLists, (msg)]
  #NOTE need to return as tuple
  return pLists, msg, vary
  # return pLists, msg, config

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

def centerDat(dat):
  """
    ARGS
      dat should be a numpy array
  """
  print("centering")
  mu = np.mean(dat)
  return dat - mu


def expDecay(n0, k, t):
  """
  Exponential decay function.

  n0 = initial value
  k  = rate of decay
  t  = time
  """
  # return(n0 * (1-lam) ** t)
  return n0 * np.exp(-k * t) 

def searchSorted2(a, b):
  """Get the index of where b would be located in a
  If bis in a, then return the index of that value instead of the next value
  """
  #out = np.zeros(a.shape)
  out = np.zeros((a.shape[0], len(b)))
  # out2 = np.zeros((a.shape[0], len(b)))
  for i in range(len(a)):
    #out[i] = np.searchsorted(a[i], b[i])
    #print("sorted search of\n", b, "within\n", a[i])
    # if debug: print(">> sorted search of", b, "within", a[i])
    out[i] = np.searchsorted(a[i], b)
    # if debug: print("sorted search of\n", b, "within\n", a[i], ":\n", out, out.shape)
    # if out[i] in b: out2[]
    # if debug: print("sorted search of\n", b, "within\n", a[i], "after accounting for exact match:\n", out, out.shape)
    # if debug: print("sorted search of\n", b, "within\n", a[i], ":\n", out[i])
    #print("index positions:", out, out.shape)
    # shouldn't the output have the shape of b?
  #print(">> index positions:\n", out, out.shape)
  return(out)
# -----------------------------------------------------------------------------
#@profile
def in1d_sorted(A,B): 
  """
  This function computes intersection of 2 arrays more quickly than intersect1d
    > ex: possible observations = intersection of observable & survey days
    
    Gets the index of each B if they were inserted in A, in order.
      > idx = np.searchsorted(B, A)

    Then makes the index of the last one zero?
      > idx[idx==len(B)] = 0

    Returns: 
      A value where A == B for each B index

        > A[B[idx] == A]
  """
  idx = np.searchsorted(B, A)
  idx[idx==len(B)] = 0
  return A[B[idx] == A]

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
  """

    USAGE:

      Replace warnings.showwarning:
        > warnings.showwarning = warn_with_traceback
      
      Also needed so repeated warnings not ignored: 
        > warnings.simplefilter("always")

    ----
    INFO:

      Source - https://stackoverflow.com/a/22376126
      Posted by mgab, modified by community. See post 'Timeline' for change history
      Retrieved 2026-03-24, License - CC BY-SA 4.0

  """

  log = file if hasattr(file,'write') else sys.stderr
  traceback.print_stack(file=log)
  log.write(warnings.formatwarning(message, category, filename, lineno, line))

def mk_per(start, end, con):

  nestPeriod = np.stack((start, end)) # +> create array of tuples
  # NOTE need the double parentheses so it knows output is tuples
  nestPeriod = np.transpose(nestPeriod) # +> an array of start,end pairs 
  return(nestPeriod)

def stormGen(frq, dur, config, rng, stormDat, stFromFile=True, db=1):
  """
    generate a list of days where storms happened.

    the probabilities and week start dates used are read from csv outside the 
    function to streamline it.

    for rng.choice: a=array of values to choose from, p=associated probabilities
    ----
    RETURNS:
      a numpy array of values
  """
  stormProb,weekStart = stormDat
  if config.debugNests>=4: dfPrint(np.array([stormProb]), names=list(weekStart))
  if stFromFile:
    out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    if db>=3: print(f">> stormGen: {weekStart=} {stormProb=}") 
    rand =  rng.choice(7, size=len(out))
    out = out + rand
    if db>=4: print(f"\t\t\tadd a random number to week start: {rand=} ; {out=}")
  else:
    out = rng.choice(40, size=frq,replace=False)
    if db>=1: print("|>SMALL: choosing storm days from np.arange(40)", end=" ")
  dr = np.arange(0, dur, 1)
  stormDays = [out + x for x in dr] # add sequential storm days when dur>1
  stormDays = np.array(stormDays).flatten()
  # splits      = np.where(np.diff(stormDays)!=1)[0] +1 # print(f"{splits=}")
  # storms      = np.split(stormDays, splits)
  # print(f"\t\t{storms=}") # arrPrint(stormDays)
  # NOTE: should i move part of the storm creation out of mk_survey_days to here?
  return(stormDays)

def svy_position(initiation, nestEnd, surveyDays, cn):
  """
    Finds index in surveyDays of iniatiation and end dates for each nest
    ----
    RETURNS
      tuple of (init date pos, end date pos)

  """
  position = np.searchsorted(surveyDays, initiation) 
  # if cn.debugObs>=4:
  #   print("\t\t>> initiation dates:")
  #   arrPrint(initiation)
  #   print("\t\t>>>> position of initiation date in survey day list:") 
  #   arrPrint( position)
  #   print("\t\t>> end dates:")
  #   arrPrint(nestEnd)
  position2 = np.searchsorted(surveyDays, nestEnd)
  surveyDays = dict(zip(np.arange(len(surveyDays)), surveyDays))
  # if cn.debugObs>=4:
  #   print("\t\t>>>> position of end date in survey day list:", position2, len(position2)) 
  #   print("\t\t>> survey days with index number:", surveyDays)
  
  return((position, position2)) # +> return a tuple


# -----------------------------------------------------------------------------

def uniquify(path):
  """
    from https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    
    Adds a number to the end of duplicate filenames
  """
  filename, extension = os.path.splitext(path)
  counter = 1
  # print("filename, extension:", filename, extension)

  while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1

  return path

# -----------------------------------------------------------------------------

def mk_param_list_list(parL: Dict[str, list],pStatic,stInd=0,fdir:str="", suf="", listRet=False, debug=False) -> list:
# def mk_param_list_list(parL: Dict[str, list],stInd=0,fdir:str="", suf="", listRet=False, debug=False) -> list:
  """
    Take the dictionary of lists of param values, then unpack the lists to a 
    list of lists. Then feed this list of lists to itertools.product using *.
    
      Also, write entire set of param lists to csv if **fdir** is specified.
      Can add a suffix to filename using **suf**
    -----
    ARGS:
      stInd = which param set to start at
    -----
    RETURNS: 
      **if !listRet:**
        list of dicts (1 per param combo), with keys!
      **if listRet:**
        the same but as a list of lists
    -----
    NOTES
      - product() takes any number of iterables as input;
      - input in the original is a bunch of lists;
      - output in the original is a list of tuples

  """
  # TODO: could add ** to surround for docstrings?
  # if debug:
    # print(f"\t\t>=> using the {parL} params lists",end=" ")
  # listVal = [parL[key] for key in parL]
  # print(f"{type(listVal)=}")
  if debug: print(f"\tbefore: {pStatic=}")
  pStatic = {k: v for k, v in pStatic.items() if k not in parL}
  if debug: print(f"\tafter: {pStatic=}")
  par_merge = {**parL, **pStatic}
  listVal = [par_merge[key] for key in par_merge]
  # pL = list(itertools.product(*listVal))
  pL = list(itertools.product(*listVal))
  if debug: print(f"{pL=}")
  if fdir:
    plfile = os.path.join(fdir, f"param-lists_{suf}.csv")
    if debug: print(f"\n\t\t|> param list file: {plfile}")
    with open(plfile, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(pL)
  
  if listRet:
    if debug: print('returning list of lists (not list of dicts)')
    return(pL)
  else:
    # +> make this list of lists into a list of dicts with the original keys:
    # paramsList = [dict(zip(parList.keys(), p_List[x])) for x in range(len(p_List))]
    # paramsList = [dict(zip(parL.keys(),pL[x])) for x in range(stInd,len(pL))]
    paramsList = [dict(zip(par_merge.keys(),pL[x])) for x in range(stInd,len(pL))]
    if debug:
      # print(f"\t\t{type(paramsList)=} \t\t{paramsList=}")
      print(f"\n\t\tParam list values ({type(paramsList)=}):")
      # print("\t\t\t","\t\t".join(parL.keys()))
      # print("\t\t","\t\t\t".join(str(val) for val in parL.values()))
      widths = [max(len(str(k)),len(str(v))) for k,v in par_merge.items()]
      print("\t\t"," ".join(f"{str(k):<{w}}" for k,w in zip(par_merge.keys(),widths)))
      print("\t\t"," ".join(f"{str(v):<{w}}" for v,w in zip(par_merge.values(),widths)))
      # widths = [max(len(str(k)),len(str(v))) for k,v in parL.items()]
      # print("\t\t"," ".join(f"{str(k):<{w}}" for k,w in zip(parL.keys(),widths)))
      # print("\t\t"," ".join(f"{str(v):<{w}}" for v,w in zip(parL.values(),widths)))
      # print("\t\t"," ".join(str(val) for val in parL.values()))
    return(paramsList)

def mk_param_list(par, debug=False):
  """
    INPUT: a dict containing all params for this set
    RETURN: a Params instance
  """
  # if debug: print(f"\t{type(par)=} ; {type(pStatic)=}")
  # if debug: print(f"\t{par=} ; {pStatic=}")
  # time.sleep(2)
  # try:
  
  # pStatic = {k: v for k, v in pStatic.items() if k not in par}
  # if debug: print(f"\tafter: {pStatic=}")
  # par_merge = {**par, **pStatic}
  if debug:

    # print("\t".join(par_merge.keys()))
    # print("\t".join(str(val) for val in par_merge.values()))
    print("\t".join(par.keys()))
    print("\t".join(str(val) for val in par.values()))
  # except TypeError as error:
    
  # return Params(**par_merge)
  return Params(**par)

# def mk_outdir(nowstr, con,seed:str="", suf="", unique=False, debug=False):
#   """
#     Create a directory w/ a unique name using datetime.today() & uniquify().
#
#     ----
#     OPTIONS:
#       unique - should the directory name be made unique using uniquify?
#
#     ----
#     RETURNS: 
#       the directory name
#
#     ----
#     NOTES:
#       > pass same nowstr to this function and mk_fnames so everything matches
#   """
#
#   # TODO: decide whether I want to include seed in dir name, or just filenames
#   # like_f_dir = con.likeDir
#   # like_f_dir = "/home/wodehouse/Projects/sim_model/out/default"
#   like_f_dir = "/home/wodehouse/Dropbox/Models/ch2_analysis/py_out"
#   # like_f_dir = "/home/wodehouse/Projects/sim_model/output"
#   if not seed: seed=con.rngSeed
#   seedStr = f"_{seed}"
#   if unique:
#     # now  = datetime.today().strftime('%m%d%Y_%H%M%S')
#     fdir   = Path(uniquify(Path.home()/ like_f_dir / (nowstr + suf)))
#     # ndir   = Path(uniquify(Path.home()/ like_f_dir / ('nests_' +nowstr + suf)))
#   else:
#     # now = datetime.today().strftime("%Y%m%d")
#     # fname  = f"ml_val_{now}.csv"
#     fdir = Path(Path.home() / like_f_dir / nowstr / suf) # need the parens or get an error about concatenating string and Path?
#     # ndir   = Path(Path.home()/ like_f_dir / ('nests_' +nowstr + suf))
#   os.makedirs(fdir, exist_ok=True)
#   if debug: print("\t\t>> save directory name:", fdir)
#   # print("\t>> nest directory name:", ndir)
#   # return((fdir,ndir))
#   return(fdir)

# def mk_fnames(suf:str, f_dir, unique=True):
# def mk_fnames(suf:str, con=config, unique=True):
# def mk_fnames(fdir, nowstr, suf:str, con=config):


