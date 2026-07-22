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


# -----------------------------------------------------------------------------

def init_from_csv(file, debug=False):
    # file="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"):
    # file=storm_init):
  """
  import initiation probabilities by week (based on real nest data).

  returns dict with key=weekStart, value=initProb
  """
  init= np.genfromtxt(
      fname=file,
      dtype=float,
      delimiter=",",
      skip_header=1,
      usecols=2
      )
  # the initprob decimals in the csv don't sum to 1 anymore
  initProb = init / np.sum(init) # make them into probabilities again
  init_weeks = np.arange(14,29,1)
  weekStart = (init_weeks * 7) - 90 # why minus 90?
  weekStart = weekStart.astype(int)
  # return(dict(zip(weekStart, initProb)))
  ret = dict(zip(weekStart, initProb))
  # if debug: print("\t>=> week: init probability = ",round(ret,3))
  # if debug: print("\t>=> loading init dates; <week start date>: <init probability> ")
  if debug: print(f"\t\t>=> loading init dates \t\t[{sum(initProb)=}] : ")
  if debug: dfPrint(np.array([list(ret.values())]), names=list(ret.keys()))
  # if debug: dfPrint(pd.DataFrame(ret,index=['i',]))
  # if debug: dfPrint(pd.DataFrame.from_dict(ret,orient="index").T, names=list(ret))
  # if debug: print("\t\t\t\t",{str(key): str(round(value,3)) for key,value in ret.items()}, end=" ")
  # if debug: print(sum(initProb))
  # if debug: print("\t", {str(key): str(round(value,3)) for key,value in ret.items()})
  # if debug: arrPrint("   ".join(map(str,ret.keys())))
  # if debug: arrPrint(" ".join(map(str,ret.values())))
  # if debug: pprint.pprint({str(key): str(round(value,3)) for key,value in ret.items()}, indent=4, width=90)
  # if debug: print({as.int(key): as.float(round(value,3)) for key,value in ret.items()})
  return(ret)
  # return(initProb)
# -----------------------------------------------------------------------------
def sprob_from_csv(file, debug=True):
    # file="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"):
  """
    import storm probabilities by week (based on real storm data).

    returns dict with key=weekStart, value=stormProb


  """

  stormProb = np.genfromtxt(
      #fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
      fname=file,
      dtype=float,
      delimiter=",",
      skip_header=1,
      usecols=3 # 4th column 
      )
  storm_weeks2 = np.arange(14,29,1)
  weekStart = (storm_weeks2 * 7) - 90 # why minus 90?
  weekStart = weekStart.astype(int)
  ret = dict(zip(weekStart, stormProb))
  # df = pd.DataFrame()
  # if debug: print("\t>=> week start date: storm probability =\n",round(ret,3))
  # if debug: print("\t\t>=> loading storm prob; <week start date>: <storm probability> ")
  if debug: print(f"\n\t\t>=> loading storm prob; \t\t[{sum(stormProb)=}] ")
  # if debug: dfPrint(np.array([stormProb]).T, names=weekStart)
  if debug: dfPrint(np.array([list(ret.values())]), names=list(ret.keys()))
  # if debug: print({key: round(value,3) for key,value in ret.items()})
  # retstr = map(str, ret)
  # if debug: print({key: round(value,3) for key,value in retstr.items()}) ##doesn't work
  # if debug: print(map(str,{key: round(value,3) for key,value in ret.items()}))##doesn't work
  # if debug: print("\t\t\t\t",{str(key): str(round(value,3)) for key,value in ret.items()}, end=" ")
  # if debug: print(sum(stormProb))
  # if debug: pprint.pprint({str(key): str(round(value,3)) for key,value in ret.items()}, indent=4, width=90)
  return(ret)

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
def mk_param_list_list(parL: Dict[str, list],stInd=0,fdir:str="", suf="", listRet=False, debug=False) -> list:
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
  listVal = [parL[key] for key in parL]
  pL = list(itertools.product(*listVal))
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
    paramsList = [dict(zip(parL.keys(),pL[x])) for x in range(stInd,len(pL))]
    if debug:
      # print(f"\t\t{type(paramsList)=} \t\t{paramsList=}")
      print(f"\n\t\tParam list values ({type(paramsList)=}):")
      # print("\t\t\t","\t\t".join(parL.keys()))
      # print("\t\t","\t\t\t".join(str(val) for val in parL.values()))
      widths = [max(len(str(k)),len(str(v))) for k,v in parL.items()]
      print("\t\t"," ".join(f"{str(k):<{w}}" for k,w in zip(parL.keys(),widths)))
      print("\t\t"," ".join(f"{str(v):<{w}}" for v,w in zip(parL.values(),widths)))
      # print("\t\t"," ".join(str(val) for val in parL.values()))
    return(paramsList)

def mk_param_list(par, pStatic, debug=False):
  """
  """
  if debug: print(f"\t{type(par)=} ; {type(pStatic)=}")
  if debug: print(f"\t{par=} ; {pStatic=}")
  # time.sleep(2)
  # try:
  
  pStatic = {k: v for k, v in pStatic.items() if k not in par}
  if debug: print(f"\tafter: {pStatic=}")
  par_merge = {**par, **pStatic}
  if debug:

    print("\t".join(par_merge.keys()))
    print("\t".join(str(val) for val in par_merge.values()))
  # except TypeError as error:
    
  return Params(**par_merge)

def calc_nests(nestData1, par,rng,survey, obsCol,repID, parID, config, db=0):
  # rng = np.random.default_rng(seed=config.rngSeed) print("calling calc_nests")
  obsCol = int(obsCol)
  # print(f"{obsCol=}")
  surveyInt = survey[1]
  # print(f"{surveyInt=}")
  # print(f"{nestData1=}")
  longest_int = max(surveyInt)
  flooded  = sum(nestData1[:,3]==2)
  hatched  = sum(nestData1[:,3]==0)
  # discover = nestData1[:,8]>0 ## where num obs > 0
  # print(f"{nestData1[:,obsCol]=}")
  discover = nestData1[:,obsCol]>0 ## where num obs > 0
  # if db>=5: print(f"\t\t\t{discover=}")
  # discover = nestData1[:,10]>0 ## where num obs > 0
  nestData = nestData1[(discover),:] # +> remove undiscovered nests
  # if db>=5: print(f"\t\t\t{nestData[:,2]=}")
  # if db>=5: print(f"\t\t\t{nestData[:,4]=}")
  flood_dsc  = sum(nestData[:,3]==2)
  hatch_dsc  = sum(nestData[:,3]==0)
  # if db>=3: print("\t\tcalc_nests: using column 8 to determine discovered/not")
  # if db>=3: print(
  #     f"\t\tcalc_nests: using column {obsCol} to determine discovered/not")
  # if db>=5: print(f"\t\t>>calc_nests: discovered: {len(nestData)=}")
  # if db>=3: print(f"{(nestData[:,5]==nestData[:,6])=}")
  # short    = (nestData[:,4]==nestData[:,5]) ## where i==j
  short    = (nestData[:,2]<nestData[:,4]) ## where end<i
  # short    = np.zeros(len(nestData))
  # if db>=5: print(f"\t\t\t{short.astype(int)=}")
  # exclude  = ((nestData[:,7] == 7) or (nestData[:,5]==nestData[:,6]))
  unknown  = (nestData[:,7]==7)
  # if db>=5: print(f"\t\t\t{unknown.astype(int)=}")

  # print(f"{(unknown.astype(int) + short.astype(int))=}")
  # both = (unknown.astype(int) + short.astype(int))
  # exclude = unknown or short
  # can also use bitwise or (|) or np.logical_or():
  exclude = (unknown.astype(int) + short.astype(int)) > 0 # at least one is true
  # if db>=5: print(f"\t\t\t{exclude=}")
  misclass = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests
  nestData = nestData[~exclude,:] # +> remove undiscovered nests
  flood_an  = sum(nestData[:,3]==2)
  hatch_an  = sum(nestData[:,3]==0)
  misclass2 = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests
  # if db>=5: print(f"\t\t>>calc_nests: analyzed:{len(nestData)=}")
  # if db>=2: print(f"{nestData[:,11]=}")
  # misclass = misclass - unknown
  avgFInt  = (nestData[:,9].sum()/len(discover))
  sNest    = nestData1[:,11].sum()
  avgK     = nestData[:,6].sum()/len(discover)
  maxI     = np.max(nestData[:,4])
  srand = rng.uniform(0.00, 10.00) # +> random init val for MARK
  # mark_s = run_optim(minimizer="norm",
  #                    fun=mark_wrapper,
  #                    z=srand,
  #                    arg=(nestData, par.brDays, config),
  #                    met=config.optimizer
  #                    )
  appDSR  = calc_dsr(nData=nestData1,
                      nestType="all",
                      calcType="apparent",
                      conf=config,
                      incTime=par.hatchTime,
                      psurv=par.probSurv,
                      debug=config.debugDSR)
  # markPSR = mark_s ** par.hatchTime
  appPSR = appDSR ** par.hatchTime
  # lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
  #                survey=survey,config=config)
  # # llDSR = lVal[0]
  # llDSR,llPSR,llDFR = lVal


  mayfDSR_an   =  calc_dsr(nData=nestData,
                           nestType="analysis",
                           calcType="mayfield",
                           conf=config,
                           incTime=par.hatchTime,
                           psurv=par.probSurv,
                           debug=config.debugDSR) 
  appDSR_an   = calc_dsr(nData=nestData,
                          nestType="analysis",
                          calcType="apparent",
                          conf=config,
                          incTime=par.hatchTime,
                          psurv=par.probSurv,
                          debug=config.debugDSR) 
  nestVals = np.array([
    # flooded,hatched,discover.sum(),exclude.sum(),unknown.sum(),
    # misclass.sum(), avgFInt, avgK, appDSR, mark_s, repID, parID])
    # parID,repID,flooded,hatched,sNest,discover.sum(),exclude.sum(),unknown.sum(),
    parID,repID,flooded,hatched,flood_dsc,hatch_dsc,flood_an,hatch_an,
    sNest,discover.sum(),exclude.sum(),unknown.sum(),
    # misclass.sum()-unknown.sum(),avgFInt,avgK,appDSR,appPSR,mayfDSR_an,appDSR_an])
    misclass.sum()-unknown.sum(),misclass2.sum(),avgFInt,avgK,maxI,longest_int,appDSR,appPSR,mayfDSR_an,appDSR_an])
    # misclass.sum(), avgFInt, avgK, appDSR,appPSR, mark_s,markPSR])
  if db>=5: print(f"\t\t{nestVals=}")
  return nestVals
# def mk_outdir(nowstr, seed:str="", suf="",con=config, unique=False):

def mk_outdir(nowstr, con,seed:str="", suf="", unique=False, debug=False):
  """
    Create a directory w/ a unique name using datetime.today() & uniquify().

    ----
    OPTIONS:
      unique - should the directory name be made unique using uniquify?

    ----
    RETURNS: 
      the directory name

    ----
    NOTES:
      > pass same nowstr to this function and mk_fnames so everything matches
  """
  # TODO: decide whether I want to include seed in dir name, or just filenames
  # like_f_dir = con.likeDir
  
  like_f_dir = "/home/wodehouse/Projects/sim_model/out/default"
  # like_f_dir = "/home/wodehouse/Projects/sim_model/output"
  if not seed: seed=con.rngSeed
  seedStr = f"_{seed}"
  if unique:
    # now  = datetime.today().strftime('%m%d%Y_%H%M%S')
    fdir   = Path(uniquify(Path.home()/ like_f_dir / (nowstr + suf)))
    # ndir   = Path(uniquify(Path.home()/ like_f_dir / ('nests_' +nowstr + suf)))
  else:
    # now = datetime.today().strftime("%Y%m%d")
    # fname  = f"ml_val_{now}.csv"
    fdir = Path(Path.home() / like_f_dir / nowstr / suf) # need the parens or get an error about concatenating string and Path?
    # ndir   = Path(Path.home()/ like_f_dir / ('nests_' +nowstr + suf))
  os.makedirs(fdir, exist_ok=True)
  if debug: print("\t\t>> save directory name:", fdir)
  # print("\t>> nest directory name:", ndir)
  # return((fdir,ndir))
  return(fdir)
# def mk_fnames(suf:str, f_dir, unique=True):
# def mk_fnames(suf:str, con=config, unique=True):
# def mk_fnames(fdir, nowstr, suf:str, con=config):
def mk_fnames(nowstr,con,test=False,seed:str="",suf:str="",fdir=None,uniq=False):
  """
    1. Create likelihood filepath (& parent dir, if necessary)
      --> ERROR if filepath exists

    2. Make a string out of the column names that can be used w/ np.savetxt()
    
    ------
    RETURNS:
      tuple of likelihood filepath & colnames string
  """
  if fdir is None:
    print("\t\tno fdir provided; using default")
    # fdir   = mk_outdir(nowstr, unique=uniq)
    fdir   = mk_outdir(nowstr,con=con)
  # print(f"{fdir=}")
  if not seed: seed=con.rngSeed
  seedStr = f"{seed}"
  if uniq:
    suf+=now

  if test:
    # lfname  = "ml_val_" + nowstr + suf + ".csv"
    full_suf =   suf + ".csv"
  else:
    # lfname  = "ml_val_" + nowstr + seedStr + suf + ".csv"
    full_suf =   seedStr + suf + ".csv"
  lfname = f"ml_val_{full_suf}"
  # nfname = f"nests_{full_suf}"
  likeF  = Path(fdir / lfname)
  # if os.path.exists(likeF):

      
  # likeF = Path(uniquify(Path.home() / like_f_dir / now + suf / fname ))
                #  'C://Users/Sarah/Dropbox/Models/sim_model/py_output' / 
                 # fname))
  likeF.parent.mkdir(parents=True, exist_ok=True)
  # print(likeF)
  # likeF = Path(Path.home() / like_f_dir / now + suf / fname )
  likeF.parent.mkdir(parents=True, exist_ok=True)
    
    # likeF = Path(Path.home()/'Dropbox/Models/sim_model/py_output/'/fname)
  # f_dir = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output/"
  # fpath = Path(f_dir/ fname)
  # f_dir = con.likeDir
  # like_f_dir = con.likeDir
  # like_f_dir = "/home/wodehouse/Projects/sim_model/out/default"
  # fpath = like_f_dir + "/" + nowstr + "/" + fname
  fpath = str(likeF)
  # print("\t\t\t> fpath (written to txt file):",fpath)
  # with open('likeFile-name.txt', 'w' ) as f:
  lfname = Path(fdir / 'likeFile-name.txt')
  with open(lfname, 'w' ) as f:
    # f.write(str(likeF))
    f.write(str(fpath))
  column_names = np.array([
    # 'mark_s', 'psurv_est', 'ppred_est', 'pfl_est', 'ss_est', 'mps_est', 'mfs_est',
    #    0          1          2           
     'dsr_est','psr_est' 'ppred_est',
    # 'mark_s', 'psurv_est', 'ppred_est',
    # 'ps_given', 'dur', 'freq', 'n_nest', 'h_time', 'obs_fr',
    #   3           4       5               6           7
    'appDSR','appDSR_an','appDSRdisc','mayfDSRdisc','mayfDSR_an',# ''
    #       8           9       10          11          12          13
    'discovered', 'excluded', 'unknown', 'misclass','flooded','hatched',
    # 'nExc', 'repID', 'parID','psurv_est2', 'ppred_est2'
    # #   14      15      16
    # 'nExc', 'repID', 'parID'
    #14      15 
    'repID', 'parID'
    # 'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 
    # 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 
    # 'storm_freq', 'psurv_real', 'psurv_found', 'psurv_given',
    # 'stormsurv_given','pflood_given', 'hatch_time','num_nests',
    # # 'obs_int', 'num_discovered','num_excluded', 'exception'
    # 'obs_int', 'num_discovered','num_excluded'
    # 'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 
    # 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 
    # 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 
    # 'obs_int', 'num_discovered','num_excluded', 'exception'
    ])
  # colnames = ', '.join([str(x) for x in column_names]) # needs to be string
  colnames = ','.join([str(x) for x in column_names]) # needs to be string

  # saveNames = dict(
  #   likeFile   = likeF,
  #   dirName  = datetime.today().strftime('%m%d%Y_%H%M%S'),
  #   todaysDate = datetime.today().strftime("%Y%m%d"),
  #   colnames = ', '.join([str(x) for x in column_names]) # needs to be string
  # )
  print("\t\t\t>=> likelihood file path:", likeF)
  # return(saveNames)
  return(likeF, colnames)
  # return(fdir, likeF, colnames)
# -----------------------------------------------------------------------------

