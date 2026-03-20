from datetime import datetime
import csv
import itertools
import numpy as np
import os
from pathlib import Path
# import matplotlib.pyplot as plt
import pprint
import sys
from typing import Dict, Generator
import yaml
# from datsim import config
from getClass import Config
from settings import config, rng
debug = config.debug
# NOTE: maybe make an indent print function for strings? instead of typing \t all the time

# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------

def init_from_csv(file):
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
  if debug: print("\t\t>=> loading init dates; <week start date>: <init probability> ")
  if debug: print("\t\t\t\t",{str(key): str(round(value,3)) for key,value in ret.items()}, end=" ")
  if debug: print(sum(initProb))
  # if debug: print("\t", {str(key): str(round(value,3)) for key,value in ret.items()})
  # if debug: arrPrint("   ".join(map(str,ret.keys())))
  # if debug: arrPrint(" ".join(map(str,ret.values())))
  # if debug: pprint.pprint({str(key): str(round(value,3)) for key,value in ret.items()}, indent=4, width=90)
  # if debug: print({as.int(key): as.float(round(value,3)) for key,value in ret.items()})
  return(ret)
  # return(initProb)
# -----------------------------------------------------------------------------
def sprob_from_csv(file):
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
  # if debug: print("\t>=> week start date: storm probability =\n",round(ret,3))
  if debug: print("\t\t>=> loading storm prob; <week start date>: <storm probability> ")
  # if debug: print({key: round(value,3) for key,value in ret.items()})
  # retstr = map(str, ret)
  # if debug: print({key: round(value,3) for key,value in retstr.items()}) ##doesn't work
  # if debug: print(map(str,{key: round(value,3) for key,value in ret.items()}))##doesn't work
  if debug: print("\t\t\t\t",{str(key): str(round(value,3)) for key,value in ret.items()}, end=" ")
  if debug: print(sum(stormProb))
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
def mk_param_list(parList: Dict[str, list], fdir: str, suf="") -> list:
  """
    Take the dictionary of lists of param values, then unpack the lists to a 
    list of lists. Then feed this list of lists to itertools.product using *.
    
    Also, write entire set of param lists to csv.
    
    Returns
    -----
    a list of dicts representing all possible param combos, with keys!
    
    Notes
    -----
    product takes any number of iterables as input;
    input in the original is a bunch of lists;
    output in the original is a list of tuples

  """
  print(f"\t\t>=> using the {parList} params lists")
  listVal = [parList[key] for key in parList]
  p_List = list(itertools.product(*listVal))
  plfile = os.path.join(fdir, f"param-lists_{suf}.csv")
  print(f"\t|> param list file: {plfile}")
  with open(plfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(p_List)
  # +> make this list of lists into a list of dicts with the original keys:
  paramsList = [dict(zip(parList.keys(), p_List[x])) for x in range(len(p_List))]
  
  return(paramsList)

def mk_outdir(nowstr, seed:str="", suf="",con=config, unique=False):
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
  like_f_dir = con.likeDir
  if not seed: seed=con.rngSeed
  seedStr = f"_{seed}"
  if unique:
    # now  = datetime.today().strftime('%m%d%Y_%H%M%S')
    fdir   = Path(uniquify(Path.home()/ like_f_dir / (nowstr + suf)))
    # ndir   = Path(uniquify(Path.home()/ like_f_dir / ('nests_' +nowstr + suf)))
  else:
    # now = datetime.today().strftime("%Y%m%d")
    # fname  = f"ml_val_{now}.csv"
    fdir = Path(Path.home() / like_f_dir / (nowstr + suf)) # need the parens or get an error about concatenating string and Path?
    # ndir   = Path(Path.home()/ like_f_dir / ('nests_' +nowstr + suf))
  os.makedirs(fdir, exist_ok=True)
  print("\t>> save directory name:", fdir)
  # print("\t>> nest directory name:", ndir)
  # return((fdir,ndir))
  return(fdir)
# def mk_fnames(suf:str, f_dir, unique=True):
# def mk_fnames(suf:str, con=config, unique=True):
# def mk_fnames(fdir, nowstr, suf:str, con=config):
def mk_fnames(nowstr,test=False,seed:str="",suf:str="",fdir=None,con=config,uniq=False):
  """
    1. Create likelihood filepath (& parent dir, if necessary)
      --> ERROR if filepath exists

    2. Make a string out of the column names that can be used w/ np.savetxt()
    
    ------
    RETURNS:
      tuple of likelihood filepath & colnames string
  """
  if fdir is None:
    print("no fdir provided; using default")
    fdir   = mk_outdir(nowstr, unique=uniq)
  # print(f"{fdir=}")
  if not seed: seed=con.rngSeed
  seedStr = f"{seed}"
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
  like_f_dir = con.likeDir
  # fpath = like_f_dir + "/" + nowstr + "/" + fname
  fpath = str(likeF)
  print("\t\t> fpath (written to txt file):",fpath)
  # with open('likeFile-name.txt', 'w' ) as f:
  lfname = Path(fdir / 'likeFile-name.txt')
  with open(lfname, 'w' ) as f:
    # f.write(str(likeF))
    f.write(str(fpath))
  column_names = np.array([
    # 'mark_s', 'psurv_est', 'ppred_est', 'pfl_est', 'ss_est', 'mps_est', 'mfs_est',
    'mark_s', 'psurv_est', 'ppred_est',
    # 'ps_given', 'dur', 'freq', 'n_nest', 'h_time', 'obs_fr',
    'trueDSR', 'trueDSR_analysis', 'discovered', 'excluded', 'unknown', 'misclass','flooded','hatched',
    # 'nExc', 'repID', 'parID','psurv_est2', 'ppred_est2'
    'nExc', 'repID', 'parID'
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
  colnames = ', '.join([str(x) for x in column_names]) # needs to be string

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

