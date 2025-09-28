#!/usr/local/bin/python


# sudo vim -o file1 file2 [open 2 files] 
# BLAH
# :/^[^#]/ search for uncommented lines
# /^[^#]*\s*print  or /^\s*print 
# > kernprof -l simdata_vect.py > 20sepprofile.out
# > python -m line_profiler .\simdata_vect.py.lprof
# NOTE 5/16/25 - The percent bias responds more like I would expect when I use
#                the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
#                BUT I still don't know why the calculated DSR is consistently low.

# import argparse
from dataclasses import dataclass
from datetime import datetime
import decimal
from decimal import Decimal
# from itertools import product
import getopt
import itertools
# import line_profiler
# import numexpr as ne
import numpy as np 
# from os.path import exists
import os
from pathlib import Path
from scipy import optimize
import scipy.stats as stats
import sys
from typing import Dict, Generator
import csv

# NOTE: 
# 1. turned off useSM (params)
# 2. output is 9 columns for now
# -----------------------------------------------------------------------------
#  SETTINGS 
# -----------------------------------------------------------------------------
@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Params: # most importantly, Pylance recognizes the attributes, unlike 
              # dict keys
    numNests: int
    stormDur: int
    stormFrq: int
    obsFreq:  int
    hatchTime:int
    brDays:   int
    whichLike:int
    probSurv: np.float32
    SprobSurv:np.float32
    pMortFl  :np.float32
    discProb: np.float32
    # fateCues: np.float32
    stormFate:bool
    useSMat:  bool

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Config: 
    """
    use different debug var bc these will print for every time optimizer runs
    """
    # rng:         Generator
    # args:        list[str]
    nreps:       int
    debug:       bool
    debugLL:     bool
    debugNests:  bool
    debugFlood:  bool
    debugObs:    bool
    debugM:      bool
    useWSL:      bool
    # testing:     bool
    testing:     str
    fnUnique:    bool
    likeFile:    str
    # likeDir:     str
    # stormInit:   str
    colNames:    str
    numOut:      int

# for i, arg in enumerate(sys.argv):
    # print(f"Argument {i}: {arg}")
 # use different debug var bc these will print for every time optimizer runs
# debugList = dict( debug_nest = False,
                #   debug_obs = False, 
                #   debugLL = False, 
                #   debug = False,
                #   debugM = False,
                #   debugL = False )
# fname=mk_fnames() # since function uses datetime, make sure to only call once
rng = np.random.default_rng(seed=102891)
# rng = np.random.default_rng(seed=82985)
# config = Config(rng         = np.random.default_rng(seed=102891), 
# config = Config(args        = sys.argv, 
config = Config(nreps       = 500, 
# config = Config(nreps       = 1, 
                debug       = False, 
                debugLL     = False, 
                debugNests  = False, 
                debugM      = False,
                debugObs    = False,
                debugFlood  = False,
                useWSL      = False,
                # testing     = False,
                testing     = "no",
                fnUnique    = False,
                # likeFile    = fname[0], 
                likeFile    = " ",
                colNames    = " ",
                # colNames    = fname[1],
                # numOut      = 21)
                # numOut      = 18)
                numOut      = 11)
# if config.useWSL:
#     storm_init = "/mnt/c/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"
#     like_f_dir  =  "/mnt/c/Users/Sarah/Dropbox/Models/sim_model/py_output"
# else:
#     storm_init = "C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv" 
#     like_f_dir  = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output"
# import getopt, sys
try:
    # arguments, values = getopt.getopt(args, options, optVal)
    # 'o' takes an argument, so has ':'
    # opts,args = getopt.getopt(sys.argv[1:],"htdwo:v",["Help", "Test", "Debug", "WSL-true","Out-file="])
    opts,args = getopt.gnu_getopt(sys.argv[1:],"ht:do:w",["Help", "Test", "Debug", "WSL-true"])
    # for n in range(len(opts)): print(f"{opts[n]}: {args[n]}")
    # opts,args = getopt.gnu_getopt(sys.argv[1:],"htdw",["Help", "Test", "Debug", "WSL-true"])
except getopt.error as err:
    print(str(err))
    # usage()
    sys.exit(2)
debugTypes = None
# output = None
# verbose = False
for arg, val in opts:
    if arg in ("-h", "--Help"):
        print("\n-------------------------------------------------------------------------------------------------------",
               "\nsimdata_vect.py usage:\n\n",
            #   "[-t --Test] Run in testing mode w/ smaller # of nests and reduced # of params? (Default:False)\n\n",
              "[-t --Test] Choose testing mode w/ smaller # of nests and reduced # of params:\n",
              "\t\t1. 'norm' - moderate values; 2. 'storm' - extremes of storm values; 3. 'no' (default)\n\n"
              "[-d --Debug-general] Turn on simple/broad debugging statements? (Default:False)\n\n",
              "[-o --Options-debug] More specific print statements.\n",
                "\t\t\tOptions: 'like','nest','mark','flood','obs'.\n",
                "\t\t\tplace in single string with comma delim \n\n",
              "[-w --WSL-true] Use WSL? filenames will be changed to match. (Default:False)\n",
              "\n-------------------------------------------------------------------------------------------------------"
                )
        sys.exit()
    # if arg == "-v": verbose = True
    elif arg in ("-t", "--Test"):
        # config.testing = True
        config.testing = val
        # print("Config - test mode:", config.testing)
        print("Config - test mode:", config.testing)
    elif arg in ("-d", "--Debug-general"):
        config.debug = True
        print("Config - debug mode (general):", config.debug)
    elif arg in ("-o", "--Options-debug"):
        debugTypes=val
        print("Debug options=", val)
    elif arg in ("-w", "--WSL-true"):
        config.useWSL = True
        print("Config - using WSL?", config.useWSL)
    # elif arg in ("-o", "--Output"):
        # print("Output file location:", val)
        # config.likeFile = val
print("debug options:", debugTypes)
debug = config.debug
print(debug)
# v = debugList.values() #test
# now        = datetime.today().strftime('%m%d%Y_%H%M%S')
like_f_dir = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output"
storm_init = "C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv" 
fnUnique   = False
if config.useWSL:
    # storm_init = "/mnt/c/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"
    # like_f_dir = "/mnt/c/Users/Sarah/Dropbox/Models/sim_model/py_output"
    # storm_init = "~/projects/sim_data/storm_init3.csv"
    # like_f_dir = "~/proojects/sim_data/out"
    # tilde means nothing within a string
    #storm_init = "/home/wodehouse/projects/sim_data/storm_init3.csv"
    storm_init = "/home/wodehouse/projects/sim_model/storm_init3.csv"
    #like_f_dir = "/home/wodehouse/projects/sim_data/out"
    like_f_dir = "/home/wodehouse/projects/sim_model/out"
    fnUnique   = True
# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS
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
# def mk_dname(suf, unique=True):

# def mk_fnames(like_f_dir, unique=True ):
def mk_fnames(suf:str, unique=True ):
    """
    1. Create a directory w/ a unique name using datetime.today() & uniquify().
    2. Create likelihood filepath (& parent dir, if necessary)
    3. Make a string out of the column names that can be used w/ np.savetxt()
    
    Returns:
    tuple of likelihood filepath & colnames string
    """
    if unique:
        now    = datetime.today().strftime('%m%d%Y_%H%M%S')
        fdir   = Path(uniquify(Path.home()/ like_f_dir / (now + suf)))
        fname  = "ml_val_" + now + suf + ".csv"
        likeF  = Path(fdir / fname)
            
        # likeF = Path(uniquify(Path.home() / like_f_dir / now + suf / fname ))
                                    #    'C://Users/Sarah/Dropbox/Models/sim_model/py_output' / 
                                   # fname))
        likeF.parent.mkdir(parents=True, exist_ok=True)
    else:
        now = datetime.today().strftime("%Y%m%d")
        # fname  = f"ml_val_{now}.csv"
        fdir = Path(Path.home() / like_f_dir / (now + suf)) # need the parens or get an error about concatenating string and Path?
        fname  = "ml_val_" + now + suf +".csv"
        likeF = Path(fdir / fname )
        # print(likeF)
        # likeF = Path(Path.home() / like_f_dir / now + suf / fname )
        likeF.parent.mkdir(parents=True, exist_ok=True)
        
        # likeF = Path(Path.home()/'Dropbox/Models/sim_model/py_output/'/fname)
    with open('likeFile-name.txt', 'w' ) as f:
        f.write(str(likeF))
    column_names = np.array([
        # 'mark_s', 'psurv_est', 'ppred_est', 'pfl_est', 'ss_est', 'mps_est', 'mfs_est',
        'mark_s', 'psurv_est', 'ppred_est',
        # 'ps_given', 'dur', 'freq', 'n_nest', 'h_time', 'obs_fr',
        'trueDSR', 'trueDSR_analysis', 'discovered', 'excluded', 'unknown', 'misclass','flooded','hatched',
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
    #     likeFile   = likeF,
    #     dirName    = datetime.today().strftime('%m%d%Y_%H%M%S'),
    #     todaysDate = datetime.today().strftime("%Y%m%d"),
    #     colnames = ', '.join([str(x) for x in column_names]) # needs to be string
    # )
    # print(">> save directory name:", dirName)
    print(">> likelihood file path:", likeF)
    # return(saveNames)
    return(likeF, colnames)
    # return(fdir, likeF, colnames)
# -----------------------------------------------------------------------------
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
            #fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
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
    if debug: print(">> week: init probability = ",ret)
    return(ret)
    # return(initProb)
# -----------------------------------------------------------------------------
# def sprob_from_csv( file=storm_init):
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
    if debug: print(">> week start date: storm probability =\n",ret)
    return(ret)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   NEST MODEL PARAMETERS:
#region-----------------------------------------------------------------------
# NOTE the main problem is that my fate-masking variable (storm activity) also 
#      leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
# staticPar = {'nruns': 1,
# These are the values that are passed to the Params class
initDat=init_from_csv(storm_init) # this will evaluate after storm_init has been changed for wsl
stormDat=sprob_from_csv(storm_init) # is evaluated later, can account for wsl filenames


staticPar = {'brDays': 180,
             'SprobSurv': 0.2, # never actually used
            #  'pMortFl': 0.1,
             'discProb': 0.7,
             'whichLike': 1,
            #  'stormFate': True,
             'useSMat': False
             }

# parLists = {'numNests' : [500,1000],
# parLists = {'numNests' : [150, 300],
parLists = {'numNests' : [250, 500],
# parLists = {'numNests' : [300],
            # 'probSurv' : [0.95, 0.97],
            'probSurv' : [0.96],
            'pMortFl'  : [0.9, 0.75, 0.6], # flood/storm severity
            # 'stormDur' : [3],
            'stormDur' : [1, 2],
            # 'stormFrq' : [5],
            # 'stormFrq' : [1, 3, 5],
            'stormFrq' : [1, 2, 3],
            'obsFreq'  : [3, 5, 7],
            # 'obsFreq'  : [7],
           'stormFate': [False,True],
            # 'hatchTime': [28] }
            'hatchTime': [16, 20, 28] }

plTest  = {'numNests'  : [100],
# plTest  = {'numNests'  : [50],
           'probSurv'  : [0.96],
           'pMortFl'   : [0.75],
        #    'stormDur'  : [1],
           'stormDur'  : [2],
        #    'stormFrq'  : [2],
           'stormFrq'  : [1,2],
        #    'obsFreq'   : [3],
           'obsFreq'   : [3, 7],
           'stormFate': [False,True],
           'hatchTime' : [20, 28] }
        #    'hatchTime' : [20],
            # 'useSMat'  : [True, False]
            # }

plTestFlood  = {'numNests'  : [100],
# plTest  = {'numNests'  : [30],
               'probSurv'  : [0.96],
           'pMortFl'   : [0.9, 0.6],
        #    'stormDur'  : [1, 3],
           'stormDur'  : [2],
        #    'stormFrq'  : [1, 3],
           'stormFrq'  : [1, 4],
        #    'obsFreq'   : [3, 5],
           'obsFreq'   : [3, 7],
           'stormFate': [False,True],
        #    'hatchTime' : [20, 28] }
           'hatchTime' : [16, 28],
            # 'useSMat'  : [True, False]
            }

plDebug = {'numNests'  : [50],
# plTest  = {'numNests'  : [30],
               'probSurv'  : [0.96],
           'pMortFl'   : [0.75],
        #    'stormDur'  : [1, 3],
           'stormDur'  : [2],
        #    'stormFrq'  : [1, 3],
           'stormFrq'  : [1, 4],
        #    'obsFreq'   : [3, 5],
           'obsFreq'   : [3, 7],
           'stormFate': [False,True],
           'hatchTime' : [20] }
        #    'hatchTime' : [16, 28]}
            # 'useSMat'  : [True, False]
            

def mk_param_list(parList: Dict[str, list]) -> list:
    """
    Take the dictionary of lists of param values, then unpack the lists to a 
    list of lists. Then feed this list of lists to itertools.product using *.
    
    Can also uncomment some code to write entire set of param lists to csv.
    
    Returns: a list of dicts representing all possible param combos, with keys!
    """
    # product takes any number of iterables as input
    # input in the original is a bunch of lists
    # output in the original is a list of tuples
    # listVal = parList.values() # doesn't seem to be what i want
    # p_List = list(product(parList.values()))
    print(f"using the {parList} params lists")
    listVal = [parList[key] for key in parList]
    p_List = list(itertools.product(*listVal))
    # print(p_List)
    with open("param-lists-4.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(p_List)
    # p_List = list(product(parList)) # doesn't work either; just gets keys
    # make this list of lists into a list of dicts with the original keys 
    paramsList = [dict(zip(parList.keys(), p_List[x])) for x in range(len(p_List))]
    
    return(paramsList)
# pl = mk_param_list(parList=plTest) # test
#endregion---------------------------------------------------------------------
#   SAVE FILES 
#region-----------------------------------------------------------------------
# name for unique directory to hold all output:
#endregion--------------------------------------------------------------------
#   FUNCTIONS
# -----------------------------------------------------------------------------
# Some are very small and specific (e.g. logistic function); others are 
# quite involved.
# def stormGen(frq, dur, wStart, pStorm):
def stormGen(frq, dur):
        # file="/mnt/c/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"
        # )):
    """
    generate a list of days where storms happened.

    the probabilities and week start dates used are read from csv outside the 
    function to streamline it.

    for rng.choice: a=array of values to choose from, p=associated probabilities
    """
    # stormDat=sprob_from_csv(storm_init) # is evaluated later, can account for wsl filenames
    # out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    out = rng.choice(a=[*stormDat], size=frq, replace=False, p=list(stormDat.values()))
    dr = np.arange(0, dur, 1)
    stormDays = [out + x for x in dr] # add sequential storm days when dur>1
    stormDays = np.array(stormDays).flatten()
    print(">> storm days:", stormDays)
    return(stormDays)
# -----------------------------------------------------------------------------
def triangle(x0, y0):
    """
    This function remaps values from R^2 into the lower left triangle located 
    within the unit square.
    """
    if y0 > x0:
        ret = triangle(y0, x0)
        return ret[1], ret[0]

    r0 = np.sqrt( x0**2 + y0**2)
    m  = 1.0
    if y0 != x0:
        m = y0/x0

    theta = np.arctan(m)
    r3    = r0 * 1.0/(1.0 + m)
    x3    = r3 * np.cos(theta)
    y3    = r3 * np.sin(theta)
    return x3, y3
# -----------------------------------------------------------------------------
#def logistic(x)->np.float128:
def logistic(x)->np.longdouble:
    """This is just the logistic function"""
    # Trying out type hints (PEP 484) to keep output from overflowing
    #return 1.0/( 1.0 + math.exp(-x) )
    return 1.0/( 1.0 + np.exp(-x) )
# -----------------------------------------------------------------------------
# Generate survey days & interval between each pair of survey days
def mk_surveys(stormDays, obsFreq, breedingDays):
    """
    This function creates the list of survey days by taking a random start date 
    from the first 5 breeding days and creating a range with step size determined
    by observation frequency. Then remove storm days.
    
    surveyInts = interval between survey days. surveyInts[0]=0

    Returns:
    A tuple of surveyDays & surveyInts
    """
    # first day of each week because the initiation probability is weekly 
    # the upper value should not be == to the total number of season days 
    # because then nests end after season is over 

    start       = rng.integers(1, high=5) # random day of 1st survey from 1st 5 breeding days          
    end         = start + breedingDays
    surveyDays  = np.arange(start, end, step=obsFreq)
    stormSurvey = np.isin(surveyDays, stormDays) 
    surveyDays  = surveyDays[np.isin(surveyDays, stormDays) == False] # keep only values that aren't in storm_days 
    # survey interval for first obs is 0:
    surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
    # surveyInts  = np.append(surveyInts, )
    if debug: 
        print(">> all survey days, minus storms:\n", surveyDays, len(surveyDays)) 

    return(surveyDays, surveyInts)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ------ NEST DATA HELPER FUNCTIONS -------------------------------------------
# -----------------------------------------------------------------------------
# def mk_init(weekStart, initProb, numNests):
# @profile
def mk_init(numNests):
        # file="/mnt/c/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"
        # )): 
    # initDat=init_from_csv(storm_init) # this will evaluate after storm_init has been changed for wsl
    # initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
    initWeek = rng.choice(a=[*initDat], size=numNests, p=list(initDat.values()))  # random starting weeks; len(a) must equal len(p)
    initiation = initWeek + rng.integers(7)                    # add a random number from 1 to 6 (?) 
    # if debug: print(">> initiation week start days:\n", initWeek) 
    return(initiation)
# -----------------------------------------------------------------------------
def mk_surv(numNests, hatchTime, pSurv, con=config):
    # 4. Decide how long each nest is active
    # >> use a negative binomial distribution - distribution of number of 
    #    failures until success 
    #     >> in this case, "success" is actually the nest failing, so use 
    #        1-pSurv (the failure probability) 
    #     >> gives you number of days until the nest fails (survival)
    #     >> if survival > incubation time, then the nest hatches 
    # >> then use survival to calculate end dates for each nest 
    #    (end = initiation + survival)

    survival = np.zeros(shape=(numNests), dtype=np.int32)
    survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests) 
    survival = survival - 1 # but since the last trial is when nest fails, need to subtract 1
    # if debug: print(">> survival in days:\n", survival, len(survival)) 
    
    ## >> set values > incubation time to = incubation time (nest hatched): 
    ##      (need to because you are summing the survival time)
    #        >> once nest reaches incubation time (+/- some error) it hatches
    #           and becomes inactive

    survival[survival > hatchTime] = hatchTime # add some amt of error?
    # if con.debugNests: print(">> survival in days:\n", survival, len(survival)) 
    # hatched = survival >= hatchTime # the hatched nests survived for >= hatchTime days 
    # if con.debugNests: print("hatched (no storms):", hatched, hatched.sum())
    ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
    # if debug: print("real hatch proportion:", hatched.sum()/numNests)
    return(survival)
# -----------------------------------------------------------------------------
def mk_per(start, end, con=config):

    # nestPeriod = np.stack((nestData[:,1], (nestData[:,1]+nestData[:,2]))) # create array of tuples
    # need the double parentheses so it knows output is tuples
    nestPeriod = np.stack((start, end)) # create array of tuples
    nestPeriod = np.transpose(nestPeriod) # an array of start,end pairs 
    # if con.debugNests: print(
    #     ">> start and end of nesting period:\n", 
    #     nestPeriod, 
    #     nestPeriod.shape
    #     )
    return(nestPeriod)
# -----------------------------------------------------------------------------
# def mk_nests(par, init, weekStart, nestData): 
def mk_nests(par, nestData): 
    """
    nestData is an empty np array to be filled.
        
    Returns:
    -------
    3 columns: nest ID, initiation date, end date
    """

    # 1. Unpack necessary parameters
    # NOTE about the params at the beginning of the script:
    # some have only 1 member, but they are still treated as arrays, not scalars

    # hatchTime = int(params[5]) 
    # obsFreq   = int(params[6]) 
    # numNests  = int(params[0]) 
    # pSurv     = params[1]       # daily survival probability
    # fateCuesPresent = 0.6 if obsFreq > 5 else 0.66 if obsFreq == 5 else 0.75
    # if debug: print(
    #     ">> observation frequency:", obsFreq, 
    #     ">> prob of correct fate:", fateCuesPresent,
    #     )

    # 2. Assign values to the dataframe
    # nestData[:,0] = np.arange(1,par.numNests+1) # column 1 = nest ID numbers 
    nestData[:,0] = np.arange(par.numNests) # column 1 = nest ID numbers 
    nestData[:,1] = mk_init(par.numNests)                              # record to a column of the data array
    # nestData[:,1] = mk_init(weekStart, initProb, numNests)                              # record to a column of the data array
    #s if debug: print(">> end dates:\n", nestEnd, len(nestEnd)) 
    survival = mk_surv(par.numNests, par.hatchTime, par.probSurv)
    nestData[:,2] = nestData[:,1] +survival
    ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
    # nestData[:,3] = nestData[:,2] > hatchTime
    # nestData[:,3] = nestData[:,1] + nestData[:,2]
    # don't need to add end date to dataframe
    # NOTE Remember that int() only works for single values 
    # if debug: print(nestData[1:6,:])
    return(nestData)

# ---- FLOODING & SUCH -------------------------------------------------------
# def storm_nest(nestPeriod, surveysDays, stormDays, con=config):
def storm_nest(stormFreq, nestPeriod, stormDays, con=config):
    """
    Returns:
    -------
    a list containing numStorms & stormNestIndex
    """
    # >> stormNestIndex searches for storm days w/in active period of each nest
    #     >> returns index where storm day would be within the active interval: 
    #             0 = before init; 2 = after end; 1 = within interval
    #     >> fate cues should become harder to interpret after storms
    stormNestIndex = np.zeros((len(nestPeriod), stormFreq))
    stormNestIndex = searchSorted2(nestPeriod, stormDays)
    # stormNest = np.any(stormNestIndex == 1, axis=1) 
    # numStorms = np.sum(stormNestIndex==1, axis=1) # axis=1 means summing over rows?

    # if con.debugNests: print("where were storms in active period?", stormNestIndex)
    # if index == 1, then storm Day is within the period interval: 
    # whichStorm = np.zeros((len(nestPeriod), stormFreq))
    # whichStorm = []
    
    # if stormNestIndex.shape[1] > 1:
    #     # whichStorm = np.where(stormNestIndex == 1, )
    #     for n in range(len(nestPeriod)):
    #         # whichStorm[n] = np.zeros(stormFreq)
    #         # whichStorm[n] = np.array([i for i,val in enumerate(stormNestIndex[n]) if val==1 else -1] )
    #         if stormNest[n]:
    #             whichStorm[n] = [i for i,val in enumerate(stormNestIndex[n]) if val==1]
    #         else:
    #             whichStorm[n] = [-4]
    #     # try:
    #         # whichStorm = stormNestIndex.index(1)
    # stormNestDays = 
    # stormNestInd2 = np.zeros(len(nestPeriod))
    # for n in range(sum(stormNest)):
    #     stormNestInd2[n]  = searchSorted2(stormDays, nestPeriod[n])
    # NOTE I *think* this is actually number of storm intervals, which is what 
    # we want for the likelihood function. so that would be good...
    # return([numStorms, stormNestIndex])
    # return(numStorms)
    return(stormNestIndex)
# -----------------------------------------------------------------------------
def mk_flood( stormDays, pMortFl, stormIndex, numNests, con=config):
    """
    NEED TO KNOW WHICH STORM SO CAN CHANGE END DATE
    Decide which nests fail from flooding:
        1. Create a vector of random probabilities drawn from a uniform dist
        2. Compare the random probs to pfMort
        3. If flooded=1 and it was during a storm, then nest flooded
        
    Arguments:
        params - really only using one of these (pfMort)
        numStorms - vector telling how many storm periods intersected with 
                    active period, for each nest

    Returns: list: [0] num storms [1] which storm first flooded [2] T/F nest flooded AND during storm? 
    """
    # pfMort = params[2]       # prob of surviving (not flooding) during storm
    # print("prob of failure due to flooding:", pfMort)
    # pflood = rng.uniform(low=0, high=1, size=numNests) 
    # numStorms, stormIndex= stormOut
    # stormIndex = stormIndex.astype(int)
    numStorms = np.sum(stormIndex==1, axis=1) # axis=1 means summing over rows?
    # stormNest = numStorms >= 1
    # snCount   = sum(stormNest)
    flooded = np.zeros(numNests, dtype=np.int32) # maybe don't need
    #can be zeros, but need to remember 0 is also an index
    whichStorm = np.zeros(numNests, dtype=np.int32)
    # totStormDays = 
    # flP = rng.uniform(low=0, high=1, size=sum(numStorms>0)) # not quite right because prob of flooding stays the same or each nest in different storms
    flP = rng.uniform(low=0, high=1, size=sum(numStorms)) 
    x=0
    # np.savetxt("storm_index.csv",stormIndex, delimiter=",")
    for n in range(numNests):
        # storms = np.zeros(len(stormDays))
        # for s in range(numStorms[n]):
        if numStorms[n] > 0:
            # flood = np.zeros(len(stormIndex[n]), dtype=np.int32)
            flood = np.zeros(len(stormDays), dtype=np.int32)
            # if config.debugFlood: print(f"nest {n} experienced >1 storm")
            # flood = list(range(len(stormIndex[n])))
            for s in range(len(flood)):
                # if config.debugFlood: print("s=", s)
                # print("s=",s)
                if stormIndex[n,s] == 1:
                    # flP = rng.uniform(low=0, high=1, size=1)
                    # flood[s] = flP[x] > pMortFl
                    flood[s] = flP[x] < pMortFl # I changed how pMortFL was defined.
                    x=x+1
                    # if config.debugFlood: print("nest flooded?", flood[s])
                    # print(f"nest {n}: flooded?", flood[s])
            if any(flood.astype(bool)):
                flooded[n] = 1
                # but if default val is 0, could be confused for index 0...
                whichstorm = np.where(flood==1)[0] # first index where val==True
                whichStorm[n] = stormDays[whichstorm[0]]
            # if any(~flood.astype(bool)):

            # x=x+1
            
            # if config.debugFlood: print("first storm occurred while nest was active on:",stormDays[whichStorm[n]])
            # if config.debugFlood: print("--------------------------------------------------------")
    # np.concatenate((np.arange(numNests),numStorms,whichStorm, flooded))
    # stormInfo = np.concatenate((stormInfo, stormIndex), axis=1)
    # need to check whether this is the correct distribution 
    # NOTE: still needs to be conditional on nest having failed already...  
    # NOTE np.concatenate joins existing axes, while np.stack creates new ones
    # np.savetxt("storm_out.csv", np.concatenate(stormInfo, stormIndex))
    # flooded = np.where(pflood>pMortFl, 1, 0) # if pflood>pfMort, flooded=1, else flooded=0 
    # and/or/not don't work bc it's a vector; since it's 1 and 0, can use arithmetic: 
    stormNest = numStorms >= 1
    # snCount   = sum(stormNest)
    # now nests only flood during storms, so not necessary:
    # both give same output, anyway...
    # floodFail = stormNest + flooded > 1 # both need to be true 
    stormInfo = np.zeros((numNests, 3))
    stormInfo[:,0] = numStorms
    # stormInfo[:,1] = stormDays[whichStorm]
    stormInfo[:,1] = whichStorm
    # stormInfo[:,2] = floodFail # true number flooded
    stormInfo[:,2] = flooded # true number flooded
    # print(sum(flooded))
    if config.debugFlood: 
        print("prob of failure due to flooding:", pMortFl)
        print("flooded:", flooded)
        print("storm nests:", stormNest)
        # print("flooded and during storm:", floodFail, floodFail.sum())
        print("flooded and during storm:", flooded, flooded.sum())
        # print(f"ID, #storms, which, fl, stormIndex 1-5:\n", stormInfo)
        print(f"ID, #storms, which, fl, stormIndex 1-5:\n")
        for i, row in enumerate(np.concatenate((stormInfo, stormIndex), axis=1)): 
            print(f"{i}: {row}")
    # nestData[:,4] = floodFail.astype(int) 
    # return(floodFail, whichStorm)
    # return(whichStorm) # if value >0, then nest failed during storm
    return(stormInfo)
# -----------------------------------------------------------------------------
# def mk_fates(nestDat, numNests, hatched, flooded, con=config):
# def mk_fates(nestDat, numNests, hatched, whichS, stormDays, con=config):
def mk_fates(nestDat, numNests, hatched,stormInfo, stormDays, con=config):
    """
    Want number flooded to derive organically from the storm activity, instead of being a preset value

    Runs mk_flood() to update end dates to account for storms 
    Then adds a column for true fate to nest data.

    Returns: nest data with true fate added and end dates for storm nests updated.
    """
    
    trueFate = np.empty(numNests) 
    # if con.debugNests: print(">> hatched:", hatched, sum(hatched))
    #print("length where flooded = True", len(trueFate[floodedAll==True]))
    trueFate.fill(1) # nests that didn't flood or hatch were depredated 
    # flooded, endDate = flooded
    # flooded = numStorm > 0
    # can't just do num storms >0 bc didn't necessarilydie; or which strm > 0 bc 0 is an index
    flooded = stormInfo[:,2].astype(int)
    whichStorm = stormInfo[:,1].astype(int) # now this is the actual storm DAY, not the index
    # np.savetxt("stormInfo.csv", stormInfo, delimiter=",")
    # np.savetxt("nestdata_beforeflood.csv", nestDat, delimiter=",")
    #trueFate[flooded==True] = 2 
    #trueFate[floodedAll==True] = 2 
    trueFate[hatched == True] = 0 # was nest discovered?  
    trueFate[flooded == True] = 2  # should override the nests that "hatched" that were actually during storm
    if con.debugNests: print(">> end date before storms accounted for:", nestDat[:,2])
    # nestDat[:,2][flooded] = stormDays[whichS[flooded]]
    # nestDat[:,2][flooded] = stormDays[whichStorm[flooded]]
    # had to add the ==True for some reason
    nestDat[:,2][flooded==True] = whichStorm[flooded==True]
    if con.debugNests: print("did nest hatch?\n", hatched, sum(hatched), "\ndid nest flood?\n", flooded, sum(flooded))
    
    nestDat = np.concatenate((nestDat, trueFate[:,None]), axis=1)
    #OH, but I don't ever return nestDat anyway. so maybe this should be a function that ADDS true fate to nestDat.

    # now it's the day, not the index, so nests that didn't flood will still be zero but w/o the cost of the mask
    # plus that one didn't seem to be working anyhow..``
    # except we don't want the end date to be zero for everything else  either :/
    # nestDat[:,2] = whichStorm
    # np.savetxt("nestdata_afterflood.csv", nestDat, delimiter=",")
    # if con.debugNests: print(">> end date after:", nestDat[:,2])
    
    # if flooded:
        # nestDat[:,2] = stormDays[whichS]
    # trueFate[floodedAll ] = 2 
    # hatched fate is assigned last, so hatched is taking precedence over flooded
    # so maybe that's why the true DSR is always higher than 0.93, but what about true DSR of discovered?
    # fates = [np.sum(trueFate==x) for x in range(3)]
    # if con.debugNests: print(">>>>> true final nest fates:\n", trueFate, len(trueFate))
    # return(trueFate)
    return(nestDat)

    # # ---- TRUE DSR ------------------------------------------------------------

    # # Calculate proportion of nests hatched and use to calculate true DSR
    # #   daily mortality = num failed / total exposure days
    # #     (num failed =  total-num hatched) 
    # #     (total exposure days = add together survival periods)
    # #   DSR = 1 - daily mortality
    # trueHatch = trueFate==0 # true/false did nest hatch (after storms accounted for)?
    # nestData[:,3] = trueHatch.astype(int)
    
    # trueDSR2 = 1 - ( (numNests - trueHatch.sum()) / survival.sum() ) 
    # if debug: print(">>>> total exposure days (unobserved):", survival.sum())
    # if debug: print(">>>>> and true DSR, calculated correctly:", trueDSR2)
# -----------------------------------------------------------------------------
# ---- NEST DISCOVERY & OBSERVATION ----------------------------------------
# def observer(discProb, numNests, fates, surveyDays, nData, out):
# def observer(discProb, numNests, fateCues, obsFreq, fates, surveyDays, nData, out):
# -----------------------------------------------------------------------------
# How does the observer assign nest fates? 
def assign_fate(fateCuesPresent, trueFate, numNests, obsFr, intFinal, stormFate, cn=config):
    """
    The observer assigns the correct fate based on a comparison of a set 
    probability to random draws from a uniform distribution. If observer is 
    incorrect, then they assign a fate of unknown unless stormFate==True, in
    which case all nests that ended in a period that contained a storm are 
    assumed to have failed due to the storm.
    
        fateCuesProb=random values to compare
        fateCuesPres=probability of fate cues being present
            has different value if there was a storm in final interval.
    
    Returns: vector w/ assigned fate for each nest
    """
    
    assignedFate = np.zeros(numNests) # if there was no storm in the final interval, correct fate is assigned 
    assignedFate.fill(7) # default is unknown; fill with known fates if field cues allow

    fateCuesProb = rng.uniform(low=0, high=1, size=numNests)
    fateCuesPres = np.zeros(numNests)
    fateCuesPres.fill(fateCuesPresent)
    # fateCuesPres[intFinal==True] = 0.1
    fateCuesPres[intFinal > obsFr] = 0.1 # nests with longer final interval have lower chance of cues
    # if cn.debugObs: 
        
    assignedFate[fateCuesProb < fateCuesPres] = trueFate[fateCuesProb < fateCuesPres] 
    if stormFate: assignedFate[intFinal > obsFr] = 2
    if cn.debugObs: 
        print(">> compare random probs to fateCuesPresent:\n", 
              [fateCuesProb,fateCuesPres], 
              fateCuesProb.shape)
        print(">> assigned fates:", assignedFate, sum(assignedFate))
        print(">> nests with storm in final interval:", np.where(intFinal>obsFr))
        print(">> assigned fates after storm fates assigned:", assignedFate, sum(assignedFate))
    # fate cues prob should be affecting all nest fates equally, not just failures.
    # if debug: print(">> proportion of nests assigned hatch fate:", np.sum((assignedFate==0)[discovered==True])/(sum(discovered==True)),"vs period survival:", pSurv**hatchTime)
    # print(">> assigned fate array & its shape:\n", assignedFate, assignedFate.shape)
    return(assignedFate)
# -----------------------------------------------------------------------------
def svy_position(initiation, nestEnd, surveyDays, cn=config):
    """ Finds index in surveyDays of iniatiation and end dates for each nest """
    # if cn.debugObs: print(">> initiation dates:\n", initiation)
    position = np.searchsorted(surveyDays, initiation) 
    # if cn.debugObs: print(">>>> position of initiation date in survey day list:\n", position, len(position)) 
    # if cn.debugObs: print(">> end dates:\n", nestEnd)
    position2 = np.searchsorted(surveyDays, nestEnd)
    # if cn.debugObs: print(">>>> position of end date in survey day list:\n", position2, len(position2)) 
    
    surveyDays = dict(zip(np.arange(len(surveyDays)), surveyDays))
    # if cn.debugObs: print(">> survey days with index number:\n", surveyDays)
    
    return((position, position2)) # return a tuple
    # position2
# -----------------------------------------------------------------------------
# def observer(par, fateCues, fates, surveyDays, nData, out):
# def observer(nData, par, cues, fate, surveys, out, cn=config):
# @profile
def observer(nData, par, cues, surveys, out, cn=config):
    """
    The observer searches for nests on survey days. Surveys til discovery (success)
    are calculated as random draws from a negative binomial distribution with
    daily success probability of discProb. If surveys til discovery is less
    than total number of surveys while nest is active, then nest is discovered.

    The observer then assigns fate in assign_fate. 
    
    Remember, pos[0] is the first survey after initiation, and pos[1] is the first survey after end.

    output: ndarray w/ nrows=numNests. 
    cols= i, j, k, assigned fate, num *normal* obs ints, intFinal
    """
    initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
    numNests, obsFreq, discProb, stormFate = par # unpack par
    surveyDays, surveyInts = surveys

    pos = svy_position(initiation, end, surveys[0])
    num_svy          = pos[1] - pos[0]   
    svysTilDiscovery = rng.negative_binomial(n=1, p=discProb, size=numNests) # see above for explanation of p 
    discovered       = svysTilDiscovery < num_svy
    num_svy[~discovered] = 0
    # stormIntFinal    = surveyInts[pos[1]] > obsFreq  # was obs interval longer than usual? (== there was a storm)
    intFinal    = surveyInts[pos[1]] # actual length of final interval for each nest
    kVal = surveyDays[pos[1]]
    jVal = surveyDays[pos[1]-1]
    jVal[fate==0] = kVal[fate==0]
    out[:,0] = surveyDays[pos[0]+svysTilDiscovery] # i
    out[:,1][discovered] = jVal[discovered] 
    out[:,2][discovered] = kVal[discovered]
    out[:,3] = assign_fate(cues, fate, numNests, obsFreq, intFinal, stormFate)
    out[:,4] = num_svy - svysTilDiscovery  # number of observations for the nest
    out[:,5] = intFinal.astype(int) # length of final interval - transform to integer for the ndarray
    if cn.debugObs: 
        print("surveys til discovery; discovered T/F, total obs days, total active days:")
        for i in range(len(out)):
            print(f"{i:02}: {svysTilDiscovery[i]} | {discovered[i]} | {(out[:,2]-out[:,0])[i]} | {(nData[:,2]-nData[:,1])[i]}")
    return(out)

# -----------------------------------------------------------------------------
# @profile
def make_obs(par, storm, survey, config=config):
    """
    1. Call functions mk_nests, mk_per, storm_nest, mk_flood, mk_fates, & observer
    2. Combine the output into an array: 
          [0]:nest ID....................[1]:initiation............[2]:survival(w/o storm)..
          [4]:first found.............[5]:last active..........[6]:last checked........
          [7]:assigned fate.......[8]:num obs int......[9]:days in final interval..............

    Returns:
        numpy ndarray containing nest & observation data (column indices above)
        
        Can also uncomment lines to save nest data to .npy file
        
        And other lines to make nest data that's compatible with the old script.
    """
    nd       = np.zeros(shape=(par.numNests, 3), dtype=int)
    nd2      = np.zeros(shape=(par.numNests, 6), dtype=int)
    # fateCues directly correlates to obsFreq, so doesn't need to be param
    fateCues   = 0.65 if par.obsFreq > 5 else 0.71 if par.obsFreq == 5 else 0.75
    # NOTE should I make sure all nests live for at least a day?
    # ---- make the nests: ---------------------------------------------------
    # nData          = mk_nests(par=par, init=init[0], 
    # nestfile = Path(uniquify(Path.home()/
    #                         'C://Users/sarah/Dropbox/Models/sim_model/other'/
    #                         # dirName/
    #                         ('nest_data.npy')))
    # nData          = mk_nests(par=par, nestData=dfs[0])
    nData          = mk_nests(par=par, nestData=nd)
    # nestPeriod     = mk_per(nData[:,1], (nData[:,1]+nData[:,2]))
    nestPeriod     = mk_per(nData[:,1], (nData[:,2])) # changed output of mk_nests 
    stormOut  = storm_nest(par.stormFrq, nestPeriod, storm)
    # stormNest = np.any(stormOut[] == 1, axis=1) 
    # flooded        = mk_flood(par, stormsPerNest, numNests=par[par.numNests])
    # flooded        = mk_flood(stormDays, par.pMortFl, stormOut, numNests=par.numNests)
    # output from mk_flood has 3 cols
    # whichStorm = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests)
    stormDat = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests)
    # inclFlood        = mk_flood(nData, par.pMortFl, stormsPerNest, numNests=par.numNests)
    # flooded        = stormDat[:,2] # need more than just whether nest flooded; need date
    hatched        = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
    if config.debugNests: print("hatched (before storms)=", hatched, sum(hatched))
    # nestFate       = mk_fates(nData, par.numNests, hatched, flooded)
    # nestFate       = mk_fates(nData, par.numNests, hatched, whichStorm, storm)
    # nestFate       = mk_fates(nData, par.numNests, hatched, stormDat, storm)
    nData      = mk_fates(nData, par.numNests, hatched, stormDat, storm)
    # hatched        = (nData[:,2]-nData[:,1]) >= par.hatchTime
    # ---- observer: ---------------------------------------------------------
    par2      = [par.numNests, par.obsFreq, par.discProb, par.stormFate]
    # obs       = observer(nData, par=par2, cues=fateCues, fate=nestFate, 
    obs       = observer(nData, par=par2, cues=fateCues, surveys=survey, out=nd2)
                        #  surveys=surveyDays, out=dfs[1])
    # ---- concatenate to make data for the nest models: ---------------------
    # fl, ha, ff, la, lc, nInt, sTrue = obsDat
    # disc = "True  True  True  True  True  True False  True  True  True  True False True  True False  True False  True  True  True  True  True  True  True False  True  True  True  True False"
    # stildisc = "1 1 3 0 0 1 1 0 1 0 2 2 1 0 0 1 1 0 1 0 0 0 0 0 3 0 1 0 0 0"
    # sTilDisc = ','.join(stildisc)
    # disc1 = np.fromstring(disc, dtype=bool, sep=' ')
    # disc = disc.replace(" ", ",")
    # disc = disc.split()
    # disc1 = np.fromstring(disc, dtype=bool, sep=',')
    # disc1 = np.array(disc)
    # disc1 = (disc1 == "True")
    # sTilDisc = np.fromstring(stildisc, dtype=int, sep=' ')
    nestData = np.concatenate((nData, 
                            #    nestFate[:,None], 
                            #    np.zeros((par.numNests,4)),
                               obs
                            #    stormOut[0][:,None] # storms per nest
                            #    stormOut # is this need here?? also, supposed to be stormDat??
                            #    stormDat # on't think these are ever used
                            #    np.zeros((par.numNests,2))
                               ), axis=1)
    # nestData[:,3] = (nestFate==0)
    # nestData[:,4] = (nestFate==2)
    # nestData[:,5] = sTilDisc
    # nestData[:,6] = disc1
    # nestData[:,12] = nestData[:,12] > 3
    # nestData[:,14] = 416 # exposure dayso
    # nestData[:,15] = nestData[:,2] - nestData[:,1]
    
    # if config.debug: print("nestData:\n", nestData)
    if config.debugNests: print("\nnestData:\n", nestData)
    # np.savetxt("nestdata_afterflood.csv", nestData, delimiter=",")
    # np.save("nest_data.npy", nestData)
    # np.save(nestfile, nestData)
    return(nestData)
# -----------------------------------------------------------------------------
#   MAYFIELD & JOHNSON
# -----------------------------------------------------------------------------
def calc_exp(inp, expPercent=0.5, cn=config): 
    """
    Calculate the exposure period for a nest (number of days observed)
        
    Arguments
    ---------
    inp = [i,j,k] for all nests in set\n
    default expPercent is from Mayfield; Johnson recommended 0.4
    
    Returns
    -------
    ndarray. nrows=len(inp); cols=alive_days, final_int, exposure
    
    More info
    ----------
        > the ijk values should tell you failed vs hatched
        > I think I couldn't get it to work as vectorized, so I used a loop
    For the basic case where psurv is constant across all nests and times:
      1. count the total number of alive days when nest was observed
      2. count the number of days in the final interval (for failed nests)
      3. calculate the exposure
            > #days under observation before final int + (#days in final int * expPercent)
                > expPercent = percent of final interval nest is assumed alive
                > Mayfield used 50%, Johnson corrected it to 40%
                > final interval assumed to be zero days for hatched nests, which
                 were found after hatching (exposure of incubation period is over)
            > not calculating nestling exposure bc precocial/semi-precocial chicks
              leave the nest so early 
    """
    # expo = np.zeros((numNests, 3))
    expo = np.zeros((len(inp), 3))
    # inp = ijk (cols) for all nests (rows)
    for n in range(len(inp)-1): # want n to be the row NUMBER
        # alive_days = inp[n,1] - inp[n,0] # interval from first found to last active 
        expo[n,0] = inp[n,1] - inp[n,0] # interval from first found to last active 
                                         # all nests are KNOWN to be alive
        # expo[n,0] = expo[n,0] - 1 # since this is essentially 1-day intervals, 
                                    # need 1 fewer than total number
        expo[n,1] = inp[n,2] - inp[n,1] # interval from last active to last checked
        # if expo[n,1]!=0: expo[n,1] = expo[n,1]- 1 # for hatched nests, stays 0
        # exposure = sum(alive days) + days in final int * expPercent
        expo[n,2]   = expo[n,0] + (expo[n,1]*expPercent)
        # NOTE need nests to be alive for at least one interval
    # if cn.debugM: print("output from exposure function:", expo)
    return(expo)
# -----------------------------------------------------------------------------
# def mayfield(ndata, expo):
# def mayfield(num_fail, expo, all=False):
def mayfield(num_fail, expo):
    """ 
    The Mayfield estimator of DSR 
    
    Mayfield's original estimator was defined as: 
            > DSR = 1 - (# failed nests / # exposure days)
    so if DSR = 1 - daily mortality, then:
            > daily mortality = # failed nests / # exposure days
    
    Arguments:
        num_fail = count of failed nests (total-hatched)
        expo     = just the exposure days output of calc_exp() (out[:,2])

    Returns: the daily mortality 
    """
#    I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
#    dat = ndata[
    # hatched = np.sum(ndata[:,3])
    # failed = len(ndata) - hatched
    # expo is output from exposure function
    # exposure = expo[:,2]
    # hatch  = sum(ndata[:,3==0])
    # fail   = ~hatch 
    # fail = sum(ndata[:,3]!=0) # number of failed nests
    
    # failExp = sum()
    # if debug: print(">> exposure percentage for final interval:", expPercent)
    # if debug: print(">> hatch:\n",hatch,"\n>> and fail:\n",fail) 
    # if debug: print(">>>> Calculate Mayfield estimator.")
    # if debug: print(">> number of nests hatched:", hatched, "and failed", failed)
    # mayf = failed / (hatched + 0.5*failed)
    # mayf = failed / (hatched + (expPercent*failed))
    mayf = num_fail / (expo.sum())
    # if debug: print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

    return(mayf)
# -----------------------------------------------------------------------------
def johnson(ndata, srn):
    """
    NOTE: Johnson (1979) provided a mathematical derivation that allowed the 
          calculation of variance for the estimate.
    He ALSO came to the conclusion that the Mayfield method is pretty much 
    equivalent to his ML estimator, w/ adjustment for long intervals.
    > for a single day:
       > probability of survival is s    
       > probability of failure is (1-s)
    > for interval of length k days:
       > prob of survival is s**k 
       > prob of failure is s**(1/2k-1)(1-s)
    > ex. - prob of a nest surviving three days and failing on the fourth is:
            s*s*s*(1-s) 
        > this assumes that a failed nest survived half (minus a day)
          of interval and then failed
    Johnson's rewriting of the Mayfield estimator:
           mortality = (f1 + sum(ft)) / (h1 + sum(t*ht) + f1 + 0.5*sum(t*ft)) 
    > created by differentiating the log-likelihood equation and setting to 
      zero (maximizing)
    > ht = hatched or survived til next visit; ft = failed by next visit
    > f1 and h1 represent an interval between visits of one day, which is not 
      used in our studies
      > so we end up with: sum(ft) / (sum(t*ht) + 0.5*sum(t*ft)) 
             where t = interval length, and 
             f and h represent number of failures and hatches, respectively
    Johnson's Mayfield-40 estimator: 
           mortality = sum(ft) / (sum(t*ht) + 0.4*sum(t*ft))
    Johnson's modified ML estimator:
           1/s*(sum(t*ht)) = sum( (t * ft * s^t-1) / (1 - s^t))
    -----------------------------------------------------------------------------

    """
    jEst = (1/srn) * sum()
# -----------------------------------------------------------------------------
# @profile
def calc_dsr(nData, nestType):
    """ 
    Calculate exposure and DSR for a given set of nests. 

    pass i,j,k from nest data to calc_exp() and then run mayfield()
    
    Returns DSR value. 
    """
    nNests  = len(nData)
    # hatched = len(nData[:,3] == 0)
    # failed  = nNests-hatched
    
    # expDays = exposure(nestData[:,6:9], numNests=numN, expPercent=0.4)
    # expDays = calc_exp(nData[:,6:9], expPercent=0.4)
    if nestType=="all":
        expDays = sum((nData[:,2]-nData[:,1]))
        hatched = sum(nData[:,3] == 0)
    else:
        expDays = calc_exp(nData[:,4:7], expPercent=0.4)
        expDays = expDays[:,2]
        hatched = sum(nData[:,7] == 0)
    dmr     = mayfield(num_fail=nNests-hatched, expo=expDays)
    return(1-dmr)

    # def pr_fates_dsr(nData, expo, trueDSR, nestType):
                # if debug:
                #     print(
                #         "> DISCOVERED NESTS - total | analyzed: hatched:", 
                #         discovered, "|", analyzed,
                #         # "excluded from analysis:", excluded,
                #         "failed:", failed, "|", failed2,
                #         # nestData.shape[0] - sum(nestData[:,3])
                #         # "exposure days:", expDays, "|", sum(nestData[:,15])
                #         "true DSR:", trueDSR_disc, "|", trueDSR_an
                #         )
# -----------------------------------------------------------------------------
# --- PRINT FUNCTIONS ---------------------------------------------------------
# -----------------------------------------------------------------------------
def print_observer(svysTilDiscovery, discovered):
    print("surveys til discovery; discovered T/F:", svysTilDiscovery, discovered)
    # if cn.debugObs: print("nestID, init, end, tfate, i, j, k, afate, nnobs, intFin:\n", 
    #                       np.concatenate((nData,fate[:,None],out), axis=1))
    # if cn.debugObs: print("total observed days:\n", out[:,2]-out[:,0], 
    #                       "& total days nest was active:\n", nData[:,2] - nData[:,1])
    
# -----------------------------------------------------------------------------
# def print_prop(assignedFate, trueFate, discovered):  
def print_prop(nData):  
    assignedFate, trueFate = nData[:,7], nData[:,3]
    discovered = nData[:,4] != 0
    aFates = [np.sum((assignedFate == x)[discovered==True]) for x in range(4)]
    # this proportion needs to be out of nests discovered AND assigned
    aFatesProp = [np.sum((assignedFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
    tFates = [np.sum((trueFate == x)[discovered==True]) for x in range(4)]
    tFatesProp = [np.sum((trueFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
    print(
            ">> assigned fate (hatched, depredated, flooded, unknown):", 
            # ">> assigned fate proportions (hatched, depredated, flooded, unknown):", 
            aFatesProp[0:3], 
            (np.sum(discovered==True) - np.sum(aFates)) / (np.sum(discovered==True)),
            # (np.sum(aFates==7) )/ (np.sum(discovered==True)),
            "\n\n>> proportions of known (assigned) fates (H, D, F):",
            aFates[0:3]/np.sum(aFates),
            "\n>> vs. actual proportions for discovered only (H, D, F):",
            tFatesProp[0:3],
            # np.sum(discovered==True)- np.sum(tFates)
            )
    # nestData[:,10] = assignedFate
    # expDays = survival.sum()
    # fail = ~trueHatch
    # expF = survival[fail==True].sum()
    # if debug: print(">> exposure days:", expDays) # why make this a separate column?
                                                #   could just sum survival column?
    # exp = np.zeros(numNests)
    # exp.fill(expDays)
    # exposure varies based on the nest fate (failed nests are assumed to have 
    # survived half, or 60% for Johnson, of the final interval)
    # nestData[:,14] = exp
 
    # numDisc = discovered.sum()
    # numDiscH = trueHatch[discovered==True].sum()
    # survDisc = survival[discovered==True].sum()
    # if debug: print(">> number discovered:", numDisc, ", number discovered that hatched:", numDiscH, ", and exposure days (discovered nests):", survDisc)
    # trueDSR_disc = 1 - ((numDisc - numDiscH) / survDisc) # num failed / total exposure days
    # if debug: print(">> true DSR of discovered nests only:", trueDSR_disc) 
    # # NOTE issue may be that not enough flooded nests are discovered, not that too many hatched nests are
    
    # print(">> nest discovered?", discovered)
    # print(">> discovered & hatched:", discovered[trueHatch==True])
    # print(">> calculate exposure days for discovered nests by summing this list:", survival[discovered==True])
    #trueDSR_disc = 1 - ((discovered.sum() - hatched.sum()) / survival.sum()) 

    # nests = np.stack((discovered, trueFate, firstFound, lastActive, lastChecked, totalReal, numStorms))
    # nests = np.transpose(nests)
    # print(">> discovered?, fate, i, j, k, num surveys, num storm intervals:\n", nests)
    # print(
    #     ">> discovered?, fate, i, j, k, num surveys, num storm intervals:\n", 
    #     nests[:5,:], 
    #     "\n ... \n",
    #     nests[-5:,:]
    #     )

    #@#print(">> nest data:\n----id--ini-end-hch-fld-std-dsc-i--j--k-fate-nobs-sfin-nstm\n", nestData)
# -----------------------------------------------------------------------------
def print_nd(nestData, nDisc, pSurv, hatchTime):
    # if debug_nest: print("nestData, discovered only:\n", nestData)
    print("nestData, discovered only:\n", nestData)
    print(">> proportion of nests assigned hatch fate:", 
                    np.sum(nestData[:,3]==0)/(nDisc),
                    "vs period survival:", 
                    pSurv**hatchTime)

    print(
        ">> nests w/ only 1 obs while active:",
        np.where(nestData[:,6] == nestData[:,7]), # where i==j
        "& nests w/ unknown fate:",
        np.where(nestData[:,9] == 7)
        ) 
                # if debug:
                #     print(
                #         "\n>> assigned DSR:",
                #         pSurv,
                #         "true DSR of all nests:", 
                #         trueDSR, 
                #         "discovered nests:",
                #         trueDSR_disc,
                #         "and nests used in analysis:", 
                #         trueDSR_an
                #         )

def print_mayf(expo):
    print("output from exposure function:", expo)
    
    for n in range(len(expo)):
        print(
            "days nest was alive:", expo[n,0],
            "& final int:", expo[n,1], 
            "& exposure:", expo[n,2]
            )
        
        
    # if debug:
    #     print(
    #         # f"> {nestType} nests - hatched:", hatched.sum(),
    #         f"> {nestType} nests - hatched:", hatched,
    #         "; failed:", nNests - hatched, 
    #         # "; exposure days:", expDays[:,2].sum(),
    #         "; exposure days:", expDays.sum(),
    #         "; & Mayfield-40 DSR:", 1-dmr
    #         )
# -----------------------------------------------------------------------------
#   PROGRAM MARK 
# -----------------------------------------------------------------------------
# The model used in Program MARK is based on Dinsmore (2002) -  
#        allows for variance in DSR & use of covariates

# These functions are based on info in 'Program MARK: A Gentle Introduction' 

# It also has a wrapper function that transforms the initial optimizer values
# using the logistic function.
# This way, the optimizer can work over the range of -infinity:infinity, but
# the values fed to the function are between 0 and 1 (probabilities)

# Lastly, it has a function to generate the probabilities before running the optimizer 
# on the MARK function, so I can take the for loop out of the function that is optimized.

# -----------------------------------------------------------------------------
# def prog_mark(s, ndata, probs, nocc, con=config):
# @profile
def prog_mark(s, ndata, nocc, con=config):
    """
    Run the Program MARK algorithm
    1. Grab the data for the input for MARK 
          > First, grab only discovered nests
          > Then, only the needed columns
            (nest ID, first found, last active, last checked, assigned fate)
             inp[0] = ID | inp[1] = i | inp[2] = j | inp[3] = k | inp[4] = fate `
    2. Extract rows where j minus i does not equal zero (nest wasn't only 
       observed as active for one day)
          > Model requires all nests to have at least two observations while active

    """
    # prob, dof = probs
    # allp, alldof = mark_probs(s=s, ndata=ndata)
    # ALL IN ONE FUNCTION:
    allp   = np.array(range(1,len(ndata)), dtype=np.longdouble) # all nest probabilities 
    expo = calc_exp(inp=ndata[:,4:7], expPercent=0.4)
    for n in range(len(ndata)-1): # want n to be the row NUMBER
        alive_days = expo[n,0] - 1
        final_int  = expo[n,1] - 1
    
        if final_int > 0: # final int for hatched nests == 0
            p   = (s**alive_days)*(1-(s**final_int)) 
        else:
            p   = s**alive_days
        allp[n]   = p # NOTE this line is throwing the Deprecation Warning
    nll = sum(-np.log(allp))
    # SEPARATE FUNCTION:
    # allp = mark_probs(s=s, ndata=ndata)
    # nll = sum(-np.log(allp))
    
    # print("all nests:", len(ndata)) 
    # disc = ndata[np.where(ndata[:,6]!=0)] # i != 0 # should already have filtered out unobserved nests
    # inp = disc[:,np.r_[0,7:11]] # doesn't include index 11
    # inp = disc[:,np.r_[0,4:8]] # doesn't include index 11
    # inp = ndata[:,np.r_[0,4:8]] # doesn't include index 11
    # l    = len(inp)
    # incompatible nests have already been excluded
    # inpInd = inp[(inp[:,2]-inp[:,1]) != 0] # access all rows; 
    #                                     create mask, index inp using it
    # allp = prob[inpInd]
    # alldof = dof[inpInd]
    # lnp  = -np.log(allp) # vector of all log-transformed probabilities
    # NOTE these if statements take up lots of time, esp inside the optimizer
    # if con.debugM:
    #     print(">>>>> Program MARK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     print("> number of nests:", len(ndata), "discovered nests:", len(disc))
    #     print("inp (ID, i, j, k, fate:)\n",inp)
    #     print("l=", l, "| s=", s, "| nocc=", nocc)
    #     print("----------------------------")
    #     print(">> all nest cell probabilities:\n", allp)
    #     print(">> all degrees of freedom:\n", alldof)
    #     print("log of all nest cell probabilities:", lnp)
    #     print(">> sum to get negative log likelihood of the data:", NLL)
    #print(">> negative log likelihood of each nest cell probability:", lnp)
    #lnSum = lnp.sum()
    #NLL = -1*lnp.sum() # all caps are some kind of Python constant? in the style guide
    # nll = Decimal(0.0)
    # nll = ne.evaluate('sum(-log(allp))')
    # NLL = Decimal(0.0)
    # NLL = lnp.sum()
    return(nll)
# -----------------------------------------------------------------------------
def mark_wrapper(srn, ndata, nocc):
    """
    This function calls the program MARK function when given a random starting 
    value (srn) and some nest data (ndata)
        > values given to optimizer are transformed then passed to MARK function
            > allows larger range of values for optimizer to work over w/o overflow
            > but values given to the function are still between 0 and 1, as required
        > Create vector to store the log-transformed values, then fill

    """
    # s = np.ones(numNests, dtype=np.longdouble) # why is s an array?
    s = logistic(srn)
    #@#print("logistic of random starting value for program MARK:", s, s.dtype)
    # the logistic function tends to overflow if it's a normal float; make it np.float128
    ret = prog_mark(s, ndata, nocc)
    #@#print("ret=", ret)
    return ret
# -----------------------------------------------------------------------------
#   THE LIKELIHOOD FUNCTION
# -----------------------------------------------------------------------------
#def like_old(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData, stormDays, surveyDays, obs_int):
def like_old(argL, obsFreq, nestData, surveyDays, stormDays, numNests):
    """

    This function computes the overall likelyhood of the data given the model parameter estimates.
    
    The model parameters are expected to be received in the following order:
    - a_s   = probability of survival during non-storm days
    - a_mp  = conditional probability of predation given failure during non-storm days
    - a_mf  = conditional probability of flooding given failure during non-storm days
    - a_ss  = probability of survival during storm days
    - a_mfs = conditional probability of predation given failure during storm days
    - a_mps = conxditional probability of predation given failure during storm days
    - sM    = for program MARK?

    """
  
    nCol = 18
    a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
    obs_int = obsFreq
    likeData = np.zeros(shape=(numNests, nCol), dtype=np.longdouble) 

    stillAlive  = np.array([1, 0, 0])
    mortFlood   = np.array([0, 1, 0])
    mortPred    = np.array([0, 0, 1])

    # > starting matrix, from etterson 2007:
    startMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
    # > use this matrix for storm weeks:
    stormMatrix = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]]) 
        # how is this matrix actually being incorporated during the analysis?

    logLike = Decimal(0.0)          # initialize the overall likelihood counter
    #logLike = float(logLike)
    rowNum = 0
    for row in nestData:
    # columns for the likelihood comparison datarframe:
    #         num obs, log lik of nest, log lik period 1, period 2, etc.  
        
    # FOR EACH NEST -------------------------------------------------------------------------------------

        nest    = row         # choose one nest (multiple output components from mk_obs())
        # print('obs_int check: ', obs_int)

        # disc    = nest[7].astype(int)    # first found
        disc    = nest[0].astype(int)    # first found
        # endObs  = nest[9].astype(int)    # last observed
        endObs  = nest[2].astype(int)    # last observed
        # hatched = nest[3]
        # flooded = nest[4]
        hatched = nest[3] == 0
        flooded = nest[3] == 2

        if flooded == True & hatched == False:
            fate = 2
        elif flooded == False & hatched == False:
            fate = 3
        else:
            fate = 1
        # fate    = nest[7].astype(int)    # assigned fate

        if np.isnan(disc):
            ###print("this nest was not discovered but made it through")
            continue
        
        num     = len(np.arange(disc, endObs, obs_int)) + 1 # number o observations
        # print('#############################################################')
        # print('nest =', nest[0], 'row number =', rowNum, 'number of obs =', num)

        likeData[rowNum, 0:3]  = np.array([nest[0], fate, num] )

        # print("num=", num)
        obsDays = in1d_sorted(
            # (np.linspace(disc, endObs, num=num)), surveyDays)
            np.linspace(disc, endObs, num=num),
            surveyDays)
        # print("obs days for nest:", obsDays)
        obsPairs = np.fromiter(
            itertools.pairwise(obsDays), 
            dtype=np.dtype((int,2))
            ) # do elements of numpy arrays have to be floats?
        # print("date pairs in observation period:", obsPairs) 

        # > make a list of intervals between each pair of observations 
        #   (necessary for likelihood function)
        intList = obsPairs[:,1] - obsPairs[:,0]
        # print("interval list:", intList)
        
        # > start off with all intervals = alive
        obs     = [stillAlive for _ in range(len(obsPairs)+1)] 

        # > change the last obs if nest failed:
        if fate == 2:
            obs[-1] = mortFlood
        elif fate == 3:
            obs[-1] = mortPred

        # print("fate, obs = ", fate, " , ", obs) # check that last entry in obs corresponds to fate

        # if hatch, leave as is?

        # place this likelihood counter inside the for loop so it resets 
        # with each nest:
        logLikelihood = Decimal(0.0)   
        #logLikelihood = float(logLikelihood)

        # likeData[0, rowNum] = nest[0]
        obsNum = 0
        for i in range(len(obs)-1):
        # FOR EACH OBSERVATION OF THIS NEST ---------------------------------------------------------------------

            # print("observation number:", obsNum)
            
            intElt  = (intList[i-1]).astype(int)  # access the (i-1)th element of intList,
                                    # which is the interval from the (i-1)th
                                    # to the ith observation

            #stateF  = obs[i]
            stateF  = obs[i+1] 
            stateI  = obs[i]
            # print("stateF:",stateF)
            TstateI = np.transpose(stateI)
            # print("TstateI:", TstateI)

            # if any(d in storm_days for d in range(i-1, i)):
            if any(d in stormDays for d in range(i-1, i)):
                # if any of the days in the current observation interval (range) is also in storm days, use storm matrix
                # print("using storm matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(stormMatrix, intElt))
                # this is the dot product of the current state of the nest and the storm matrix ^ interval length
           # look into using @ instead of nest dot calls 
            else:
                # print("using normal matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(startMatrix, intElt))

            lPer = np.dot(lDay, TstateI)
            # print("likelihood for this interval:", lPer)

            logL = Decimal(- np.log(lPer))
            # print("negative log likelihood of this interval:", logL)

            #logL = float(logL)

            logLikelihood = logLikelihood + logL # add in the likelihood for this one observation
            # print("log likelihood of nest observation history:", logLikelihood)
            colNum = obsNum + 4 
            likeData[rowNum, colNum] = logL
            obsNum = obsNum + 1

        likeData[rowNum,3] = logLikelihood
        logLike = logLike + logLikelihood        # add in the likelihood for the observation history of this nest
        rowNum  = rowNum + 1
        # print("increment row number:", rowNum)
        # print("overall log likelihood so far:", logLike)
        

    # print(
    #     "nest num, fate, num obs, lik(obs hist), lik(each obs int) ... \n",
    #     likeData[:10,:] # print first ten rows
    # )
    return(logLike) 
# -----------------------------------------------------------------------------
# def state_vect(numNests, flooded, hatched):# can maybe calculate these only once
def state_vect(nNest, fl, ha):# can maybe calculate these only once
# def state_vect(flooded, hatched):# can maybe calculate these only once
    """
    numNests is not the total number (param value) but the number not excluded
    1. Define state vectors (1x3 matrices) - all the possible nest states 
          [1 0 0] (alive)   [0 1 0] (fail-predation)   [0 0 1] (fail-flood) 
    ---------------------------------------------------------------------------------------------------
    2. Create arrays to hold state vectors for all nests:
       a. state of nest on date nest was first found (stateFF)
       b. state of nest on date nest was last checked (stateLC) - this is the fate as observed
    Could also calculate bassed on nest fate value
    """
    stillAlive = np.array([1,0,0]) 
    mortFlood  = np.array([0,1,0])
    mortPred   = np.array([0,0,1])
    # FOR THE INITIAL STATE (stateFF), just one vector (see notebook) - later in code
    # numNests = len
    stateEnd    = np.empty((nNest, 3))     # state at end of normal interval
    stateLC     = np.empty((nNest, 3))     # state at end of final interval
     # > use broadcasting - fill doesn't work with arrays as the fill value:
    stateEnd[:] = stillAlive # alive at end of normal interval
    stateLC[:]  = mortPred   # default is still depredation

    stateLC[fl==True] = mortFlood  # flooded status gets flooded state vector
    stateLC[ha==True] = stillAlive # hatched nests stay alive the entire time

    # nests always start alive, or they wouldn't be checked
    # TstateI = np.transpose(stillAlive)  # this is just one, not a vector?
    
    return([stateEnd, stateLC])
    ##print(">> transpose of initial state vector:", TstateI, TstateI.shape)
    #                                             this will depend on how many storms/when they are
    # NOTE need the transpose of each row, not entire matrix. 
    # print(">> state at the end of a normal interval:\n",stateEnd) 
    # print(">> state on final nest observation:\n", stateLC)
    # ---------------------------------------------------------------------------------------------------
    # 5. Compose the matrix equation for one observation interval.
    #        The formula used is from Etterson et al. (2007) 
    #    For this, you need the nest state at the beginning and end of the interval, plus interval length
    #    > intElt - length in days of the observation interval being assessed 
    #    > initial state (stateI) - state of the nest at the beginning of this interval 
    #    > stateF - state of the nes at the end of this interval
    #    There is a transition matrix that is multiplied for each day in the interval 
    #    > in this case, the nest started the interval alive and ended it alive as well 
    #    > daily nest probabilities: s - survival; mp - mortality from predation; mf - mortality from flooding 
    #    > these are daily probabilities, so raise transition matrix to the power of number of days in interval  
    #
    #                                     _         _  intElt           _   _ 
    #              [ 1 0 0 ]             |  s  0  0  |                 |  1  | 
    #                             *      |  mp 1  0  |            *    |  0  | 
    #                                    |_ mf 0  1 _|                 |_ 0 _|  
    #                              
    #      {  transpose(stateI) * trMatrix, raised to intElt power * stateF } 
    #
    # Then, you can multiply this equation times number of intervals (numIntTotal)
    #    Single in  interval --> all intervals --> likelihood

    # in the following code, we calculate all matrix multiplications for all nests, and then turn them on/off
    # based on whether we need them. Would it be faster to only do the multiplications we need? That would require indexing
    # maybe calculate the one for normal and storm intervals (not final interval) once, and the final intervals for each nest
    # (based on what the actual fate is)
    # > all nests start interval as alive - don't need stateFF
    # > transition matrix, from etterson 2007:
    # the values being optimized are integral to the matrices, so can't pull that out
    # NOTE NOTE which of these are vectorized and which aren't??
# def nest_mat(a_s, a_mf, a_mp, obsFreq, stormTrue):
# def interval(pwr, stateEnd, stateLC ): 
# ---------------------------------------------------------------------------------------------------
# def nest_mat(argL, obsFreq, stormFin, useStormMat, config=config):
# @profile
def nest_mat(argL, obsFreq, stormFin, useStormMat, config=config):
    """
    Purpose
    -------
    1. Create transition matrix for normal intervals & storm intervals
    2. Raise transition matrix to the power of interval length
    - intervals with storms are longer (obs_int x 2)
    - there is no separate storm matrix anymore (see notes)
    
    Inputs
    ------
    - argL = vals for the minimizer
    - useStormMat = T/F should you use the storm transition matrix for storm intervals
    
    Returns
    -------
    - list containing the two matrices [pwr, pwrStm]

    """
    
    # a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
    a_s, a_mp, a_mf = argL
    # if debug: print("observation interval, storm in final interval?, use storm matrix?\n",obsFreq, stormFin, useStormMat)
    trMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
    pwr = np.linalg.matrix_power(trMatrix, obsFreq) # raise the matrix to the power of the number of days in obs int
    # storm matrix just has a longer observation interval
    pwrStm = np.linalg.matrix_power(trMatrix, obsFreq*2) 

    return([pwr, pwrStm])
# def logL(normalInt, normalFinal, stormFinal, numInt):
# ---------------------------------------------------------------------------------------------------
# def interval(pwr, stateMat, cn=config): 
# def interval(pwr,numNests, fl, ha ,cn=config): 
# @profile
def interval(pwr, numNests, fl, pr, cn=config): 
# def interval(pwr,numNests,sNest, fl, ha ,cn=config): 
    """
    Purpose
    -------
    Create matrix multiplication to get the likelihood for each type of interval:
    - active nests in all intervals start as alive
    - regular (alive-->alive), storm(alive-->alive during storm -- rare),
      final (alive-->failed), storm final (alive-->flooded)
    - final interval for hatched nests = 0
    - longer obs int for nests that survived storms should already be accounted for? 

    Arguments
    ---------
    - pwr = output from nest_mat()
    - stateMat = array of 1x3 end-of-interval and final matrices for all nests
    
    Returns
    -------
    list of 4 likelihood values, one for each type of interval:
    [normalInt, stormInt, finalInt, stormFinal]

    """
    stillAlive = np.array([1,0,0]) 
    mortFlood  = np.array([0,1,0])
    mortPred   = np.array([0,0,1])
    
    # nests always start alive, or they wouldn't be checked
    TstateI = np.transpose(stillAlive)  # this is just one, not a vector? yes
    pwrN, pwrS = pwr
    
    # --------------------------------------------------------------------------
    # DO MATRIX MULTIPLICATION ONCE AND BROADCAST TO ARRAY:
    # matrix multiplication resolves to a single value:
    oneNormInt = stillAlive@pwrN@TstateI
    oneFinalPr = mortPred@pwrN@TstateI
    oneFinalSt = mortFlood@pwrS@TstateI
    
    normalInt  = np.empty((numNests))
    finalInt   = np.ones((numNests))
    normalInt.fill(oneNormInt)
    finalInt[pr] = oneFinalPr
    finalInt[fl] = oneFinalSt
    # there is no final interval for hatched. log(1) will become zero beforee summing
    # --------------------------------------------------------------------------
    # DO MATRIX MULTIPLICATION ON THE ARRAYS:
    # stateEnd, stateLC = stateMat
    # # if cn.debugLL: print(">> end state of normal / final interval:\n", stateEnd, "/", stateLC)
    # # if cn.debugLL: print(">> end state of normal / final interval:\n",np.column_stack(((stateEnd), (stateLC))))
    # normalInt = stateEnd@pwrN@TstateI
    # stormInt = stateEnd@pwrS@TstateI
    # # oneStInt   = 
    # # The final interval is one of these two (ends in final state):
    # # normalFinal = stateLC@pwr@TstateI
    # finalInt = stateLC@pwrN@TstateI
    # # stormFinal  = stateLC@pwrS@TstateI
    # # finalInt[stormTrue] = stormFinal
    # --------------------------------------------------------------------------
    # if cn.debugLL: 
    #     print(">>>> ")
    #     print(">> end state of normal interval:\n", stateEnd)
    #     print(">> end state of final interval:\n",  stateLC)
    #     print(">> likelihood of one normal interval:\n",
    #     normalInt,
    #     normalInt.shape,
    #     normalInt.dtype)
    #     print(">> likelihood of final interval:\n",
    #     finalInt,
    #     finalInt.shape,
    #     finalInt.dtype)
    # NOTE now pwr has storms incorporated
    # print("final interval:", normalFinal, "and -log likelihood:", -np.log(normalFinal))

    # return([normalInt, stormInt, finalInt, stormFinal])
    return([normalInt, finalInt])
# -----------------------------------------------------------------------------
def printLL(numNests, logLik, logLikFin, numInt, logL):
    """ print entire likelihood equation """

    for x in range(numNests):
        # print(">> likelihood equation: (",logLik[x],"*",numIntNorm[x],")+(",logLikStm[x],"*",numIntStm[x],")+(",logLikFin[x],"**(1 -",stormDuringFin[x],")+(", logLikFinStm[x],"**",stormDuringFin[x])
        print(
           f">> likelihood equation for nest {x}: " 
        #    f"{numInt[x]:.0f} * {logLik[x]:.5f} + "
           f"({numInt[x]:.0f} * {logLik[x]:.5f}) + {logLikFin[x]:.5f} ="
        #    f"{logLikFinStm[x]:.5f} * (1-{stormDuringFin[x]:.0f}) + " 
        #    f"{logLikFinStm[x]:.2f} * {stormDuringFin[x]:.2f} = "
        #    f"{logLikelihood[x]:.2f}")
           f"{logL[x]:.2f}"
           )
# -----------------------------------------------------------------------------
# def logL(numNests, normalInt, finalInt, numInt, ha, config=config):
# def logL(numNests, intervals, numInt, ha, fl, config=config):
# @profile
def logL(numNests, intervals, numInt, config=config):
    """
    Purpose
    -------
    Calculate the negative log likelihood for each nest:
    1. Create a counter using Decimal for precision
    2. Create an array of -log likelihood for each nest for:
        - regular interval (where storm==True, this is the storm interval)
        - final interval (where fl==True, this is the storm final interval)
        - for matrices, 1x3 * 3x3 * 3x1 leaves you with a single value for each nest
    3. Sum the -log likelihood values to get an overall value.
    
    - using the stormInt is too complicated for how little it affects things...

    Inputs
    ------
    normalInt, finalInt, & stormFinal are output from interval() - represent likelihood of one interval
    
    ha = True if nest hatched; fl = True if nest flooded
    
    Returns
    -------
    logLike
    
    """
# def logL(normalInt, finalInt, numInt):
    # numNests is not the total number (param value) but the number not excluded
    # 2. Initialize the overall likelihood counter; Decimal gives more precision
    # numNests = len(normalInt)
    # normalInt, stormInt, finalInt, stormFinal = intervals
    normalInt, finalInt = intervals
    # print("numInt excluding final interval:", numInt)
    # NOTE NOTE does multiplying the log likelihoods together give the same result as 
    # multiplying the matrices together?
    # numInt[~ha] = numInt[~ha]+1
    # if config.debugLL: print("numInt after adding final interval:", numInt)
    # logLike = logLikelihood = Decimal(0.0)    # maybe switching types is also slowing things down?     
    logLik  = np.empty(numNests, dtype=np.longdouble) # this should give it enough precision & avoid errors
    logLikFin = np.empty(numNests, dtype=np.longdouble)
    # logLik  = logLik * np.log(normalInt) * -1 # dtype changes to float64 unless you multiply it by itself
    logLik  = -np.log(normalInt) # dtype changes to float64 unless you multiply it by itself
    # logLik[stormInt==True] = logLik * np.log(stormInt) * -1
    # if config.debugLL: 
    # logLikFin    = np.ones(numNests, dtype=np.longdouble)
    # logLikFin.fill(-np.log(normalFinal))
    # logLikFin= -np.log(normalFinal)
    logLikFin= -np.log(finalInt)
    # logLikFin[hatched == True] = 0
    # logLikFin[ha==True] = 0 
    # now stormFinal should be part of normalFinal (finalInt)
    # logLikFin[fl==True] = -np.log(stormFinal[fl==True])
    # logLikFin[]    = logLikFin * (-np.log(normalFinal))
    # logLikFinStm = logLikFinStm * (-np.log(stormFinal))

    # stormDuringFin = nestData[:,12] # was there a storm during the final interval?
    logLikelihood  = (logLik*numInt) + (logLikFin) # elementwise multiplication, then add final interval NLL
    # logLikelihood  = ne.evaluate('(logLik*numInt) + (logLikFin)')

    logLike        = np.sum(logLikelihood)
    # logLike        = ne.evaluate('sum(logLikelihood)')
    # if config.debugLL:
    #     print("number of nests:", numNests, "\n hatched:", ha, "\n flooded:, fl") 
    #     print("numInt excluding final interval:", numInt)
    #     print(">> -log likelihood of 1 interval:", logLik)
    #     print(">> -log likelihood final interval, updated with storms/hatch:\n", logLikFin)
    #     print(">> -log likelihood of each nest history:", logLikelihood)
    #     print(">>>> overall -log likelihood:", logLike)
    #     printLL(numNests=numNests, logLik=logLik, logLikFin=logLikFin, 
    #                     numInt=numInt, logL=logLikelihood)
    return(logLike) # this is what is being optimized
# -----------------------------------------------------------------------------
# try to keep these in numpy:
# def like(perfectInfo, hatchTime, argL, numNests, obsFreq, obsDat, surveyDays):
# def like(argL, numN, obsFr, obsDat, stMat, sTrue, useSM):
# This one uses a matrix multiplication equation created from building blocks
#   > so you create these blocks:
#   > a normal interval, a final interval, and a final interval with storm
# def like(argL, numN, obsFr, obsDat, stMat, useSM, con=config):
# @profile
def like(argL, numN, obsFr, obsDat, useSM, con=config):
    """
    perfectInfo == 0 or 1 to tell you whether you know all nest fates or not
    ---------------------------------------------------------------------------------------------------
    1. Unpack:
       a. Initial values for optimizer:
    a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
       b. Observation history values from nest data:
    fl, ha, ff, la, lc, nInt, sTrue = obsDat
    fl, ha, ff, la, lc, nInt = obsDat
    
    """
    # numpy array should be unpacked along the first dimension, so transpose:
    # if con.debugLL: print("-------------------------------------------------------------------------------------")
    # if con.debugLL: print("------------------------- the MCMC ML function --------------------------------------")
    # if con.debugLL: print("-------------------------------------------------------------------------------------")
    ff, la, lc, fate, nInt, sFinal = obsDat.T
    fl = fate==2 # are these necessary for more than print statements?
    # ha = fate==0
    pr = fate==1
    # NOTE NOTE should this be the assigned fate or true fate?
    # NOTE these if .. print statements, esp inside the optimizer, take lots of time:
    # if con.debugLL:
    #     print("number of nests and observation interval:", numN, obsFr)
    #     print("nest observation data - ff, la, lc, fate, nInt, sFinal\n", obsDat)
    #     print("did nest hatch?\n", ha, "\ndid nest flood?\n", fl)
    #     print(
    #         ">>>> run likelihood function - ",
    #         "num flooded:", sum(fl), "& hatched:", sum(ha),
    #         "number of obs intervals:", nInt,
    #         # "\n>> ijk:\n", [ff, la, lc]
    #         "\n>> ijk:\n", np.column_stack((ff, la, lc))
    #         )
    # stEnd, stFin = stMat 
    # pwrOut = nest_mat(argL=argL, obsFreq=obsFr, stormFin=sFinal, useStormMat=useSM)
    pwrOut = nest_mat(argL=argL, obsFreq=obsFr, stormFin=sFinal, useStormMat=useSM)
    # pwr, pwrStm = pwrOut
    # inter  = interval(pwr=pwr, stateEnd=stEnd, stateLC=stFin)   
    # inter = interval(pwr=pwrOut, stateMat=stMat)   
    # inter = interval(pwr=pwrOut,ha=ha, fl=fl, numNests=numN)   
    inter = interval(pwr=pwrOut, fl=fl, pr=pr, numNests=numN)   
    # norm, fin, sfin = inter
    # llVal = logL(normalInt=norm, normalFinal=fin, stormFinal=sfin, numInt=nInt)
    # llVal = logL(numNests=numN, intervals=inter, numInt=nInt, ha=ha, fl=fl)
    llVal = logL(numNests=numN, intervals=inter, numInt=nInt)
    # make sure numN is the number of analyzed nests, not the param value (total number)
    
    return(llVal)
# this function does the matrix multiplication for a SINGLE interval of length intElt days 
 # during observaton, nest state is assessed on each visit to form an observation history 
 # the function calculates the negative log likelihood of one interval from the observation history 
 # these can then be multiplied together to get the overall likelihood of the observation history 
# -----------------------------------------------------------------------------
#   THE LIKELIHOOD WRAPPER FUNCTION
# -----------------------------------------------------------------------------
# @profile
def like_smd( 
        # x, perfectInfo, hatchTime, nestData, obsFreq, 
        # stormDays, surveyDays, whichRet):
        # x, obsData, obsFreq, stateMat, useSM, stormDays, surveyDays, whichRet=1, config=config):
        # x, obsData, obsFreq, stateMat, useSM, stormDays, surveyDays, whichRet, config=config):
        x, obsData, obsFreq, useSM, stormDays, surveyDays, whichRet, config=config):
        # x, obsData, obsFreq, stateMat, useSM, whichRet=1, **kwargs):
    # use kwargs for stormDays & surveysDays, which are only needed w/ like_old
    """
    Arguments:
        x: output from randArgs(); 5 values, only 4 used by LL function
        
        obsData: just the columns from observer() output: 
        i, j, k, assigned fate, num obs, finalIntStorm
        
        whichRet: which like() function should be used? 1=new (default); 2=old

    The values are log-transformed before running them thru the likelihood 
    function, so the values given to the optimizer are the untransformed values,
    meaning the optimizer output will also be untransformed.
        >> Therefore, need to transform the output as well.
    """


    # unpack the initial values:
    s0   = x[0]
    mp0  = x[1]
    # ss0  = x[2]
    # mps0 = x[3]
    # sM   = x[4]
    # if config.debugLL: print("initial values:", s0, mp0, ss0, mps0, sM)

    # transform the initial values so all are between 0 and 1:
    s1   = logistic(s0)
    mp1  = logistic(mp0)
    # ss1  = logistic(ss0)
    # mps1 = logistic(mps0)
    #@#print("logistic-transformed initial values:", s1, mp1, ss1, mps1)

    # further transform so they remain in lower left triangle:
    tri1 = triangle(s1, mp1)
    # tri2 = triangle(ss1, mps1)
    s2   = tri1[0]
    mp2  = tri1[1]
    # ss2  = tri2[0]
    # mps2 = tri2[1]
    # if config.debugLL: print("log- & triangle-transformed initial values:", s2, mp2, ss2, mps2)

    # compute the conditional probability of mortality due to flooding:
    mf2  = 1.0 - s2 - mp2
    # mfs2 = 1.0 - ss2 - mps2

    numNests = obsData.shape[0]
    #@#print(">> number of nests:", numNests)

    # call the likelihood function:
    # argL = np.array([s2,mp2,mf2,ss2,mps2,mfs2, sM])
    argL = np.array([s2,mp2,mf2])
    #ret = like(argL, ndata, obs, storm, survey)
    #ret = like(argL, nestData, obsFreq, stormDays, surveyDays)
    # def like(argL, numN, obsFr, obsDat, stMat, useSM):
    # obsDat is a subset of nestData
    
    # if whichRet == 1:
    ret = like(argL, numN=numNests, obsFr=obsFreq, obsDat=obsData, 
    # ret = like(argL, numN=numNests, obsFr=obsFreq, obsDat=obsData, stMat=stateMat,
               useSM=useSM)
        # if config.debugLL: print('like_smd(): Msg : ret = ', ret)
    
    # else:
    # elif whichRet == 2:
    #     ret = like_old(argL, obsFreq, obsData, surveyDays[0], stormDays, numNests)
    #     print('like_smd(): Msg : using old function; ret = ', ret)
    
    # else:
    #     print(' argument whichRet is invalid ')
    
    # rets = np.array([ret, ret2])

    return(ret)

# ---------------------------------------------------------------------------------------
# def ansTransform(ans, unpack):
# if you add unpack param, can use for transforming input as well as output
# def ansTransform(ans, unpack=True):
def ansTransform(ans):
    """
    Transform the optimizer output so that it is between 0 and 1, and the 3 
    probabilities sum to 1. 

    'ans' is an object of type 'OptimizeResult', which has a number of components
    """
    # if unpack:
        # ans = ans.x    
    # s0   = ans.x[0]         # Series of transformations of optimizer output.
    s0   = ans[0]         # Series of transformations of optimizer output.
    mp0  = ans[1]         # These make sure the output is between 0 and 1, 
    # ss0  = ans[2]         # and that the three fate probabilities sum to 1.
    # mps0 = ans[3]

    s1   = logistic(s0)
    mp1  = logistic(mp0)
    # ss1  = logistic(ss0)
    # mps1 = logistic(mps0)

    ret2 = triangle(s1, mp1)
    s2   = ret2[0]
    mp2  = ret2[1]
    mf2  = 1.0 - s2 - mp2

    # ret3 = triangle(ss1, mps1)
    # ss2  = ret3[0]
    # mps2 = ret3[1]
    # mfs2 = 1.0 - ss2 - mps2
    
    #ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.float128)
    # ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.longdouble)
    ansTransformed = np.array([s2, mp2, mf2], dtype=np.longdouble)
    # print(">> results as an array:\n", ansTransformed)
    # if debug: print(">> results (s, mp, mf, ss, mps, mfs, ex):\n", s2, mp2, mf2, ss2, mps2, mfs2, ex)
    return(ansTransformed)

# -----------------------------------------------------------------------------
def randArgs():
    """
    Choose random initial values for the optimizer.
    These will be log-transformed before going through the likelihood function
    
    Returns:
    array of s, mp, ss, mps (for like_smd) and srand (for mark_wrapper)
    """
    s     = rng.uniform(-10.0, 10.0)       
    mp    = rng.uniform(-10.0, 10.0)
    # ss    = rng.uniform(-10.0, 10.0)
    # mps   = rng.uniform(-10.0, 10.0)
    srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
    # z = np.array([s, mp, ss, mps, srand])
    z = np.array([s, mp])

    return(z)
# -----------------------------------------------------------------------------
#   CREATE NEST DATA AND RUN THE OPTIMIZER 
# -----------------------------------------------------------------------------
# need to loop through the param combinations
# within the loop, need to unpack the params and run the optimizer
# PARAM NAMES:
if False:
    print("hi")
    # numNests: int
    # stormDur: int
    # stormFrq: int
    # obsFreq:  int
    # hatchTime:int
    # brDays:   int
    # probSurv: np.float32
    # SprobSurv:np.float32
    # pMortFl  :np.float32
    # discProb: np.float32
    # fateCues: np.float32
    # stormFate:bool
    # useStmMat:bool

#def run_params(paramsList, dirName):
# NOTE need to create storms after params are chosen
# NOTE is there even a reason to have this in a function?
# NOTE NOTE do I have to pass every object to each function explicitly?
# @profile
def run_optim(fun, z, arg, met='Nelder-Mead'):
    """
    Run scipy.optimize.minimize on 'fun'. Will return value of -1 or -2 if exceptions occur.

    If all is well, transform the output (using ansTransform() for MCMC model, and
    logistic() for MARK model)

    Returns:
    the transformed output.
    """
    try:
        out = optimize.minimize(fun, z, args=arg, method=met)
        ex = 0.0
    except decimal.InvalidOperation as error2:
        # ex=100.0
        ex=-1.0
        print(">> Error: invalid operation in decimal:", error2, "Go to next replicate.")
        return(ex)
        # continue
    except OverflowError as error3:
        # ex=200.0
        ex=-2.0
        print(
            ">> Error: overflow error:", 
            error3, 
            "Go to next replicate."
            )
        return(ex)
        # continue
    #
    if fun==like_smd: 
        # print("Success?", out.success, out.message, "answer=", out.x)
        res = ansTransform(ans=out.x)
    else:
        # res=ansTransform(ans, unpack=False)
        # res=ansTransform(ans=out)
        res = logistic(out.x[0])
    return(res)
    
# def make_obs(par, init, dfs, stormDays, surveyDays, config=config):
# @profile
    
# def rep_loop(par, repID, nData, vals, storm, survey, config):
# def rep_loop(par, nData, vals, storm, survey, config):
# @profile
def rep_loop(par, nData, storm, survey, config):
    """
    For each data replicate, call this function, which:
        - takes the reduced nest data as input
        - calls the optimizer on like_smd() and mark_wrapper()
        
    Returns:
        like_val
        [0]
    """
    # ---- empty array to store data for this replicate: ---------
    # like_val  = np.zeros(shape=(config.numOut), dtype=np.longdouble)
    # like_val  = np.zeros(shape=(3), dtype=np.longdouble)
    # perfectInfo = 0
    # whichL = par.whichLike
    # ex = np.longdouble("0.0")
    # stMat = state_vect(nNest=len(nData), fl=(nData[:,3]==2), ha=(nData[:,3]==0))
    dat = nData[:,4:10] # doesn't include column index 10
    # ans      = run_optim(fun=like_smd, z=randArgs(), 
    res      = run_optim(fun=like_smd, z=randArgs(), 
                        #  arg=(dat, par.obsFreq, stMat, par.useSMat, storm, survey, 2))
                        #  arg=(dat, par.obsFreq, stMat, par.useSMat, storm, survey, 1))
                        #  arg=(dat, par.obsFreq, stMat, par.useSMat, storm, survey, par.whichLike))
                         arg=(dat, par.obsFreq, par.useSMat, storm, survey, par.whichLike))
                                # args=( dat, obsFr, stMat, useSM, 1),
    # res = ansTransform(ans)
    srand = rng.uniform(-10.00, 10.00)
    # markProb = mark_probs(s=srand, ndata=nData)
    # mark_s = run_optim(fun=mark_wrapper, z=srand, arg=(nData, markProb, par.brDays))
    mark_s = run_optim(fun=mark_wrapper, z=srand, arg=(nData, par.brDays))
    #NOTE ans2 is an "OptimizeResult" object; need to extract "x"
    # Transform the MARK optimizer output so that it is between 0 and 1:
    # mark_s = logistic(ans2.x[0]) # answer.x is a list itself - need to index
    # print("> logistic of MARK answer:", mark_s)
    # s2,mp2,mf2,ss2,mps2,mfs2 = res # unpack like() function output
    # NOTE scott was probably right - mps doesn't make sense. and DSR includes storms already
    # so check whether mort flood probability goes up with more intense storms?
    s2, mp2 = res[0], res[1]
    # mp2 = res[1]

    # s2,mp2,mf2,ss2,mps2,mfs2 = res2 # unpack like_old() function output
    # like_val = [ mark_s,s2,mp2,mf2,ss2,mps2,mfs2]
    like_val = np.array([ mark_s,s2,mp2], dtype=np.longdouble)
    # if config.debug: print(">> like_val:\n", like_val)
    # if config.debugLL: print(">> like_val:\n", like_val)
    # like_val = np.array(like_val, dtype=np.longdouble)
    return(like_val)
    
# def save_vals(parID, repID, like_val, ):
# def main(testing=False, fname=mk_fnames(), pStatic=staticPar):
# def main(testing=False, config=config, pStatic=staticPar):
# def set_debug(deb, debN, debS, config=config):
# def set_debug(deb=debugTypes):
def set_debug(deb):
    """
    called from within main() to set the debug config

    arguments to main():

        debN = nests & observer (options: "all", "nests", "obs", "none"[default])

        deb = MLE & program mark (options: "all", "like", "mark", "none"[default])

        debS = flooding (options: "all", "none"[default])
    """
    if deb=="all": 
        config.debugFlood = config.debugLL = config.debugM = config.debugNests = config.debugObs = True
    elif "," in deb:
        deb = deb.split(",")
        # print(deb)
    else:
        deb = [deb]
        # print(deb)
        # for i, val in enumerate(deb):
    for val in deb:
        # print(val)
        if val == "like": config.debugLL = True
        # elif val == ("nest" or "nests"): config.debugNests = True
        elif val == "nest": config.debugNests = True
        elif val == "flood": config.debugFlood = True
        elif val == "mark": 
            config.debugM = True
            # print("mark=True")
        elif val == "obs": config.debugObs = True
        # else: print("WARNING-invalid debug value. options: 'like', 'nest|nests', 'mark|MARK', 'flood', 'obs|observer'. place in single string with , delim ")
        else: print("WARNING-invalid debug value. options: 'like','nest','mark','flood','obs'. place in single string with , delim ")
            
        # print("debug options in config:", config.debugM)
        # if "like" in deb:
        #    config.debugLL =True
        # if "nest" or "nests" in deb:
        #     config.debugNests = True
        # if "flood" in deb:
        #     config.debugFlood = True
        # if "mark" in deb:
        #     config.debugM = True
        
    # if debN == "all":
    #     config.debugNests = config.debugObs = True
    # elif debN == "nest":
    #     config.debugNests = True
    # elif debN == "obs":
    #     config.debugObs = True

    # if deb == "all":
    #     config.debugLL=config.debugM=True
    # elif deb == "mark":
    #     config.debugM = True
    # elif deb == "like":
    #     config.debugLL = True
    
    # if debS =="all":
    #     config.debugFlood = True
    
def print_nest_info(nestData, discover, exclude):
    if debug: print("nests not discovered:", nestData[:,0][~discover])
    if debug: print("nests to exclude from analysis:", nestData[:,0][exclude])
    if debug: print("nest data, analysis nests only:\n",
                    "ID, init, survival, true fate, i, j, k, assigned fate, num normal obs, intFinal, num storms:\n",
                     nestData[~discover and exclude])
# def main(fnUnique, useWSL=False, testing=True, deb="none", debN="none", debS="none", config=config, pStatic=staticPar):
# def main(fnUnique, testing=config.testing, deb="none", debN="none", debS="none", config=config, pStatic=staticPar):
# def main(fnUnique, debugOpt, testing=config.testing, config=config, pStatic=staticPar): # calls config too early
# @profile
def main(fnUnique, debugOpt, testing, config=config, pStatic=staticPar):
    """
    Debug options:
        'deb' is for the likelihood/program MARK. options: 'all', 'mark', 'like'
        'debN' is for the nest-making & observer. options: 'all', 'nest', 'obs'
        'debS' = storms/flooding (options: "all", "none"[default])
    If 'fnUnique'==True, filename is "uniquified" and includes H:M:S
        Otherwise, just the date.
    """
    lf_suffix=""
    # these if-else statements only run once:
    if testing == "norm":
        config.nreps=400
        # print("changed config values:",config.debug, config.nreps)
        pList = plTest
        # global debug 
        debug = True
        lf_suffix = "-test"
        print("using test values. global debug = ", debug)
    elif testing=="storm":
        config.nreps=10
        config.debugFlood=True
        config.debugObs=True
        pList = plTestFlood
        # global debug
        debug = True
        lf_suffix = "-flood"
        print("using storm test values. global debug = ", debug)
    elif testing=="debug":
        config.nreps=100
        debug=True
        pList=plDebug
        lf_suffix="-debug"
    else:
        pList = parLists # don't need to update any settings if not testing?
    # set_debug(deb=deb, debN=debN, debS=debS)
    if debugOpt != None:
    # if deb:
    # if hasattr(args,"")
        # set_debug(debugTypes)
        set_debug(debugOpt)
    # fname = mk_fnames(like_f_dir=like_f_dir) if fnUnique else mk_fnames(unique=False)
    fname = mk_fnames(suf = lf_suffix) if fnUnique else mk_fnames(suf =lf_suffix, unique=False)
    fdir  = fname[0].parent
    # fdir  = os.path.split(fname[0])[0]
    # print(fdir)
    config.likeFile = fname[0]
    config.colNames = fname[1]
    print(config)
    with open(config.likeFile, "wb") as f: # doesn't need to be 'a' bc file is open
        paramsArray = mk_param_list(parList=pList)
        # if debug: print(f">>>> there will be {len(paramsArray)*config.nreps} total rows")
        print(f">>>> there will be {len(paramsArray)} param sets & {len(paramsArray)*config.nreps} total rows")
        parID       = 0
        for i in range(0, len(paramsArray)): # for each set of params
            # if debug: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            par        = paramsArray[i] 
            par_merge  = {**par, **pStatic}
            par        = Params(**par_merge)
            print(">>>>>>>>> param set number:", parID, "and params in set:", par)
            stormDays  = stormGen(frq=par.stormFrq, dur=par.stormDur)
            survey     = mk_surveys(stormDays, par.obsFreq, par.brDays)
            # surveyDays, surveyInts = survey
            repID = numMC = nEx = 0 # number of nests misclassified, number of exceptions
            # likeVal    = np.zeros(shape=(config.nreps,config.numOut))
            for r in range(config.nreps): 
                # if debug: print("\n>>>>>>>>>>>> replicate ID: >>>>>>>>>>>>>>>>>>>>>>>>>>", repID)
                try:
                    nestData1 = make_obs(par=par, storm=stormDays, survey=survey) 
                # except:
                except IndexError as error:
                    print(
                        ">> IndexError in nest data:", 
                        error,
                        ". Go to next replicate")
                    nEx = nEx + 1
                    # nestData = np.zeros((par.numNests, 11))
                    # return(nestData)
                    continue

                trueDSR  = calc_dsr(nData=nestData1, nestType="all")
                flooded  = sum(nestData1[:,3]==2)
                hatched  = sum(nestData1[:,3]==0)
                # print_prop(nestData[:,7], nestData[:,3], )
                discover = nestData1[:,6]!=0
                nestData = nestData1[(discover),:] # remove undiscovered nests
                exclude  = ((nestData[:,7] == 7) | (nestData[:,4]==nestData[:,5]))                         
                unknown  = (nestData[:,7]==7)
                misclass = (nestData[:,7]!=nestData[:,3])
                nestData    = nestData[~(exclude),:]    # remove excluded nests 
                trueDSR_an   = calc_dsr(nData=nestData, nestType="analysis") 
                lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
                                   survey=survey,config=config)
                # pars = np.array([par.probSurv, par.stormDur, par.stormFrq, par.numNests, 
                #                  par.hatchTime,par.obsFreq]) 
                # nVal = np.array([trueDSR, trueDSR_an, sum(discover), sum(exclude), repID])  
                nVal = np.array([trueDSR, trueDSR_an, sum(discover), sum(exclude), sum(unknown), sum(misclass), flooded, hatched, nEx, repID, parID])  
                like_val = np.concatenate((lVal, nVal))
                colnames=config.colNames
                # if (trueDSR_an - lVal[1]) / trueDSR_an > 40:
                # print("bias:",(trueDSR-lVal[1])/trueDSR)
                if (trueDSR - lVal[1]) / trueDSR > 0.40:

                    print("high bias")
                #     np.save(f"{fdir}/nestdata_{parID:02}_{repID:02}_bias.npy", nestData1)
                # else:
                #     print("low bias")
                #     np.save(f"{fdir}/nestdata_{parID:02}_{repID:02}.npy", nestData1)

                # if parID == 0 and like_val[17] == 0: # only the first line gets the header
                if parID == 0 and like_val[12] == 0: # only the first line gets the header
                    np.savetxt(f, [like_val], delimiter=",", header=colnames)
                    # if debug: print(">> ** saving likelihood values with header **")
                else:
                    np.savetxt(f, [like_val], delimiter=",")
                    # if debug: print(">> ** saving likelihood values **")
                # need to save it in the function where f was opened?
                # likeVal[r] = like_val
                repID = repID + 1
            # like_val = param_loop(par=par, parID=parID, storm=stormDays, 
            #                       survey=survey, config=config)
            # colnames=config.colNames
            # if parID == 0 and repID == 0: # only the first line gets the header
            # if parID == 0 and like_val[0,17] == 0: # only the first line gets the header
            # # if firstLine == True: # only the first line gets the header
            # # if parID == 0 : # only the first line gets the header
            #     np.savetxt(f, [like_val], delimiter=",", header=colnames)
            #     #np.savetxt(f, like_val, delimiter=",", header=colnames)
            #     if debug: print(">> ** saving likelihood values with header **")
            # else:
            #     np.savetxt(f, [like_val], delimiter=",")
            #     if debug: print(">> ** saving likelihood values **")
                  
                  #    nreps=settings.nreps)
                
            parID = parID + 1

    
# rng = config.rng
# debug_nest=config.debugNests
# debugM=config.debugLL
# debugLL=config.debugLL
# args = config.args
# args = sys.argv[1:]
# args = sys.argv()
# options = "htdwo:"
# optVal  =

# except getopt.error as err:
#     print(str(err))
# test1 = False if 

# using argparse gives more functionality, but optparse gives more control:
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", 
#                     help="run in testing mode w/ limited param values & limited num nests? (default:False)", 
#                     type=bool, action='store_true')
# parser.add_argument("-w", "--wsl", 
#                     help="using WSL? - will change filenames to match (default:False)", 
#                     type=bool, action='store_true')
# parser.add_argument("-d", "--debug", 
#                     help="turn on/off the basic debugger - for more print statements, see script (default:False)", 
#                     type=bool, action='store_true')
# parser.add_argument("-f", "--file", 
#                     help="change likelihood output file location",
#                     type=str)
# args=parser.parse_args()

# config.useWSL = args.wsl


# debug = config.debug
# NOTE need to decide if it's worth saving line-by-line
# main(fnUnique=False, testing=True,deb='none', debN="all", debS="none")
# main(fnUnique=False, useWSL=False, testing=True, deb='none', debN="none", debS="none")
# main(fnUnique=False, debugOpt=debugTypes)
main(fnUnique=fnUnique, debugOpt=debugTypes, testing=config.testing)
# main(fnUnique=False, testing=False, deb='none', debN="all")
# if len(args) > 1:
#     # debug = args[1] == "debugTrue"
#     # enumerate() give sboth index and value
#     for i in range(len(args)):
#         debugList[args[i]] = True
# for key, val in debugList.items():
#     globals()[key] = val
#     print(globals()[key], val)
    
# llArgs = randArgs()
# testLL  = like_smd(x=llArgs, obsData=dat, obsFreq=par.obsFreq, 
#                    stateMat=stMat, useSM=par.useSMat, whichRet=1)



