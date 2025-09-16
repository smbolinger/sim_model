# sudo vim -o file1 file2 [open 2 files] 
# BLAH
# NOTE 5/16/25 - The percent bias responds more like I would expect when I use
#                the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
#                BUT I still don't know why the calculated DSR is consistently low.

from dataclasses import dataclass
from datetime import datetime
import decimal
from decimal import Decimal
# from itertools import product
import itertools
import numpy as np 
from os.path import exists
import os
from pathlib import Path
from scipy import optimize
import scipy.stats as stats
import sys
from typing import Dict, Generator

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
    probSurv: np.float32
    SprobSurv:np.float32
    pMortFl  :np.float32
    discProb: np.float32
    fateCues: np.float32
    stormFate:bool
    useSMat:  bool

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Config: 
    rng:         Generator
    args:        list[str]
    nreps:       int
    debug:       bool
    debugLL:     bool
    debugNests:  bool
    likeFile:    str
    colNames:    str
    numOut:      int

# for i, arg in enumerate(sys.argv):
    # print(f"Argument {i}: {arg}")
 # use different debug var bc these will print for every time optimizer runs
debugList = dict( debug_nest = False,
                  debug_obs = False, 
                  debugLL = False, 
                  debug = False,
                  debugM = False,
                  debugL = False )
if len(args) > 1:
    # debug = args[1] == "debugTrue"
    # enumerate() give sboth index and value
    for i in range(len(args)):
        debugList[args[i]] = True
for key, val in debugList.items():
    globals()[key] = val
    print(globals()[key], val)
# v = debugList.values() #test
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
def searchSorted2(a, b):
    """Get the index of where b would be located in a"""
    #out = np.zeros(a.shape)
    out = np.zeros((a.shape[0], len(b)))
    for i in range(len(a)):
        #out[i] = np.searchsorted(a[i], b[i])
        #print("sorted search of\n", b, "within\n", a[i])
        # if debug: print(">> sorted search of", b, "within", a[i])
        out[i] = np.searchsorted(a[i], b)
        # if debug: print("sorted search of\n", b, "within\n", a[i], ":\n", out, out.shape)
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
    """
    idx = np.searchsorted(B, A)
    idx[idx==len(B)] = 0
    return A[B[idx] == A]
# -----------------------------------------------------------------------------
#   NEST MODEL PARAMETERS:
#region-----------------------------------------------------------------------
# NOTE the main problem is that my fate-masking variable (storm activity) also 
#      leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
# staticPar = {'nruns': 1,
# These are the values that are passed to the Params class
staticPar = {'brDays': 160,
             'SprobSurv': 0.2,
             'probMortFlood': 0.1,
             'discProb': 0.7,
             'stormFate': True,
             'useStormMat': False
             }

parLists = {'numNests' : [500,1000],
            'probSurv' : [0.95, 0.96],
            'stormDur' : [1, 2, 3],
            'stormFrq' : [1, 3, 5],
            'obsFreq'  : [3, 5, 7],
            'hatchTime': [16, 20, 28] }

plTest  = {'numNests'  : [50],
           'probSurv'  : [0.95],
           'stormDur'  : [1, 2],
           'stormFrq'  : [1, 3],
           'obsFreq'   : [3, 5],
           'hatchTime' : [20, 28] }

def mk_param_list(parList: Dict[str, list]) -> list:
    """
    Take the dictionary of lists of param values, then unpack the lists to a 
    list of lists. Then feed this list of lists to itertools.product using *.
    
    Returns: a list of dicts representing all possible param combos, with keys!
    """
    # product takes any number of iterables as input
    # input in the original is a bunch of lists
    # output in the original is a list of tuples
    # listVal = parList.values() # doesn't seem to be what i want
    # p_List = list(product(parList.values()))
    if debug: print(f"using the {parList} params lists")
    listVal = [parList[key] for key in parList]
    p_List = list(itertools.product(*listVal))
    # p_List = list(product(parList)) # doesn't work either; just gets keys
    # make this list of lists into a list of dicts with the original keys 
    paramsList = [dict(zip(parList.keys(), p_List[x])) for x in range(len(p_List))]
    return(paramsList)
# pl = mk_param_list(parList=plTest) # test
#endregion---------------------------------------------------------------------
#   SAVE FILES 
#region-----------------------------------------------------------------------
# name for unique directory to hold all output:
def mk_fnames():
    """
    1. Create a directory w/ a unique name using datetime.today() & uniquify().
    2. Create likelihood filepath (& parent dir, if necessary)
    3. Make a string out of the column names that can be used w/ np.savetxt()
    
    Returns:
    tuple of likelihood filepath & colnames string
    """
    dirName    = datetime.today().strftime('%m%d%Y_%H%M%S'),
    likeF = Path(uniquify(Path.home() / 
                               'C://Users/Sarah/Dropbox/nest_models/py_output' / 
                               dirName / 
                               'ml_values.csv'))
    likeF.parent.mkdir(parents=True, exist_ok=True)
    column_names = np.array([
        'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 
        'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 
        'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 
        'obs_int', 'num_discovered','num_excluded', 'exception'
        ])
    colnames = ', '.join([str(x) for x in column_names]) # needs to be string
    # saveNames = dict(
    #     likeFile   = likeF,
    #     dirName    = datetime.today().strftime('%m%d%Y_%H%M%S'),
    #     todaysDate = datetime.today().strftime("%Y%m%d"),
    #     colnames = ', '.join([str(x) for x in column_names]) # needs to be string
    # )
    print(">> save directory name:", dirName)
    print(">> likelihood file path:", likeF)
    # return(saveNames)
    return(likeF, colnames)
#endregion--------------------------------------------------------------------
#   FUNCTIONS
# -----------------------------------------------------------------------------
# Some are very small and specific (e.g. logistic function); others are 
# quite involved.
# -----------------------------------------------------------------------------
def init_from_csv(
        file="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"):
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
    print(">> week: init probability = ",ret)
    return(ret)
    # return(initProb)
# -----------------------------------------------------------------------------
def sprob_from_csv(file="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"):
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
    print(">> week: storm probability = ",ret)
    return(ret)
# -----------------------------------------------------------------------------
def stormGen(frq, dur, wStart, pStorm):
    """
    generate a list of days where storms happened.

    the probabilities and week start dates used are read from csv outside the 
    function to streamline it.
    """
    # out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    out = rng.choice(a=wStart, size=frq, replace=False, p=pStorm)
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
    surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
    # surveyInts  = np.append(surveyInts, )
    if debug: 
        print(">> all survey days, minus storms:\n", surveyDays, len(surveyDays)) 

    return(surveyDays, surveyInts)
# -----------------------------------------------------------------------------
def survey_int(surveyDays):
    """ create a list of the lengths of all intervals between survey days. """
    surveyInts = np.array( [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
    #print(">> interval between current survey and previous survey:\n", surveyInts, len(surveyInts))

    return(surveyInts)
# -----------------------------------------------------------------------------
# ------ NEST DATA HELPER FUNCTIONS -------------------------------------------
# -----------------------------------------------------------------------------
def mk_init(weekStart, initProb, numNests):
    initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
    initiation = initWeek + rng.integers(7)                    # add a random number from 1 to 6 (?) 
    if debug: print(">> initiation week start days:\n", initWeek) 
    return(initiation)
# -----------------------------------------------------------------------------
def mk_surv(numNests, hatchTime, pSurv):
    # 4. Decide how long each nest is active
    # >> use a negative binomial distribution - distribution of number of 
    #    failures until success 
    #     >> in this case, "success" is actually the nest failing, so use 
    #        1-pSurv (the failure probability) 
    #     >> gives you number of days until the nest fails (survival)
    #     >> if survival > incubation time, then the nest hatches 
    # >> then use survival to calculate end dates for each nest 
    #    (end = initiation + survival)

    survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests) 
    survival = survival - 1 # but since the last trial is when nest fails, need to subtract 1
    # if debug: print(">> survival in days:\n", survival, len(survival)) 
    
    ## >> set values > incubation time to = incubation time (nest hatched): 
    ##      (need to because you are summing the survival time)
    #        >> once nest reaches incubation time (+/- some error) it hatches
    #           and becomes inactive

    survival[survival > hatchTime] = hatchTime # add some amt of error?
    if debug: print(">> survival in days:\n", survival, len(survival)) 
    hatched = survival >= hatchTime # the hatched nests survived for >= hatchTime days 
    if debug: print("hatched (no storms):", hatched, hatched.sum())
    ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
    # if debug: print("real hatch proportion:", hatched.sum()/numNests)
    return(survival)
# -----------------------------------------------------------------------------
def mk_per(start, end):

    # nestPeriod = np.stack((nestData[:,1], (nestData[:,1]+nestData[:,2]))) # create array of tuples
    # need the double parentheses so it knows output is tuples
    nestPeriod = np.stack((start, end)) # create array of tuples
    nestPeriod = np.transpose(nestPeriod) # an array of start,end pairs 
    if debug: print(
        ">> start and end of nesting period:\n", 
        nestPeriod, 
        nestPeriod.shape
        )
    return(nestPeriod)
# -----------------------------------------------------------------------------
def storm_nest(nestPeriod, stormDays):
    # >> stormNestIndex searches for storm days w/in active period of each nest
    #     >> returns index where storm day would be within the active interval: 
    #             0 = before init; 2 = after end; 1 = within interval
    #     >> fate cues should become harder to interpret after storms
    stormNestIndex = searchSorted2(nestPeriod, stormDays)
    if debug: print("where were storms in active period?", stormNestIndex)
    # if index == 1, then storm Day is within the period interval: 
    stormNest = np.any(stormNestIndex == 1, axis=1) 
    if debug: print(
        "which nests were active during a storm? How many?", 
        stormNest, 
        stormNest.sum()
        )
    numStorms = np.sum(stormNestIndex==1, axis=1) # axis=1 means summing over rows?
    # NOTE I *think* this is actually number of storm intervals, which is what 
    # we want for the likelihood function. so that would be good...
    if debug: print(">> number of storm intervals during nesting period:\n", numStorms)
    return(numStorms)
# -----------------------------------------------------------------------------
def mk_nests(params, init, weekStart, nestData): 

    # 1. Unpack necessary parameters
    # NOTE about the params at the beginning of the script:
    # some have only 1 member, but they are still treated as arrays, not scalars

    hatchTime = int(params[5]) 
    # obsFreq   = int(params[6]) 
    numNests  = int(params[0]) 
    pSurv     = params[1]       # daily survival probability
    # fateCuesPresent = 0.6 if obsFreq > 5 else 0.66 if obsFreq == 5 else 0.75
    # if debug: print(
    #     ">> observation frequency:", obsFreq, 
    #     ">> prob of correct fate:", fateCuesPresent,
    #     )

    # 2. Assign values to the dataframe
    nestData[:,0] = np.arange(1,numNests+1) # column 1 = nest ID numbers 
    nestData[:,1] = mk_init(weekStart, init, numNests)                              # record to a column of the data array
    # nestData[:,1] = mk_init(weekStart, initProb, numNests)                              # record to a column of the data array
    # if debug: print(">> end dates:\n", nestEnd, len(nestEnd)) 
    nestData[:,2] = mk_surv(numNests, hatchTime, pSurv)
    ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
    # nestData[:,3] = nestData[:,2] > hatchTime
    # nestData[:,3] = nestData[:,1] + nestData[:,2]
    # don't need to add end date to dataframe
    # NOTE Remember that int() only works for single values 
    # if debug: print(nestData[1:6,:])
    return(nestData)
# ---- FAILED NESTS --------------------------------------------------------

def mk_flood(params, numStorms,numNests):
    """
    Decide which nests fail from flooding:
        1. Create a vector of random probabilities drawn from a uniform dist
        2. Compare the random probs to pfMort
        3. If flooded=1 and it was during a storm, then nest flooded
        
    Arguments:
        params - really only using one of these (pfMort)
        numStorms - vector telling how many storm periods intersected with 
                    active period, for each nest
        numNests  

    Returns: T/F did nest flood?
    """
    pfMort = params[2]       # prob of surviving (not flooding) during storm
    print("prob of failure due to flooding:", pfMort)
    pflood = rng.uniform(low=0, high=1, size=numNests) 
    # need to check whether this is the correct distribution 
    # NOTE: still needs to be conditional on nest having failed already...  
    # NOTE np.concatenate joins existing axes, while np.stack creates new ones
    
    flooded = np.where(pflood>pfMort, 1, 0) # if pflood>pfMort, flooded=1, else flooded=0 
    if debug: print("flooded:", flooded)
    # and/or/not don't work bc it's a vector; since it's 1 and 0, can use arithmetic: 
    stormNest = numStorms >= 1
    if debug: print("storm nests:", stormNest)
    floodFail = stormNest + flooded > 1 # both need to be true 
    if debug: print("flooded and during storm:", floodFail, floodFail.sum())
    # nestData[:,4] = floodFail.astype(int) 
    return(floodFail)
# -----------------------------------------------------------------------------
def mk_fates(numNests, hatched, flooded):
    """
    Returns: list of true fates for each nest
    """
    trueFate = np.empty(numNests) 
    if debug: print(">> hatched:", hatched)
    #print("length where flooded = True", len(trueFate[floodedAll==True]))
    trueFate.fill(1) # nests that didn't flood or hatch were depredated 
    #trueFate[flooded==True] = 2 
    #trueFate[floodedAll==True] = 2 
    trueFate[hatched == True] = 0 # was nest discovered?  
    trueFate[flooded == True] = 2 
    # trueFate[floodedAll ] = 2 
    # hatched fate is assigned last, so hatched is taking precedence over flooded
    # so maybe that's why the true DSR is always higher than 0.93, but what about true DSR of discovered?
    # fates = [np.sum(trueFate==x) for x in range(3)]
    if debug: print(">>>>> true final nest fates:\n", trueFate, sum(trueFate))
    return(trueFate)

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
def assign_fate(fateCuesPresent, trueFate, numNests, stormIntFinal, stormFate):
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
    fateCuesPres[stormIntFinal==True] = 0.1
    if debug: 
        print(">> compare random probs to fateCuesPresent:\n", 
              [fateCuesProb,fateCuesPres], 
              fateCuesProb.shape)
        
    assignedFate[fateCuesProb < fateCuesPres] = trueFate[fateCuesProb < fateCuesPres] 
    if debug: print(">> assigned fates:", assignedFate, sum(assignedFate))
    if stormFate: assignedFate[stormIntFinal] = 2
    if debug: print(">> nests with storm in final interval:", stormIntFinal)
    if debug: print(">> assigned fates after storm fates assigned:", assignedFate, sum(assignedFate))
    # fate cues prob should be affecting all nest fates equally, not just failures.
    # if debug: print(">> proportion of nests assigned hatch fate:", np.sum((assignedFate==0)[discovered==True])/(sum(discovered==True)),"vs period survival:", pSurv**hatchTime)
    # print(">> assigned fate array & its shape:\n", assignedFate, assignedFate.shape)
    return(assignedFate)
# -----------------------------------------------------------------------------
def svy_position(initiation, nestEnd, surveyDays):
    """ Finds index in surveyDays of iniatiation and end dates for each nest """
    position = np.searchsorted(surveyDays, initiation) 
    if debug: print(">> position of initiation date in survey day list:\n", position, len(position)) 
    
    position2 = np.searchsorted(surveyDays, nestEnd)
    if debug: print(">> position of end date in survey day list:\n", position2, len(position2)) 
    
    return((position, position2)) # return a tuple
    # position2
# -----------------------------------------------------------------------------
# def observer(par, fateCues, fates, surveyDays, nData, out):
def observer(nData, par, cues, fate, surveys, out):
    """
    The observer searches for nests on survey days. Surveys til discovery (success)
    are calculated as random draws from a negative binomial distribution with
    daily success probability of discProb. If surveys til discovery is less
    than total number of surveys while nest is active, then nest is discovered.

    The observer then assigns fate in assign_fate. 

    output: ndarray w/ nrows=numNests. cols=assignedFate, numObsInt, i, j, k, stormIntFinal
    """
    initiation = nData[:,1]
    end        = nData[:,1] + nData[:,2]
    numNests, obsFreq, discProb, stormFate = par # unpack par

    # svy_til_disc = discover_time(discProb, numNests)
    pos              = svy_position(initiation, end, surveys)
    num_svy          = pos[1] - pos[0]   
    svysTilDiscovery = rng.negative_binomial(n=1, p=discProb, size=numNests) # see above for explanation of p 
    discovered       = svysTilDiscovery < num_svy
    num_svy[discovered] = 0
    stormIntFinal    = surveyInts[pos[1]] > obsFreq  # was obs interval longer than usual? (== there was a storm)
    # out is already a 2d array of zeros
    # out = num observations, i, j, k, assigned fate
    out[:,0] = assign_fate(cues, fate, numNests, stormIntFinal, stormFate)
    out[:,1] = num_svy - svysTilDiscovery  # number of observations for the nest
    # NOTE would these be quicker w/o the mask??
    # can get discovered/not from out[:,1], I think
    out[:,2][discovered] = surveys[pos[0]+svysTilDiscovery][discovered] # i
    out[:,3][discovered] = surveys[pos[1]][discovered] # j
    out[:,4][discovered] = surveys[pos[1]+1][discovered] # k
    out[:,5] = stormIntFinal
    
    if debug: print("assigned fate, num obs, i, j, k:\n", out)
    return(out)
# -----------------------------------------------------------------------------
#   MAYFIELD & JOHNSON
# -----------------------------------------------------------------------------
def exposure(inp, expPercent=0.5): 
    """
    Calculate the exposure period for a nest (number of days observed)
        > the ijk values should tell you failed vs hatched
        > I think I couldn't get it to work as vectorized, so I used a loop
    For the basic case where psurv is constant across all nests and times:
      1. count the total number of alive days
      2. count the number of days in the final interval (for failed nests)
      3. calculate the exposure
            > #days before final int + (#days in final int * expPercent)
                > expPercent = percent of final interval nest is assumed alive
                > Mayfield used 50%, Johnson corrected it to 40%
                > final interval assumed to be zero days for hatched nests, which
                 were found after hatching (exposure of incubation period is over)
            > not calculating nestling exposure bc precocial/semi-precocial chicks
              leave the nest so early 
        
    inp = [i,j,k] for all nests in set\n
    default expPercent is from Mayfield; Johnson recommended 0.4
    
    Returns: ndarray. nrows=len(inp); cols=alive_days, final_int, exposure
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
        if debug: print(
            "days nest was alive:", expo[n,0],
            "& final int:", expo[n,1], 
            "& exposure:", expo[n,2]
            )
        # NOTE need nests to be alive for at least one interval
    if debug: print("output from exposure function:", expo)
    return(expo)
# -----------------------------------------------------------------------------
# def mayfield(ndata, expo):
def mayfield(num_fail, expo):
    """ 
    The Mayfield estimator of DSR 
    
    Mayfield's original estimator was defined as: 
            > DSR = 1 - (# failed nests / # exposure days)
    so if DSR = 1 - daily mortality, then:
            > daily mortality = # failed nests / # exposure days
    
    Arguments:
        num_fail = count of failed nests (total-hatched)
        expo     = output from exposure function, which is a np 2darray 

    Returns: the daily mortality 
    """
#    I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
#    dat = ndata[
    # hatched = np.sum(ndata[:,3])
    # failed = len(ndata) - hatched
    # expo is output from exposure function
    exposure = expo[2]
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
    mayf = num_fail / (exposure.sum())
    if debug: print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

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
def calc_dsr(nData, nestType):
    """ Calculate exposure and DSR for a given set of nests. Returns DSR value. """
    nNests  = len(nData)
    hatched = len(nData[:,3] == 0)
    # expDays = exposure(nestData[:,6:9], numNests=numN, expPercent=0.4)
    expDays = exposure(nestData[:,6:9], expPercent=0.4)
    dmr     = mayfield(num_fail=nNests-hatched, expo=expDays)
    if debug:
        print(
            # f"> {nestType} nests - hatched:", hatched.sum(),
            f"> {nestType} nests - hatched:", hatched,
            "; failed:", nNests - hatched, 
            "; exposure days:", expDays[:,2].sum(),
            "; & Mayfield-40 DSR:", 1-dmr
            )
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
def print_prop(assignedFate, trueFate, discovered):  
    aFates = [np.sum((assignedFate == x)[discovered==True]) for x in range(4)]
    # this proportion needs to be out of nests discovered AND assigned
    aFatesProp = [np.sum((assignedFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
    tFates = [np.sum((trueFate == x)[discovered==True]) for x in range(4)]
    tFatesProp = [np.sum((trueFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
    if debug: print(
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
    if debug_nest: print("nestData, discovered only:\n", nestData)
    if debug_nest: print(">> proportion of nests assigned hatch fate:", 
                    np.sum(nestData[:,3]==0)/(nDisc),
                    "vs period survival:", 
                    pSurv**hatchTime)
    if debug_nest:
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
# def mark_probs(s, expo, ndata):
def mark_probs(s, ndata):
    """
    creates probabilities for nest observation histories for use with prog_mark.
        > uses prob surv (s), so needs to be inside the optimizer

    Probabilities for Program MARK:
        1. create vectors to store:
             a) the probability values for each nest
             b) the degrees of freedom for each value
        2. calculate the exposure, alive_days, & final_int days
        3. fill the vectors
    
    Note that failed nests have final_int>0 while hatched nests have final_int=0

    Probability equation: 
      > daily probability of survival (DSR) raised to the power of intervals 
        nest was known to be alive
      > for hatched nests, that's it
      > for failed nests, exact failure date allowed to be unknown
          > but we know nest wasn't alive for the entire final interval
          > so add in probability of NOT surviving one interval (1-DSR)
      > EX: if probability of surviving from day 1-3 is s1*s2*s3, then
      >     probability of failure sometime during days 4-6 is 1-s4*s5*s6
      > hatched nests also have one extra degree of freedom (dof)
      
    Returns: list with the allp and alldof arrays inside
    """
    # exp[hatched==True] = nDays[hatched==True] # do I need the ==True?
    allp   = np.array(range(1,len(ndata)), dtype=np.longdouble) # all nest probabilities 
    alldof = np.array(range(1,len(ndata)), dtype=np.double) # all degrees of freedom
    expo = exposure(inp=ndata[:,6:9], expPercent=0.4)
    for n in range(len(ndata)-1): # want n to be the row NUMBER
        alive_days = expo[n,0]
        final_int  = expo[n,1]
        exposure   = expo[n,2]
        if final_int > 0:
            # if final_int==0 (hatched), s^final_int = 0
            p   = (s**alive_days)*(1-(s**final_int)) 
            dof = alive_days
            #print(">> nest", inp[n,0], "failed. likelihood=", p)
        else:
            p   = s**alive_days
            dof = alive_days + 1
            #print(">> nest", inp[n,0], "hatched. likelihood=", p)
            # NOTE is it likelihood or probability??
        allp[n]   = p # NOTE this line is throwing the Deprecation Warning
        # apparently warning means that n is a 2d array, so need to index it
        # never mind, apparently it's one dimensional 
        alldof[n] = dof
        #    once we have all the probabilities then...?
        #    where do exposure days come into it?
        #    might just be alive_days + final_int
        #    technically, raise prob to power of frequency
        #    we are assuming each occurs only once
        return([allp, alldof])
# -----------------------------------------------------------------------------
def prog_mark(s, ndata, probs, nocc):
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
    prob, dof = probs
    # print("all nests:", len(ndata)) 
    disc = ndata[np.where(ndata[:,6]!=0)] # i != 0
    if debugM: print(">>>>> Program MARK >>>>>>>>>>")
    if debugM: print("> number of nests:", len(ndata), "discovered nests:", len(disc))
    inp = disc[:,np.r_[0,7:11]] # doesn't include index 11
    if debugM: print("inp (ID, i, j, k, fate:)\n",inp)
    l    = len(inp)
    if debugM: print("l=", l, "| s=", s, "| nocc=", nocc)
    if debugM: print("----------------------------")

    inpInd = inp[(inp[:,2]-inp[:,1]) != 0] # access all rows; 
    #                                     create mask, index inp using it
    allp = prob[inpInd]
    alldof = dof[inpInd]
    if debugM: print(">> all nest cell probabilities:\n", allp)
    if debugM: print(">> all degrees of freedom:\n", alldof)
    lnp  = -np.log(allp) # vector of all log-transformed probabilities
    if debugM: print("log of all nest cell probabilities:", lnp)
    #print(">> negative log likelihood of each nest cell probability:", lnp)
    #lnSum = lnp.sum()
    #NLL = -1*lnp.sum()
    NLL = lnp.sum()
    if debugM: print(">> sum to get negative log likelihood of the data:", NLL)
    return(NLL)
# -----------------------------------------------------------------------------
def mark_wrapper(srn, ndata, prob, nocc):
    """
    This function calls the program MARK function when given a random starting 
    value (srn) and some nest data (ndata)
        > values given to optimizer are transformed then passed to MARK function
            > allows larger range of values for optimizer to work over w/o overflow
            > but values given to the function are still between 0 and 1, as required
        > Create vector to store the log-transformed values, then fill

    """
    numNests = len(ndata)
    s = np.ones(numNests, dtype=np.longdouble)
    s = logistic(srn)
    # NOTE is this multiple random starting values (for each nest) or one random starting value?
    #@#print("logistic of random starting value for program MARK:", s, s.dtype)
    # the logistic function tends to overflow if it's a normal float; make it np.float128
    ret = prog_mark(s, ndata, prob, nocc)
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

        disc    = nest[7].astype(int)    # first found
        endObs  = nest[9].astype(int)    # last observed
        hatched = nest[3]
        flooded = nest[4]

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
def interval(pwr, stateMat): 
    stateEnd, stateLC = stateMat
    stillAlive = np.array([1,0,0]) 
    # nests always start alive, or they wouldn't be checked
    TstateI = np.transpose(stillAlive)  # this is just one, not a vector?
    normalInt = stateEnd@pwr@TstateI
    # print(
    #     ">> likelihood of one normal interval:\n",
    #     normalInt,
    #     normalInt.shape,
    #     normalInt.dtype)

    # The final interval is one of these two (ends in final state):
    # normalFinal = stateLC@pwr@TstateI
    finalInt = stateLC@pwr@TstateI
    # NOTE now pwr has storms incorporated
    # print("final interval:", normalFinal, "and -log likelihood:", -np.log(normalFinal))
    # stormFinal  = stateLC@pwrStm@TstateI
    # finalInt[stormTrue] = stormFinal

    return([normalInt, finalInt])
# -----------------------------------------------------------------------------
def nest_mat(argL, obsFreq, stormFin, useStormMat):
    
    a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
    trMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
    trMatStm = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]]) 
    if debugLL: print(">> transition matrix\n:", trMatrix, trMatrix.shape)
    pwr = np.linalg.matrix_power(trMatrix, obsFreq) # raise the matrix to the power of the number of days in obs int
    if debugLL: print(">> transition matrix to the obs int power:\n", pwr, pwr.shape)

    # NOTE NOTE should it be the regular matrix or the storm matrix??
    ###### power equation for storm intervals (longer obs int): ###############
    # pwr is a 3x3 matrix; can't index with stormFin
    # it isn't a list of 3x3 matrices to use for multiplication
    # but the output is ultimately a 1x3 matrix, so just need to do all the multiplication at once?
    if useStormMat:
        # pwr[stormFin] = np.linalg.matrix_power(trMatStm, obsFreq*2) 
        pwrStm = np.linalg.matrix_power(trMatStm, obsFreq*2) 
    else:
        pwrStm = np.linalg.matrix_power(trMatrix, obsFreq*2) 
    if debugLL: 
        print(">> transition matrix to the obs int power (w/storms):\n", pwrStm, pwr.shape)
    # if stormTrue:
    #     return(pwrStm)
    # else:                                
    #     return(pwr)
    return([pwr, pwrStm])
# def logL(normalInt, normalFinal, stormFinal, numInt):
# ---------------------------------------------------------------------------------------------------
def logL(numNests, normalInt, finalInt, numInt, ha):
# def logL(normalInt, finalInt, numInt):
    # numNests is not the total number (param value) but the number not excluded
    # 2. Initialize the overall likelihood counter; Decimal gives more precision
    # numNests = len(normalInt)
    logLike = Decimal(0.0)         
    logLik   = np.ones(numNests, dtype=np.longdouble) # this should give it enough precision & avoid errors
    logLik      = logLik * np.log(normalInt) * -1 # dtype changes to float64 unless you multiply it by itself
    if debugLL: print(">> -log likelihood of 1 interval:", logLik)
    # logLikFin    = np.ones(numNests, dtype=np.longdouble)
    logLikFin = np.empty(numNests, dtype=np.longdouble)
    # logLikFin.fill(-np.log(normalFinal))
    # logLikFin= -np.log(normalFinal)
    logLikFin= -np.log(finalInt)
    # logLikFin[hatched == True] = 0
    logLikFin[ha==True] = 0
    # now stormFinal should be part of normalFinal (finalInt)
    # logLikFin[flooded == True] = -np.log(stormFinal[flooded==True])
    # logLikFin[]    = logLikFin * (-np.log(normalFinal))
    # logLikFinStm = logLikFinStm * (-np.log(stormFinal))
    if debugLL:
        print(">> log likelihood final interval, updated with storms/hatch:\n", logLikFin)

    # stormDuringFin = nestData[:,12] # was there a storm during the final interval?
    logLikelihood  = (logLik*numInt) + (logLikFin)
    if debugLL: print(">> log likelihood of each nest history:", logLikelihood)

    logLike        = np.sum(logLikelihood)
    if debugLL: print(">>>> overall log likelihood:", logLike)
    if debugLL: printLL(numNests=numNests, logLik=logLik, logLikFin=logLikFin, 
                        numInt=numInt, logL=logLikelihood)
    return(logLike) # this is what is being optimized
# -----------------------------------------------------------------------------
def printLL(numNests, logLik, logLikFin, numInt, logL):

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
# try to keep these in numpy:
# def like(perfectInfo, hatchTime, argL, numNests, obsFreq, obsDat, surveyDays):
# def like(argL, numN, obsFr, obsDat, stMat, sTrue, useSM):
# This one uses a matrix multiplication equation created from building blocks
#   > so you create these blocks:
#   > a normal interval, a final interval, and a final interval with storm
def like(argL, numN, obsFr, obsDat, stMat, useSM):
    # perfectInfo == 0 or 1 to tell you whether you know all nest fates or not
    # ---------------------------------------------------------------------------------------------------
    # 1. Unpack:
    #    a. Initial values for optimizer:
    # a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
    #    b. Observation history values from nest data:
    # fl, ha, ff, la, lc, nInt, sTrue = obsDat
    # fl, ha, ff, la, lc, nInt = obsDat
    
    # numpy array should be unpacked along the first dimension, so transpose:
    fate, nInt, ff, la, lc, sFinal = obsDat.T
    fl = fate==2 # are these necessary for more than print statements?
    ha = fate==0
    # NOTE NOTE should this be the assigned fate or true fate?
    if debug:
        print(
            ">>>> run likelihood function - ",
            "num flooded:", sum(fl), "& hatched:", sum(ha),
            "number of obs intervals:", nInt,
            "\n>> ijk:\n", [ff, la, lc]
            )
    # stEnd, stFin = stMat 
    pwrOut = nest_mat(argL=argL, obsFreq=obsFr, stormFin=sFinal, useStormMat=useSM)
    pwr, pwrStm = pwrOut
    # inter  = interval(pwr=pwr, stateEnd=stEnd, stateLC=stFin)   
    inter  = interval(pwr=pwr, stateMat=stMat)   
    norm, fin = inter
    # llVal = logL(normalInt=norm, normalFinal=fin, stormFinal=sfin, numInt=nInt)
    llVal = logL(numNests=numN, normalInt=norm, finalInt=fin, numInt=nInt, ha=ha)
    # make sure numN is the number of analyzed nests, not the param value (total number)
    
    return(llVal)
# this function does the matrix multiplication for a SINGLE interval of length intElt days 
 # during observaton, nest state is assessed on each visit to form an observation history 
 # the function calculates the negative log likelihood of one interval from the observation history 
 # these can then be multiplied together to get the overall likelihood of the observation history 
# -----------------------------------------------------------------------------
#   THE LIKELIHOOD WRAPPER FUNCTION
# -----------------------------------------------------------------------------
def like_smd( 
        # x, perfectInfo, hatchTime, nestData, obsFreq, 
        # stormDays, surveyDays, whichRet):
        x, obsData, obsFreq, stateMat, useSM, whichRet=1, **kwargs):
    # use kwargs for stormDays & surveysDays, which are only needed w/ like_old
    """
    The values are log-transformed before running them thru the likelihood 
    function, so the values given to the optimizer are the untransformed values,
    meaning the optimizer output will also be untransformed.
        >> Therefore, need to transform the output as well.
        
    Arguments:
        x: output from randArgs(); 5 values, only 4 used by LL function
        whichRet: which like() function should be used? 1=new (default); 2=old
    """


    # unpack the initial values:
    s0   = x[0]
    mp0  = x[1]
    ss0  = x[2]
    mps0 = x[3]
    sM   = x[4]
    #@#print("initial values:", s0, mp0, ss0, mps0, sM)

    # transform the initial values so all are between 0 and 1:
    s1   = logistic(s0)
    mp1  = logistic(mp0)
    ss1  = logistic(ss0)
    mps1 = logistic(mps0)
    #@#print("logistic-transformed initial values:", s1, mp1, ss1, mps1)

    # further transform so they remain in lower left triangle:
    tri1 = triangle(s1, mp1)
    tri2 = triangle(ss1, mps1)
    s2   = tri1[0]
    mp2  = tri1[1]
    ss2  = tri2[0]
    mps2 = tri2[1]
    #@#print("triangle-transformed initial values:", s2, mp2, ss2, mps2)

    # compute the conditional probability of mortality due to flooding:
    mf2  = 1.0 - s2 - mp2
    mfs2 = 1.0 - ss2 - mps2

    numNests = obsData.shape[0]
    #@#print(">> number of nests:", numNests)

    # call the likelihood function:
    argL = np.array([s2,mp2,mf2,ss2,mps2,mfs2, sM])
    #ret = like(argL, ndata, obs, storm, survey)
    #ret = like(argL, nestData, obsFreq, stormDays, surveyDays)
    # def like(argL, numN, obsFr, obsDat, stMat, useSM):
    # obsDat is a subset of nestData
    
    if whichRet == 1:
        ret = like(argL, numN=numNests, obsFr=obsFreq, obsDat=obsData, 
                   stMat=stateMat, useSM=useSM)
        # print('like_smd(): Msg : ret = ', ret)
    
    # else:
    elif whichRet == 2:
        ret = like_old(argL, obsFreq, obsData, surveyDays, stormDays)
        # print('like_smd(): Msg : using old function; ret = ', ret)
    
    else:
        print(' argument whichRet is invalid ')
    
    # rets = np.array([ret, ret2])

    return(ret)

# ---------------------------------------------------------------------------------------
# def ansTransform(ans, unpack):
# if you add unpack param, can use for transforming input as well as output
# def ansTransform(ans, unpack=True):
def ansTransform(ans):
    # if unpack:
    #     ans = ans.x    
    s0   = ans.x[0]         # Series of transformations of optimizer output.
    mp0  = ans.x[1]         # These make sure the output is between 0 and 1, 
    ss0  = ans.x[2]         # and that the three fate probabilities sum to 1.
    mps0 = ans.x[3]

    s1   = logistic(s0)
    mp1  = logistic(mp0)
    ss1  = logistic(ss0)
    mps1 = logistic(mps0)

    ret2 = triangle(s1, mp1)
    s2   = ret2[0]
    mp2  = ret2[1]
    mf2  = 1.0 - s2 - mp2

    ret3 = triangle(ss1, mps1)
    ss2  = ret3[0]
    mps2 = ret3[1]
    mfs2 = 1.0 - ss2 - mps2
    
    #ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.float128)
    ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.longdouble)
    # print(">> results as an array:\n", ansTransformed)
    # if debug: print(">> results (s, mp, mf, ss, mps, mfs, ex):\n", s2, mp2, mf2, ss2, mps2, mfs2, ex)
    return(ansTransformed)

# -----------------------------------------------------------------------------
def randArgs():
    """
    Choose random initial values for the optimizer
    These will be log-transformed before going through the likelihood function
    
    Returns:
    array of s, mp, ss, mps (for like_smd) and srand (for mark_wrapper)
    """
    s     = rng.uniform(-10.0, 10.0)       
    mp    = rng.uniform(-10.0, 10.0)
    ss    = rng.uniform(-10.0, 10.0)
    mps   = rng.uniform(-10.0, 10.0)
    srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
    z = np.array([s, mp, ss, mps, srand])

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
def run_optim(fun, z, arg, met='Nelder-Mead'):
    """
    How do I get it to continue the loop if ex is returned?
    """
    try:
        ans = optimize.minimize(fun, z, args=arg, method=met)
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
    # #@#print("Success?", ans.success, ans.message)
    res = ansTransform(ans)
    return(res)
    
def make_obs(par, init, dfs, stormDays, surveyDays, config):
    # ---- make the nests: ---------------------------------------------------
    nData          = mk_nests(params=par, init=init[0], 
                                 weekStart=init[1], nestData=nd)
    nestPeriod     = mk_per(nData[:,1], (nData[:,1]+nData[:,2]))
    stormsPerNest  = storm_nest(nestPeriod, stormDays)
    flooded        = mk_flood(par, stormsPerNest, numNests=par[par.numNests])
    hatched        = nData[:,2] >= par.hatchTime
    nestFate       = mk_fates(par.numNests, hatched, flooded)
    # ---- observer: ---------------------------------------------------------
    par2      = [par.numNests, par.obsFreq, par.discProb, par.stormFate]
    obs       = observer(nData, par=par2, cues=par.fateCues, fate=nestFate, 
                         surveys=surveyDays, out=nd2)
    # ---- concatenate to make data for the nest models: ---------------------
    #   0. nest ID                  5. # obs int                
    #   1. initiation               6. first found       
    #   2. survival (w/o storms)    7. last active              
    #   3. fate                     8. last checked   
    #   4. assigned fate            9. stormIntFinal
    #                               10. num storms                 
    # fl, ha, ff, la, lc, nInt, sTrue = obsDat
    nestData = np.concatenate((nData, 
                               nestFate[:,None], 
                               obs,
                               stormsPerNest[:,None]
                               ), axis=1)
    if config.debug: print("nestData:\n", nestData)
    return(nestData)
    
# def rep_loop(par, repID, nData, vals, storm, survey, config):
def rep_loop(par, nData, vals, storm, survey, config):
    """
    For each data replicate, call this function, which:
        - takes the reduced nest data as input
        - calls the optimizer on like_smd() and mark_wrapper()
        MAYBE this is too much for one function???
        but don't know what it should return
        
    Returns:
        like_val
    """
    # ---- empty arrays to store data for this replicate: ---------
    likeVal  = np.zeros(shape=(config.numOut), dtype=np.longdouble)
    trueDSR_an   = calc_dsr(nData=nData, nestType="analysis") 
    perfectInfo = 0
    # ex = np.longdouble("0.0")
    stMat = state_vect(nNest=len(nData), fl=(nData[:,3]==2), ha=(nData[:,3]==0))
    dat = nData[:,4:10] # doesn't include column index 10

    llArgs = randArgs()
    testLL  = like_smd(x=llArgs, obsData=dat, obsFreq=par.obsFreq, 
                       stateMat=stMat, useSM=par.useSMat, whichRet=1)
    ans      = run_optim(fun=like_smd, z=randArgs(), 
                         arg=(dat, par.obsFreq, stMat, par.useSMat, 1))
                                # args=( dat, obsFr, stMat, useSM, 1),
    res = ansTransform(ans)
    srand = rng.uniform(-10.00, 10.00)
    markProb = mark_probs(s=srand, ndata=nData)
    # nocc     = 
    # markAns  = run_optim(fun=mark_wrapper, z=randArgs()[4], arg=(nData))
    ans2 = run_optim(fun=mark_wrapper, z=srand, arg=(nData, markProb, par.brDays))
    #NOTE ans2 is an "OptimizeResult" object; need to extract "x"
    # Transform the MARK optimizer output so that it is between 0 and 1:
    mark_s = logistic(ans2.x[0]) # answer.x is a list itself - need to index
    # print("> logistic of MARK answer:", mark_s)
    
    s2,mp2,mf2,ss2,mps2,mfs2 = res # unpack like() function output
    # s2,mp2,mf2,ss2,mps2,mfs2 = res2 # unpack like_old() function output
    like_val = [
            repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,
            trueDSR,trueDSR_an,pSurv, pSurvStorm,pMFlood,
            # hatchTime,numNests,obsFreq,discovered,excluded,ex
            hatchTime,numN,obsFreq,discovered,excluded
            ]
    if config.debug: print(">> like_val:\n", like_val)
    like_val = np.array(like_val, dtype=np.longdouble)
    return(like_val)
    
def param_loop(par, parID, storm, survey, config, nruns=1 ):
    """
    1. create repID & numMC counters
    2. for each data replicate:
        a. create empty arrays to store nest data
        b. create nest data
        c. calculate DSR for all nests & discovered nests
        d. exclude nests with assigned fate == 7 (unknown) or only 1 observation
        e. add up counts of num excluded, num discovered, & num analyzed
        f. pass the reduced data & the DSR/counts to rep_loop() for the optimizer
    """
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>> param set number:", parID)
    repID      = 0  # keep trackof replicates
    numMC      = 0 # number of nests misclassified
    nrows   = len(par)*config.nreps*nruns
    for r in range(config.nreps): 
        if config.debug: print("\n>>>>>>>>>>>> replicate ID:", repID)
        nd       = np.zeros(shape=(par.numNests, 3), dtype=int)
        nd2      = np.zeros(shape=(par.numNests, 6), dtype=int)
        nestData = make_obs(par=par, init=mk_init(), dfs=[nd,nd2], stormDays=storm, 
                            surveyDays=survey) 
        trueDSR      = calc_dsr(nData=nestData, nestType="all")
        disc = sum(np.where(nestData[:,6]!=0))
        exclude  = ((nestData[:,9] == 7) | (nestData[:,6]==nestData[:,7]))                         
        # if there's only one observation, firstFound will == lastActive
        excl = sum(exclude) # exclude = boolean array; sum = num True
        rem  = disc - excl
        # this one is mostly for debugging purposes, to make sure nothing
        # weird is happening with the discovery process:
        trueDSR_disc = calc_dsr(nData=nestData[np.where(nestData[:,6]!=0)],
                                nestType="discovered")
        # -------------------------------------------------------------
        # Then remove nests with unknown fate or only 1 observation:
        nestData    = nestData[~(exclude),:]    # remove excluded nests 
        rep_loop(par=par, nData=nestData, vals=[trueDSR,disc,excl,rem], storm=storm,
                 survey=survey,config=config)
        
def main(testing=False, fname=mk_fnames(), pStatic=staticPar):
    
    config = Config(
        rng         = np.random.default_rng(seed=102891), 
        args        = sys.argv,
        nreps       = 500,
        debug       = False,
        debugLL     = False,
        debugNests  = False,
        likeFile    = fname[0],
        colNames    = fname[1]
        numOut      = 21)

    if testing == True:
        config = Config( # update the config?
            # rng         = np.random.default_rng(seed=102891), 
            # args        = sys.argv,
            nreps       = 5,
            debug       = True,
            debugLL     = True,
            debugNests  = True,
            # likeFile    = fname[0],
            # colNames    = fname[1],
            # numOut      = 21
        )
        pList = plTest
    else:
        pList = parLists # don't need to update any settings if not testing?
    
    with open(config.likeFile, "wb") as f:
    # with open(likeFile, "ab") as f: # changing this to append didn't help...
        # append shouldn't matter if the file is just open the whole time
        paramsArray = mk_param_list(parList=pList)
        par_merge   = {paramsArray, pStatic} # merge the dictionaries
        params      = Params(par_merge)
        parID       = 0
        for i in range(0, len(paramsArray)): # for each set of params
            par        = params[i] 
            par.fateCues   = 0.65 if par.obsFreq > 5 else 0.71 if par.obsFreq == 5 else 0.75
            stormDays  = stormGen()
            surveyDays = mk_surveys(stormDays, par.obsFreq, par.brDays)
            # surveyInts = survey_int(surveyDays)
            param_loop(par=par, parID=parID, storm=stormDays, survey=surveyDays, 
                       config=config)
                    #    nreps=settings.nreps)
    
if False:
    with open(likeFile, "wb") as f:
        # with open(likeFile, "ab") as f: # changing this to append didn't help...
        # append shouldn't matter if the file is just open the whole time
        paramsArray = mk_param_list(parList=plTest)
        parID     = 0
        for i in range(0, len(paramsList)): # for each set of params
            #par = paramsList[i]
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(">>>>>>>>> param set number:", parID)
            # paramsArray is a 2d array; loop through the rows 
            # each row is a set of parameters we are trying
            #   0          1        2           3           4            5
            #numNests, probSurv, stormDur, stormFreq, hatchTime, obsFreq
            par        = paramsArray[i] 
            # in an array, all values have the same numpy dtype (float in this case) 
            # after selecting the row, unpack the params & change dtype as needed:
            numN, pSurv, freq, dur, hTime, obsFr, stormF, useSM = par
            fateCues   = 0.6 if obsFr > 5 else 0.66 if obsFr == 5 else 0.75
            # 80% chance that field cues about nest fate are present/observed
            # based on around 80% success rate in identifying nest fate (from camera study)
            # this is essentially uncertainty?
            # NOTE do you want new storms/survey days for each replicate 
            #      or each parameter set?
            stormDays  = stormGen()
            surveyDays = mk_surveys(stormDays, obsFr, brDays)
            surveyInts = survey_int(surveyDays)
            repID      = 0  # keep trackof replicates
            numOut     = 21 # number of output params
            numMC      = 0 # number of nests misclassified
            nrows   = len(paramsList)*nreps*nruns

            print(
                ">>> nest params in this set:", 
                pSurv, 
                pSurvStorm, 
                dur, 
                freq, 
                hatchTime, 
                obsFreq, 
                fateCues
                )
            
            if True: # don't record nest data to file
                # For each replicate:
                # > create nest data
                # > create random starting values
                # > run the optimizer on the MCMC function & the MARK function
                # > record the output to the bigger list
                for r in range(nreps): 
                    if debug:
                            print(
                            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                            "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    if debug: print("\n>>>>>>>>>>>> replicate ID:", repID)
                    # -------------------------------------------------------------
                    # ---- empty arrays to store data for this replicate: ---------
                    likeVal =  np.zeros(shape=(numOut), dtype=np.longdouble)
                    nd      = np.zeros(shape=(numN, 3), dtype=int)
                    nd2     = np.zeros(shape=(numN, 6), dtype=int)
                    # -------------------------------------------------------------
                    # ---- make the nests: ----------------------------------------
                    nData          = mk_nests(params=par, init=initProb, 
                                                 weekStart=weekStart, nestData=nd)
                    nestPeriod     = mk_per(nData[:,1], (nData[:,1]+nData[:,2]))
                    stormsPerNest  = storm_nest(nestPeriod, stormDays)
                    flooded        = mk_flood(par, stormsPerNest, numNests=numN)
                    hatched        = nData[:,2] >= hatchTime
                    nestFate       = mk_fates(numN, hatched, flooded)
                    # -------------------------------------------------------------
                    # ---- observer: ----------------------------------------------
                    par2      = [numN, obsFreq, pDisc, stormF]
                    # obs       = observer(par2, fateCues, nestFate, surveyDays, nData, nd2)
                    obs       = observer(nData, par=par2, cues=fateCues, fate=nestFate, 
                                         surveys=surveyDays, out=nd2)
                    # -------------------------------------------------------------
                    # / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / 
                    # -------------------------------------------------------------
                    # concatenate to make data for the nest models:
                    #   0. nest ID                  5. # obs int                
                    #   1. initiation               6. first found       
                    #   2. survival (w/o storms)    7. last active              
                    #   3. fate                     8. last checked   
                    #   4. assigned fate            9. stormIntFinal
                    #                               10. num storms                 
                    # fl, ha, ff, la, lc, nInt, sTrue = obsDat
                    nestData = np.concatenate((nData, 
                                               nestFate[:,None], 
                                               obs,
                                               stormsPerNest[:,None]
                                               ), axis=1)
                    if debug: print("nestData:\n", nestData)
                    trueDSR = calc_dsr(nData=nestData, nestType="all") 
                    # not sure where indexError will show up again
                    # try: # create nest/observer data
                    #     # np.save(n, nestData) # make sure this is correct kind of save
                    # except IndexError as error:
                    #     print(
                    #         ">> !! IndexError in nest data:", 
                    #         error,
                    #         ". Go to next replicate.")
                    #     continue
                    # -------------------------------------------------------------
                    # / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / 
                    # -------------------------------------------------------------
                    # Keep only discovered nests, then count them:
                    nestData   = nestData[np.where(nestData[:,6]!=0)] 
                    discovered = int(nestData.shape[0])
                    exclude  = ((nestData[:,9] == 7) | 
                                (nestData[:,6]==nestData[:,7]))                         
                    # if there's only one observation, firstFound will == lastActive
                    excluded = sum(exclude) # exclude = boolean array; sum = num True
                    analyzed = discovered - excluded
                    trueDSR_disc = calc_dsr(nData=nestData, nestType="discovered")
                    # -------------------------------------------------------------
                    # Then remove nests with unknown fate or only 1 observation:
                    nestData    = nestData[~(exclude),:]    # remove excluded nests 
                    trueDSR_an  = calc_dsr(nData=nestData, nestType="analyzed")
                    # -------------------------------------------------------------
                    # / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / 
                    # -------------------------------------------------------------
                    # Run the optimizer on the likelihood function you choose:
                    # ------- LIKE_SMD --------------------------------------------
                    # print("main.py: Msg: Running optimizer") 
                    # args to like_smd:
                    #       x, nestData, obsFreq, stateMat, useSM, 
                    #       sFin, whichRet=1, **kwargs):
                    z = randArgs()
                    # NOTE: optimizer for matrix function takes an array (z) but 
                    #       optimizer for MARK takes one param (srand) which just 
                    #       happens to be in z
                    srand = z[4]
                    perfectInfo = 0
                    ex = np.longdouble("0.0")
                    stMat = state_vect(numNests=len(nestData),
                                       flooded=(nestData[:,3]==2),
                                       hatched=(nestData[:,3]==0)
                                       )
                    dat = nestData[:,4:10] # doesn't include column index 10
                    expsr = exposure(inp=nestData[:,6:9], expPercent=0.4)

                    llArgs = randArgs()
                    testLL  = like_smd(x=llArgs, obsData=dat, obsFreq=obsFr, 
                                       stateMat=stMat, useSM=useSM, whichRet=1)
                    # -------------------------------------------------------------
                    try:
                        ans = optimize.minimize(
                                like_smd, z, 
                                # args=( nestData, obsFreq, stMat, useSM, 1),
                            # obs doesn't have nests excluded
                                # args=( obs, obsFreq, stMat, useSM, 1),
                                args=( dat, obsFr, stMat, useSM, 1),
                                method='Nelder-Mead'
                                # method='SLSQP'
                                )
                        ex = 0.0
                    except decimal.InvalidOperation as error2:
                        ex=100.0
                        print(
                            ">> Error: invalid operation in decimal:", 
                            error2, 
                            "Go to next replicate."
                            )
                        continue
                    except OverflowError as error3:
                        ex=200.0
                        print(
                            ">> Error: overflow error:", 
                            error3, 
                            "Go to next replicate."
                            )
                        continue
                    #
                    # #@#print("Success?", ans.success, ans.message)
                    res = ansTransform(ans)
                    # -------------------------------------------------------------
                    #OPTIMIZER: MARK function
                    # inp = disc[:,np.r_[0,7:11]] # doesn't include index 11
                    markProb = mark_probs(s=srand, expo=expsr)
                    markAns  = optimize.minimize(
                            mark_wrapper, srand,
                            args=(nestData), 
                            method='Nelder-Mead'
                            )
                    # did the optimizer converge?
                    #@#print(">> success?", markAns.success, markAns.message) 
                    # this MLE seems unlikely to need exceptions, but who knows...
                    #NOTE markAns is an "OptimizeResult" object - does not match 
                    #     the other objects in like_val; need to extract "x"
                    if markAns.success == True:
                        # Transform the MARK optimizer output so that it is 
                        # between 0 and 1
                        mark_s = logistic(markAns.x[0])
                        # answer.x is a list itself - need to index
                        # print("> logistic of MARK answer:", mark_s)
                    else:
                        mark_s = 10001

                    # if debugM: print(">>> MARK:", mark_s, "new:", res[0], "old:", res2[0])

                    # Compile the optimizer output for each replicate with other important info 
                    # including params and the number of nests actually used in the analysis
                    # can get a proportion because we know how many nests were generated initially
                    # -------------------------------------------------------------
                    # Decide which likelihood function output to use for results:
                    # -------------------------------------------------------------
                    # s = res2[0]
                    s2,mp2,mf2,ss2,mps2,mfs2 = res # unpack like() function output
                    # s2,mp2,mf2,ss2,mps2,mfs2 = res2 # unpack like_old() function output
                    like_val = [
                            repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,
                            trueDSR,trueDSR_an,pSurv, pSurvStorm,pMFlood,
                            # hatchTime,numNests,obsFreq,discovered,excluded,ex
                            hatchTime,numN,obsFreq,discovered,excluded
                            ]
                    # is there a reason this wasn't a list?
                    if debug: print(">> like_val:\n", like_val)
                    #, "lengths:", [len(x) for x in like_val])
                    #like_val = np.array(like_val, dtype=np.float128)
                    like_val = np.array(like_val, dtype=np.longdouble)
                    
                    #####################
                    # this is being saved w/o the delimiter
                    # stack exchange says to make it a list of only one item
                    #np.savetxt(f, like_val, delimiter=",")
                    column_names     = np.array([
                        'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 
                        'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 
                        'storm_freq', 'psurv_real', 'psurv_found', 'psurv_given',
                        'stormsurv_given','pflood_given', 'hatch_time','num_nests',
                        # 'obs_int', 'num_discovered','num_excluded', 'exception'
                        'obs_int', 'num_discovered','num_excluded'
                        ])
                    # header values need to be stored to np.ndarray for np.savetxt; actually, needs to be a string
                    colnames = ', '.join([str(x) for x in column_names])
                    #if repID == 0:
                    #print(">> saving likelihood values")
                    # if parID == 0 | repID == 0: # only the first line gets the header
                    if parID == 0 and repID == 0: # only the first line gets the header
                    # if parID == 0 : # only the first line gets the header
                        np.savetxt(f, [like_val], delimiter=",", header=colnames)
                        #np.savetxt(f, like_val, delimiter=",", header=colnames)
                        if debug: print(">> ** saving likelihood values with header **")
                    else:
                        np.savetxt(f, [like_val], delimiter=",")
                        if debug: print(">> ** saving likelihood values **")
                    # except this adds the header before every line
                    # could just remove them later in R

                    repID = repID + 1
                    # print(">> repID increased:", repID)

            parID = parID +1
                    
        if False: # record nest data to file
            with open(nestfile, "wb") as n:
                # append data from each repl to nestfile as you loop through them
                for r in range(nreps):
                    # if you write likelihoods to a file as you go, shouldn't need 
                    # an array to store them -
                    # this should save memory, at least in this case?
                    print(
                        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                        "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print("\n>>>>>>>>>>>> replicate ID:", repID)
                    likeVal =  np.zeros(shape=(numOut), dtype=np.longdouble)
                    try:
                        nestData = mk_nests(par, initProb, stormDays, surveyDays)
                        np.save(n, nestData) # make sure this is correct kind of save
                    except IndexError as error:
                        print(
                            ">> IndexError in nest data:", 
                            error,
                            ". Go to next replicate")
                        continue
                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    hatched = nestData[:,3] # true hatched (storms accounted for)
                    #hatchProp = sum(hatched)/numNests
                    expDays = nestData[0,14]
                    trueDSR = 1 - ((numNests - hatched.sum()) / expDays)
                    print(
                        "> ALL NESTS - hatched:",
                        hatched.sum(),
                        "; failed:", 
                        numNests - hatched.sum(), 
                        "; exposure days:", 
                        expDays,
                        "; true DSR:",
                        trueDSR
                        )
                    # NOTE rearrange columns to make more sense?
                    ###########################################################
                    # Keep only discovered nests, then count them:
                    nestData   = nestData[np.where(nestData[:,6]==1)] 
                    discovered = nestData.shape[0]   
                    print(
                        ">> nests w/ only 1 obs while active:",
                        np.where(nestData[:,7] == nestData[:,8]),
                        "& unknown fate:",
                        np.where(nestData[:,10] == 7)
                        ) 
                    exclude  = ((nestData[:,10] == 7) | 
                                (nestData[:,7]==nestData[:,8]))                         
                    # if there's only one observation, firstFound will == lastActive
                    excluded = sum(exclude) # exclude = boolean array; sum = num True
                    analyzed = discovered - excluded
                    failed   = nestData.shape[0] - sum(nestData[:,3]) # num hatched
                    expDisc  = sum(nestData[:,15])
                    # trueDSR_disc = 1 - ((nestData.shape[0] - sum(nestData[:,3])) / 
                    # NOTE: should this be calculated before or after removing nests?
                    trueDSR_disc = 1 - (failed / sum(nestData[:,15]))
                    nestData    = nestData[~(exclude),:]    # remove excluded nests 
                    print(
                        "> DISCOVERED NESTS - total | analyzed: hatched:", 
                        discovered, "|", analyzed,
                        # "excluded from analysis:", excluded,
                        "failed:", failed, "|",
                        nestData.shape[0] - sum(nestData[:,3]),
                        "exposure days:", expDisc, "|", sum(nestData[:,15])
                        )
                    # (num nests - num hatched nests) / expDays
                    trueDSR_analysis = 1 - ((nestData.shape[0] - sum(nestData[:,3])) / 
                                              sum(nestData[:,15]) )
                    z = randArgs()
                    # NOTE: optimizer for matrix function takes an array (z) but 
                    #       optimizer for MARK takes one param (srand) which just 
                    #       happens to be in z
                    srand = z[4]
                    perfectInfo = 0
                    ex = np.longdouble("0.0")

                    # -------------------------------------------------------------
                    # Run the optimizer on the likelihood function you choose:
                    # -------------------------------------------------------------
                    # Run the optimizer with messages for exceptions
                    # -------------------------------------------------------------
                    
                    # ------- LIKE_SMD --------------------------------------------
                    # print("main.py: Msg: Running optimizer") 
                    try:
                        ans = optimize.minimize(
                                like_smd, z, 
                                args=(
                                    perfectInfo, 
                                    hatchTime, 
                                    nestData, 
                                    obsFreq, 
                                    stormDays, 
                                    surveyDays, 
                                    1
                                    ),
                                method='Nelder-Mead'
                                # method='SLSQP'
                                )
                        ex = 0.0
                    except decimal.InvalidOperation as error2:
                        ex=100.0
                        print(
                            ">> Error: invalid operation in decimal:", 
                            error2, 
                            "Go to next replicate."
                            )
                        continue
                    except OverflowError as error3:
                        ex=200.0
                        print(
                            ">> Error: overflow error:", 
                            error3, 
                            "Go to next replicate."
                            )
                        continue
                    #
                    # #@#print("Success?", ans.success, ans.message)

                    # # ------- LIKE_SMD_OLD ---------------------------------------
                    # print("main.py: Msg: Running optimizer on old function")
                    # try:
                    #     ans2 = optimize.minimize(
                    #             like_smd, z, 
                    #             #args=(nestData, obsInt, stormDays, surveyDays),
                    #             args=(
                    #                 perfectInfo, 
                    #                 hatchTime, 
                    #                 nestData, 
                    #                 obsFreq, 
                    #                 stormDays, 
                    #                 surveyDays,
                    #                 2
                    #                 ),
                    #             method='Nelder-Mead'
                    #             # method='SLSQP'
                    #             )
                    #     ex = 0.0
                    # except decimal.InvalidOperation as error2:
                    #     ex=100.0
                    #     print(
                    #         ">> Error: invalid operation in decimal:", 
                    #         error2, 
                    #         "Go to next replicate."
                    #         )
                    #     continue
                    # except OverflowError as error3:
                    #     ex=200.0
                    #     print(
                    #         ">> Error: overflow error:", 
                    #         error3, 
                    #         "Go to next replicate."
                    #         )
                    #     continue

                    # print("> answer:", ans.x)
                    # print("> answer (old function):", ans2.x)
                    res = ansTransform(ans)
                    # res2 = ansTransform(ans2)
                    # print("> res:\n", res)
                    # print("> res (old function):\n", res2)
                    print(
                        ">> true DSR of all nests:", 
                        trueDSR, 
                        "discovered nests:",
                        trueDSR_disc,
                        "and nests used in analysis:", 
                        trueDSR_analysis
                        )

                    #OPTIMIZER: MARK function
                    markAns  = optimize.minimize(
                            mark_wrapper, srand,
                            args=(nestData), 
                            method='Nelder-Mead'
                            )
                    # did the optimizer converge?
                    #@#print(">> success?", markAns.success, markAns.message) 
                    # this MLE seems unlikely to need exceptions, but who knows...
                    #NOTE markAns is an "OptimizeResult" object - does not match 
                    #     the other objects in like_val; need to extract "x"
                    if markAns.success == True:
                        # Transform the MARK optimizer output so that it is 
                        # between 0 and 1
                        mark_s = logistic(markAns.x[0])
                        # answer.x is a list itself - need to index
                        # print("> logistic of MARK answer:", mark_s)
                    else:
                        mark_s = 10001

                    # print(">>> MARK:", mark_s, "new:", res[0], "old:", res2[0])

                    # Compile the optimizer output for each replicate with other important info 
                    # including params and the number of nests actually used in the analysis
                    # can get a proportion because we know how many nests were generated initially
                    #firstPart = np.array([repID, mark_s])
                    #secondPart = np.array([dur, freq, pSurv, pSurvStorm, pMFlood, hatchTime, numNests, obsFreq, discovered, excluded, ex])
                    #like_val   = np.concatenate((firstPart, res, secondPart))

                    # -------------------------------------------------------------
                    # Decide which likelihood function output to use for results:
                    # -------------------------------------------------------------
                    # s = res2[0]
                    s2,mp2,mf2,ss2,mps2,mfs2 = res # unpack like() function output
                    # s2,mp2,mf2,ss2,mps2,mfs2 = res2 # unpack like_old() function output
                    like_val = [
                            repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,
                            trueDSR,trueDSR_analysis,pSurv, pSurvStorm,pMFlood,
                            # hatchTime,numNests,obsFreq,discovered,excluded,ex
                            hatchTime,numNests,obsFreq,discovered,excluded
                            ]
                    # is there a reason this wasn't a list?
                    print(">> like_val:\n", like_val)
                    #, "lengths:", [len(x) for x in like_val])
                    #like_val = np.array(like_val, dtype=np.float128)
                    like_val = np.array(like_val, dtype=np.longdouble)
                    
                    #like_val   = np.array([
                    #    repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,pSurv,pSurvStorm, 
                    #    pMFlood,hatchTime,numNests,obsFreq,discovered,excluded,ex
                    #    ], dtype=np.float128) # NOTE problem could be multiple types in same array?

                        #repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,stormDur,stormFreq,probSurv,SprobSurv, 
                        #probMortFlood,SprobMortFlood,hatchTime,numNests,obs_int,discovered,excluded,
                        #ex,stormMat
                    #####################
                    # this is being saved w/o the delimiter
                    # stack exchange says to make it a list of only one item
                    #np.savetxt(f, like_val, delimiter=",")
                    column_names     = np.array([
                        'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 
                        'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 
                        'storm_freq', 'psurv_real', 'psurv_found', 'psurv_given',
                        'stormsurv_given','pflood_given', 'hatch_time','num_nests',
                        # 'obs_int', 'num_discovered','num_excluded', 'exception'
                        'obs_int', 'num_discovered','num_excluded'
                        ])
                    # header values need to be stored to np.ndarray for np.savetxt; actually, needs to be a string
                    colnames = ', '.join([str(x) for x in column_names])
                    #if repID == 0:
                    #print(">> saving likelihood values")
                    if parID == 0 | repID == 0: # only the first line gets the header
                        np.savetxt(f, [like_val], delimiter=",", header=colnames)
                        #np.savetxt(f, like_val, delimiter=",", header=colnames)
                        # print(">> saving likelihood values")
                    else:
                        np.savetxt(f, [like_val], delimiter=",")
                        #np.savetxt(f, like_val, delimiter=",")
                        # print(">> saving likelihood values")
                    # except this adds the header before every line
                    # could just remove them later in R

                    repID = repID + 1
                    # print(">> repID increased:", repID)

            parID = parID +1
            # print(">> param set ID increased:", parID)
            

    # # sudo vim -o file1 file2 [open 2 files] 
    # # BLAH
    # # NOTE 5/16/25 - The percent bias responds more like I would expect when I use
    # #                the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
    # #                BUT I still don't know why the calculated DSR is consistently low.

    # from datetime import datetime
    # import decimal
    # from decimal import Decimal
    # from itertools import product
    # import itertools
    # import numpy as np 
    # from os.path import exists
    # import os
    # from pathlib import Path
    # from scipy import optimize