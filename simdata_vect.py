# sudo vim -o file1 file2 [open 2 files] 
# BLAH
# NOTE 5/16/25 - The percent bias responds more like I would expect when I use
#                the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
#                BUT I still don't know why the calculated DSR is consistently low.

from datetime import datetime
import decimal
from decimal import Decimal
from itertools import product
import itertools
import numpy as np 
from os.path import exists
import os
from pathlib import Path
from scipy import optimize
import scipy.stats as stats
import sys

# -----------------------------------------------------------------------------
#  SETTINGS 
# -----------------------------------------------------------------------------
rng = np.random.default_rng(seed=102891) 
args = sys.argv

debug = True
debug_obs = True
debugM = False # use different debug var bc these will print for every time optimizer runs
debugL = False
if len(args) > 1:
    debug = args[1] == "debugTrue"

# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS
# -----------------------------------------------------------------------------
# from https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    # print("filename, extension:", filename, extension)

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

# -----------------------------------------------------------------------------
def searchSorted2(a, b):
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
# This function computes intersection of 2 arrays more quickly than intersect1d
#    > possible observations = intersection of observable & survey days

#@profile
def in1d_sorted(A,B): 
    idx = np.searchsorted(B, A)
    idx[idx==len(B)] = 0
    return A[B[idx] == A]

# -----------------------------------------------------------------------------
#   OPTIMIZER PARAMETERS: 
# -----------------------------------------------------------------------------
nruns    = 1 # number of times to run  the optimizer 
#nreps   = 1000 # number of replicates of simulated nest data
# nreps   = 500
nreps    = 5
# nreps    = 1
makeNest = True
num_out  = 21 # number zof output parameters
#dir_name = datetime.now().strftime("%m%d%Y")
# -----------------------------------------------------------------------------
#   NEST MODEL PARAMETERS:
# -----------------------------------------------------------------------------
# NOTE the main problem is that my fate-masking variable (storm activity) also 
#      leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
breedingDays1   = [160] 
# discProb1       = [0.7, 0.9] 
discProb1       = [0.9] 
# numNests       = [500,1000]
# numNests       = [500]
# numNests       = [200]
numNests1        = [20]
uncertaintyDays = [1]
stormFate1       = [0,1]
print(
        "STATIC PARAMS: breeding season length:", breedingDays1[0],
        "; discovery probability:", discProb1[0]
        # "; number of nests:", numNests
        # "how long nest fate is discoverable (in days):", uncertaintyDays
        )
# only really need to be in lists if they are part of the combinations below
probSurv1       = [0.95 ]   # daily prob of survival
# probSurv       = [0.93, 0.95 ]   # daily prob of survival
probMortFlood1  = [0.1] # 10% of failed nests - not of all nests
# probMortFlood  = [0.95] # 95% of nests active during storm fail
SprobSurv1     = [0.2] # daily survival prob during storms - like intensity?
# SprobSurv      = [0.05] # daily survival prob during storms - like intensity?
SprobMortFlood = [1.0] # all failed nests during storms fail due to flooding
# floodSurv      = []
# stormDur       = [1,2,3]
# NOTE maybe duration should be in hours so I can test more values?
stormDur1       = [1,3]
# stormFreq      = [1,2,3]
stormFreq1      = [3,5]
# obsFreq        = [3,5,7]
obsFreq1        = [3,6,9]
# obsFreq        = [3]
# hatchTime      = [16,20,24,28]
# hatchTime      = [16,20,28]
hatchTime1      = [16, 28]
# whether or not nests that end during storms are marked "unknown" (vs. flooded)
#assignUnknown  = [0,1] 
assignUnknown1  = [0] 
#probCorrect     = 0.8
#fateCuesPresent = [0.7,0.8,0.9] 
fateCuesPresent1 = [0.8] 
# 80% chance that field cues about nest fate are present/observed
# based on around 80% success rate in identifying nest fate (from camera study)
# this is essentially uncertainty?
numMC           = 0 # number of nests misclassified
paramsList      = list(
    product(
        #    0        1         2          3         4          5          
        numNests1, probSurv1, SprobSurv1, stormDur1, stormFreq1, hatchTime1, 
        #   6          7              8           9           10
        # obsFreq, probMortFlood, breedingDays, discProb, fateCuesPresent
        obsFreq1, probMortFlood1, breedingDays1, discProb1, stormFate1
        )
        )
# NOTE fate cues prob will now be decided based on obs_int (11-14)
# NOTE the above probably only needs to be for the params that i'm testing?
#print(">> params list:", paramsList, "length:", len(paramsList))
paramsArray = np.array(paramsList) # don't want prob surv to be an integer!
print("NUMBER OF PARAM SETS:", len(paramsList))
nrows   = len(paramsList)*nreps*nruns
# -----------------------------------------------------------------------------
#   IMPORT REAL DATA
#   do it here once instead of every time you run the function
#   weekly storm probability and weekly nest initiation probability:
# -----------------------------------------------------------------------------
init= np.genfromtxt(
        #fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
        fname="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv",
        dtype=float,
        delimiter=",",
        skip_header=1,
        usecols=2
        )
# the initprob decimals in the csv don't sum to 1 anymore
initProb = init / np.sum(init) # make them into probabilities again

stormProb = np.genfromtxt(
        #fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
        fname="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv",
        dtype=float,
        delimiter=",",
        skip_header=1,
        usecols=3 # 4th column 
        )
storm_weeks2 = np.arange(14,29,1)
weekStart = (storm_weeks2 * 7) - 90 # why minus 90?
weekStart = weekStart.astype(int)
if debug:
    print(">> storm prob, by week:",stormProb, len(stormProb),"sum=",np.sum(stormProb))
    print(">> initiation prob, by week:",initProb,len(initProb),"sum=",np.sum(initProb))
    print(">> start day for each week:", weekStart, len(weekStart))
# -----------------------------------------------------------------------------
#   SAVE FILES 
# -----------------------------------------------------------------------------
# name for unique directory to hold all output:
dirName    = datetime.today().strftime('%m%d%Y_%H%M%S') 
print(">> save directory name:", dirName)
todaysDate = datetime.today().strftime("%Y%m%d")
likeFile   = Path(uniquify(Path.home() / 
                           'C://Users/Sarah/Dropbox/nest_models/py_output' / 
                           dirName / 
                           'ml_values.csv'))
likeFile.parent.mkdir(parents=True, exist_ok=True)
print(">> likelihood file path:", likeFile)
column_names     = np.array([
    'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 
    'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 
    'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 
    'obs_int', 'num_discovered','num_excluded', 'exception'
    ])
colnames = ', '.join([str(x) for x in column_names]) # needs to be string
# -----------------------------------------------------------------------------
#   FUNCTIONS
# -----------------------------------------------------------------------------
# Some are very small and specific (e.g. logistic function); others are 
# quite involved.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def stormGen(frq, dur):
    out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    dr = np.arange(0, dur, 1)
    stormDays = [out + x for x in dr]
    stormDays = np.array(stormDays).flatten()
    print(">> storm days:", stormDays)
    return(stormDays)

# -----------------------------------------------------------------------------
# This function remaps values from R^2 into the lower left triangle located 
# within the unit square.
def triangle(x0, y0):
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
# This is just the logistic function
# Trying out type hints (PEP 484) to keep output from overflowing
#def logistic(x)->np.float128:
def logistic(x)->np.longdouble:
    #return 1.0/( 1.0 + math.exp(-x) )
    return 1.0/( 1.0 + np.exp(-x) )

# -----------------------------------------------------------------------------
# This function creates the list of survey days by taking a random start date 
# from the first 5 breeding days and creating a range with step size determined
# by observation frequency. Then remove storm days.
def mk_surveys(stormDays, obsFreq, breedingDays):
    # first day of each week because the initiation probability is weekly 
    # the upper value should not be == to the total number of season days 
    # because then nests end after season is over 

    start       = rng.integers(1, high=5) # random day of 1st survey from 1st 5 breeding days          
    end         = start + breedingDays
    surveyDays  = np.arange(start, end, step=obsFreq)
    stormSurvey = np.isin(surveyDays, stormDays) 
    surveyDays  = surveyDays[np.isin(surveyDays, stormDays) == False] # keep only values that aren't in storm_days 
    if debug: 
        print(">> all survey days, minus storms:\n", surveyDays, len(surveyDays)) 

    return(surveyDays)

# -----------------------------------------------------------------------------
def survey_int(surveyDays):
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
#   NEST DATA COLUMNS AND PARAM LIST: 
# -----------------------------------------------------------------------------
# 0) ID number              | 4) flooded? (T/F)        | 8) j (last active)  
# 1) initiation date        | 5) surveys til discovery | 9) k (last checked) 
# 2) end date               | 6) discovered? (T/F)     | 10) assigned fate
# 3) hatched? (T/F)         | 7) i (first found)       | 11) num obs intervals 
# 12) final int storm (T/F) | 13) num other storms 

#paramsList      = list(product(
#          0           1         2          3         4          5
#        numNests, probSurv, SprobSurv, stormDur, stormFreq, hatchTime, 
#           6           7             8            9          10
#        obsFreq, probMortFlood, breedingDays, discProb, assignUnknown
#        ))
# -----------------------------------------------------------------------------
def mk_nests(params, init, nestData): 

    # 1. Unpack necessary parameters
    # NOTE about the params at the beginning of the script:
    # some have only 1 member, but they are still treated as arrays, not scalars

    hatchTime = int(params[5]) 
    obsFreq   = int(params[6]) 
    numNests  = int(params[0]) 
    pSurv     = params[1]       # daily survival probability
    fateCuesPresent = 0.6 if obsFreq > 5 else 0.66 if obsFreq == 5 else 0.75
    if debug: print(
        ">> observation frequency:", obsFreq, 
        ">> prob of correct fate:", fateCuesPresent,
        )

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

    # 5. Decide cause of failure for failed nests:
    # >> create a vector of probabilities, one for each failed nest, to decide
    #    whether nest flooded
    # >> probability value is compared to the prob of failure due to flooding
    #          on a regular day: prob of mortality = 1-DSR 
    #                            conditional prob of flooding = 0.05
    #          on a storm day: prob of mortality = 0.9 
    #                          conditional prob of flooding = 1
def mk_flood(params, numStorms):
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
    if debug: print(">>>>> true final nest fates:\n", trueFate)
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
    assignedFate = np.zeros(numNests) # if there was no storm in the final interval, correct fate is assigned 
    assignedFate.fill(7) # default is unknown; fill with known fates if field cues allow
    fateCuesProb = rng.uniform(low=0, high=1, size=numNests)
    if debug: print(">> random probs to compare to fateCuesPresent:\n", fateCuesProb, fateCuesProb.shape)
    # fateCuesProb[stormIntFinal] = 0.1
    # if debug: print(">> random probs after storms accounted for:\n", fateCuesProb, fateCuesProb.shape)
    if debug: print(">> nests with storm in final interval:", stormIntFinal)
    # assignedFate[stormIntFinal==True and fateCuesProb < 0.1] = trueFate[stormIntFinal==True and fateCuesProb < 0.1] 
    fateCuesPres = np.zeros(numNests)
    fateCuesPres.fill(fateCuesPresent)
    fateCuesPres[stormIntFinal==True] = 0.1
    if debug: print(">> fate cue present comparison", fateCuesPres)
    # assignedFate[fateCuesProb < fateCuesPresent] = trueFate[fateCuesProb < fateCuesPresent]
    assignedFate[fateCuesProb < fateCuesPres] = trueFate[fateCuesProb < fateCuesPres] 
    if debug: print(">> assigned fates:", assignedFate)
    if stormFate: assignedFate[stormIntFinal] = 2
    if debug: print(">> assigned fates after storm fates assigned:", assignedFate)
    # fate cues prob should be affecting all nest fates equally, not just failures.
    # if debug: print(">> proportion of nests assigned hatch fate:", np.sum((assignedFate==0)[discovered==True])/(sum(discovered==True)),"vs period survival:", pSurv**hatchTime)
    # print(">> assigned fate array & its shape:\n", assignedFate, assignedFate.shape)
    return(assignedFate)

# -----------------------------------------------------------------------------
# GET RID OF THIS ONE
def discover_time(discProb, numNests):
    # NOTE this name is a little misleading - it's actually survey days til discovery
    daysTilDiscovery = rng.negative_binomial(n=1, p=discProb, size=numNests) # see above for explanation of p 
    # nestData[:,5]    = daysTilDiscovery 
    if debug: print(">> survey days until discovery:\n", daysTilDiscovery, len(daysTilDiscovery)) 
    return(daysTilDiscovery)

# -----------------------------------------------------------------------------
def svy_position(initiation, nestEnd, surveyDays):

    position = np.searchsorted(surveyDays, initiation) 
    if debug: print(">> position of initiation date in survey day list:\n", position, len(position)) 
    
    position2 = np.searchsorted(surveyDays, nestEnd)
    if debug: print(">> position of end date in survey day list:\n", position2, len(position2)) 
    
    return((position, position2)) # return a tuple
    # position2
# -----------------------------------------------------------------------------
# get rid of this one ??
# I think this calculation may just be for debugging purposes
def svydays_nest(surveyDays, pos):
    # what are the first and last *possible* survey dates for the nest?  
    # pos is a tuple of (position, position2)
    # this finds the index of where initiation would be inserted in surveyDays 
    firstSurvey = surveyDays[pos[0]] 
    # NOTE last possible survey will be after nest ends (either hatch or fail) 
    lastSurvey = surveyDays[pos[1]] 

    possSurveyDates = np.stack((firstSurvey, lastSurvey)) # double parens so numpy knows it's not two separate arguments
    possSurveyDates = np.transpose(possSurveyDates)
    if debug: print("first and last possible survey:", possSurveyDates)
    # if debug:
        # print(
            #   ">> start of nest, end of nest, first possible survey, last possible survey:\n",
            #   np.concatenate((nestPeriod,possSurveyDates), axis=1)
            #   )
    totalSurvey = pos[0] - pos[1] + 1# this gives the difference in index values from surveyDays, AKA number of surveys
    return(totalSurvey)
    # number of surveys when nest is active (when it could be discovered)
    #print(">> total possible surveys to observe nest (position2 - position):\n", position2-position) 
    # print(">> total possible surveys to observe nest (position2 - position):\n", totalSurvey) 
# -----------------------------------------------------------------------------
# def discover_nests(svy_til_disc, pos):
def discover_nests(discProb, numNests, pos):
    # discovered = svy_til_disc < (position2-position) 

    ijk = np.zeros(numNests, 3)
    svysTilDiscovery = rng.negative_binomial(n=1, p=discProb, size=numNests) # see above for explanation of p 
    if debug: print(">> survey days until discovery:\n", svysTilDiscovery, len(svysTilDiscovery)) 
    num_svy    = pos[1] - pos[0] # total possible number of surveys
    # num_svy[num_svy < 0] = 0 # exclude negative numbers of surveys
    if debug: print("total possible number of surveys for this nest:", num_svy)
    discovered = svysTilDiscovery < num_svy
    ijk[:,1] = surveyDays[pos[0]+svysTilDiscovery]
    ijk[:,2] = surveyDays[pos[0]+svysTilDiscovery]
    ijk[:,3] = surveyDays[pos[0]+svysTilDiscovery]
    # daysTilDiscovery also acts as a T/F for whether nest was discovered:
    svysTilDiscovery[discovered==False] = 999
    
    if debug: print(
        ">> nest discovered if surveys til discovery < total possible surveys (surveys while active):\n", 
        discovered
        )
    if debug: print(">> proportion of nests discovered:", sum(discovered)/numNests, 
                    "vs. expected proportion:", discProb # expected isn't discProb
                    # this isn't right either, obviously...
                    # "vs. expected proportion:", sum(discProb**num_svy)
                    )
    # return(discovered)
    return(svysTilDiscovery)

# -----------------------------------------------------------------------------
# Calculate i, j, and k
# Need to exclude nests where surveys til discovery = 999 because 
# numpy still calculates these values and then excludes them
# so is discovered==True not a mask? 
# so can't use 999 bc index is too big. but needs to be the same type (integer)
def calc_ijk(numNests, discovered, svy_til_disc, pos, fates):
    ijk = np.zeros(numNests, 3)
    ijk[:,1] = 
    if debug: print("surveys til discovery:", svy_til_disc)
    # firstFound[discovered==True] = surveyDays[(pos[0]+svy_til_disc)][discovered==True] 
    # firstFound[svy_til_disc!=999] = surveyDays[(pos[0]+svy_til_disc)][svy_til_disc!=999] 
    discSurvey = pos[0] + svy_til_disc
    # I'm still not sure exactly how this line works:
    # firstFound[discovered] = surveyDays[(pos[0]+svy_til_disc)][discovered] 
    # I guess it is indexing surveyDays for ALL nests before then indexing
    # the result for only discovered nests?
    firstFound[discovered] = surveyDays[discSurvey][discovered] 
    if debug: print(">> nest first found:\n", firstFound, len(firstFound)) 

    lastActive = np.zeros(numNests)
    lastActive[discovered==True] = surveyDays[pos[1]][discovered==True] 
    if debug: print( ">> nest last active:\n", lastActive, len(lastActive)) 
    
    # Last checked will be one survey after last active, unless hatch == True
    lastChecked = np.zeros(numNests) 
    lastChecked[discovered==True] = surveyDays[(pos[1]+1)][discovered==True] 
    if debug: print(">> nest last checked, w/o hatch:\n", lastChecked, len(lastChecked)) 
    #lastChecked[hatched==True] = lastActive[hatched==True] # take hatch date into account
    lastChecked[fates==0] = lastActive[fates==0] # take hatch date into account
    if debug: print(">> nest last checked:\n", lastChecked, len(lastChecked)) 

    # nestData[:,7]  = firstFound 
    # nestData[:,8]  = lastActive 
    # nestData[:,9]  = lastChecked# 
    return((firstFound, lastActive, lastChecked))

# -----------------------------------------------------------------------------
def observer(par, fateCues, fates, surveyDays, nData, out):
    initiation = nData[:,1]
    end        = nData[:,1] + nData[:,2]
    numNests, obsFreq, discProb, stormFate = par # unpack par

    # svy_til_disc = discover_time(discProb, numNests)
    pos          = svy_position(initiation, end, surveyDays)
    # svy_possible = svydays_nest(surveyDays, pos)
    # discovered   = discover_nests(svy_til_disc, pos)
    svy_til_disc  = discover_nests(discProb, numNests, pos)
    stormIntFinal= surveyInts[pos[1]] > obsFreq  # was obs interval longer than usual? (== there was a storm)
    discovered = svy_til_disc != 999
    #totalReal  = totalSurvey - daysTilDiscovery + 1 # totalSurvey - (daysTilDiscovery - 1)
    # totalReal  = svy_possible - svy_til_disc  + 1# totalSurvey - (daysTilDiscovery - 1)
    ijk      = calc_ijk(numNests, discovered, svy_til_disc, pos, fates)
    # if debug_obs: print("totalReal:",totalReal)
    # need to decide if it's number of observations or observation intervals
    # can extract discovered T/F from svy_til_disc; don't need another column
    out[:,0] = svy_til_disc # do we even need this??
    # out[:,1] = discovered.astype(int) # convert to numeric for numpy array  
    # out[:,1] = discovered.astype(int) # convert to numeric for numpy array  
    # could probably get this from ijk?
    # out[:,1] = totalReal # total number of observation intervals
    out[:,2] = ijk[0]
    out[:,3] = ijk[1]
    out[:,4] = ijk[2]
    out[:,5] = assign_fate(fateCues, fates, numNests, stormIntFinal, stormFate)
    return(out)

# -----------------------------------------------------------------------------
def obsStorm(pos, fate, out):
    # ---- STORMS DURING OBSERVATION PERIOD -------------------------------------------------------------------------

    # obsPeriod      = lastChecked - firstFound 
    # print(">> length of observation period:\n", obsPeriod, len(obsPeriod)) # is the actual number of days in the period greater than expectedsPeriod) 
    # numStorms = (obsPeriod/obsFreq - totalReal) / obsFreq # number extra days in period divided by normal interval length 
    # NOTE already calculated numStorms earlier in function
    # print(">> number of storm intervals during nest active:", numStorms )
    # I changed how surveyInts was defined; now it's current minus previous, not next minus current
    # stormInterval = surveyInts > obsFreq
    #@#print(">> was there a storm in this interval?\n", stormInterval)
    stormIntFinal = surveyInts[pos[1]] > obsFreq  # was obs interval longer than usual? (== there was a storm)
    #@#print(">> was there a storm in the final interval for a given nest?\n", stormIntFinal, len(stormIntFinal)) # basically ask is final interval > 3 
    # needs to be positionEnd -1 because of how intervals are calculated 
    out[:,0] = stormIntFinal.astype(int) #nestData[:,13] = numStorms - as.integer(stormFinal) 
    #@#print(">> stormIntFinal in integer form:\n", nestData[:,12], stormIntFinal.shape)
    # out[:,1] = nstorm 


    #@#print("days from end date to final check:", lastChecked - nestEnd)
    #y = 3
    #assignedFate[lastChecked-nestEnd < y] = trueFate[lastChecked-nestEnd < y]


    # >>>>>>>>>>> SAVE NEST DATA AS CSV (may be more memory-intensive) >>>>>>>>>>>>>>
    #now       = datetime.now().strftime("%H%M%S")
    #nests  = Path.home()/'/mnt/c/Users/sarah/Dropbox/nest_models/py_output'/dirName/('nests'+now+'.csv')
    #nests.parent.mkdir(parents=True, exist_ok=True)

    #csvHead = "ID, initiation, end, hatched, flooded, surveys_til_discovery, discovered,\
    #           i, j, k, a_fate, num_obs, storm_during_final, num_storms" 

               #num_stm_survey_excl_final 
    #np.savetxt("nestdata.csv", nestData, fmt="%d", delimiter=",", header=csvHead) 
    #np.savetxt(nests, nestData, fmt="%d", delimiter=",", header=csvHead) 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    return(nestData) 

# -----------------------------------------------------------------------------
#   MAYFIELD & JOHNSON
# -----------------------------------------------------------------------------
# Mayfield's original estimator was defined as: 
#                DSR = 1 - (# failed nests / # exposure days)
# so if DSR = 1 - daily mortality, then:
# daily mortality = # failed nests / # exposure days

# NOTE: Johnson (1979) provided a mathematical derivation that allowed the 
#       calculation of variance for the estimate.
# > for a single day:
#    > probability of survival is s    
#    > probability of failure is (1-s)
# > for interval of length k days:
#    > prob of survival is s**k 
#    > prob of failure is s**(1/2k-1)(1-s)
# > ex. - prob of a nest surviving three days and failing on the fourth is:
#         s*s*s*(1-s) 
#     > this assumes that a failed nest survived half (minus a day)
#       of interval and then failed
# Johnson's rewriting of the Mayfield estimator:
#        mortality = (f1 + sum(ft)) / (h1 + sum(t*ht) + f1 + 0.5*sum(t*ft)) 
# > created by differentiating the log-likelihood equation and setting to 
#   zero (maximizing)
# > ht = hatched or survived til next visit; ft = failed by next visit
# > f1 and h1 represent an interval between visits of one day, which is not 
#   used in our studies
#   > so we end up with: sum(ft) / (sum(t*ht) + 0.5*sum(t*ft)) 
#          where t = interval length, and 
#          f and h represent number of failures and hatches, respectively
# Johnson's Mayfield-40 estimator: 
#        mortality = sum(ft) / (sum(t*ht) + 0.4*sum(t*ft))
# Johnson's modified ML estimator:
#        1/s*(sum(t*ht)) = sum( (t * ft * s^t-1) / (1 - s^t))
# -----------------------------------------------------------------------------
# Calculate the exposure period for a nest (number of days observed)
#   > the ijk values should tell you failed vs hatched
#   > I think I couldn't get it to work as vectorized, so I used a loop
# For the basic case where psurv is constant across all nests and times:
#   1. count the total number of alive days
#   2. count the number of days in the final interval (for failed nests)
#   3. calculate the exposure
#       > # days prior to final interval + # days in final interval * expPercent
#            > expPercent = percent of final interval nest is assumed alive
#            > Mayfield used 50%, Johnson corrected it to 40%
#            > final interval assumed to be zero days for hatched nests, which
#              were found after hatching (exposure of incubation period is over)
#       > not calculating nestling exposure bc precocial/semi-precocial chicks
#         leave the nest so early 

def exposure(inp, numNests, expPercent): # inp = ijk (cols) for all nests (rows)
    exp = np.zeros(numNests, 3)
    for n in range(len(inp)-1): # want n to be the row NUMBER
        alive_days = inp[n,2] - inp[n,1] # interval from first found to last active 
                                         # all nests are KNOWN to be alive
        alive_days = alive_days - 1 # since this is essentially 1-day intervals, 
                                    # need 1 fewer than total number
        final_int  = inp[n,3] - inp[n,2] 
        final_int  = final_int - 1
        exposure   = alive_days + (final_int*expPercent)
        if debug: print(
            "days nest was alive:", alive_days,
            "& final int:", final_int, 
            "& exposure:", exposure
            )
        # NOTE need nests to be alive for at least one interval
    exp[:,0] = alive_days
    exp[:,1] = final_int
    exp[:,2] = exposure# if hatch, final_int=0 so won't affect outcome
    return(exp)


# -----------------------------------------------------------------------------
def mayfield(ndata, exp):
#    I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
#    dat = ndata[
    # hatched = np.sum(ndata[:,3])
    # failed = len(ndata) - hatched
    exposure = exp[2]
    hatch  = ndata[:,3]
    fail   = ~hatch 
    # failExp = sum()
    # if debug: print(">> exposure percentage for final interval:", expPercent)
    # if debug: print(">> hatch:\n",hatch,"\n>> and fail:\n",fail) 
    # if debug: print(">>>> Calculate Mayfield estimator.")
    # if debug: print(">> number of nests hatched:", hatched, "and failed", failed)
    # mayf = failed / (hatched + 0.5*failed)
    # mayf = failed / (hatched + (expPercent*failed))
    mayf = fail / (exposure.sum())
    if debug: print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

    return(mayf)

# def johnson(ndata):

# -----------------------------------------------------------------------------

def fates_dsr(assignedFate, trueFate):
    aFates = [np.sum((assignedFate == x)[discovered==True]) for x in range(4)]
    # this proportion needs to be out of nests discovered AND assigned
    aFatesProp = [np.sum((assignedFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
    tFates = [np.sum((trueFate == x)[discovered==True]) for x in range(4)]
    tFatesProp = [np.sum((trueFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
    
    if debug:
        print(
            "> ALL NESTS - hatched:",
            hatched.sum(),
            "; failed:", 
            numNests - hatched.sum(), 
            "; exposure days:", 
            exp
            # "; true DSR:",
            # trueDSR
            )
    # print(
    #         ">> assigned fate (hatched, depredated, flooded, unknown):", 
    #         # ">> assigned fate proportions (hatched, depredated, flooded, unknown):", 
    #         aFatesProp[0:3], 
    #         (np.sum(discovered==True) - np.sum(aFates)) / (np.sum(discovered==True)),
    #         # (np.sum(aFates==7) )/ (np.sum(discovered==True)),
    #         "\n\n>> proportions of known (assigned) fates (H, D, F):",
    #         aFates[0:3]/np.sum(aFates),
    #         "\n>> vs. actual proportions for discovered only (H, D, F):",
    #         tFatesProp[0:3],
    #         # np.sum(discovered==True)- np.sum(tFates)
    #         )
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
# Probabilities for Program MARK:
# 1. create vectors to store:
#      a) the probability values for each nest
#      b) the degrees of freedom for each value
# 2. fill the vectors
#      > take the for loop out of the function that is optimized
# Note that failed nests have final_int>0 while hatched nests have final_int=0
#
# Probability equation: 
#   > daily probability of survival (DSR) raised to the power of intervals 
#     nest was known to be alive
#   > for hatched nests, that's it
#   > for failed nests, exact failure date allowed to be unknown
#       > but we know nest wasn't alive for the entire final interval
#       > so add in probability of NOT surviving one interval (1-DSR)
#   > EX: if probability of surviving from day 1-3 is s1*s2*s3, then
#   >     probability of failure sometime during days 4-6 is 1-s4*s5*s6
#   > hatched nests also have one extra degree of freedom (dof)

def mark_probs(s, exp, ndata):
    # exp[hatched==True] = nDays[hatched==True] # do I need the ==True?
    allp   = np.array(range(1,len(ndata)), 
                      dtype=np.longdouble) # all nest probabilities 
    alldof = np.array(range(1,len(ndata)), 
                      dtype=np.double) # all degrees of freedom
    for n in range(len(ndata)-1): # want n to be the row NUMBER
        alive_days = exp[n,0]
        final_int  = exp[n,1]
        exposure   = exp[n,2]
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
# Run the Program MARK algorithm
# 1. Grab the data for the input for MARK 
#       > First, grab only discovered nests
#       > Then, only the needed columns
#         (nest ID, first found, last active, last checked, assigned fate)
#          inp[0] = ID | inp[1] = i | inp[2] = j | inp[3] = k | inp[4] = fate `
# 2. Extract rows where j minus i does not equal zero (nest wasn't only 
#    observed as active for one day)
#       > Model requires all nests to have at least two observations while active

def prog_mark(s, ndata, probs, nocc):
    prob, dof = probs
    # print("all nests:", len(ndata)) 
    disc = ndata[np.where(ndata[:,6]==1)] # discovered==True
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
    alldof = dof[inoInd]
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
# This function calls the program MARK function when given a random starting 
# value (srn) and some nest data (ndata)
#   > values given to optimizer are transformed then passed to MARK function
#      > allows larger range of values for optimizer to work over w/o overflow
#      > but values given to the function are still between 0 and 1, as required
#   > Create vector to store the log-transformed values, then fill

def mark_wrapper(srn, ndata):
    s = np.ones(numNests, dtype=np.longdouble)
    s = logistic(srn)
    # NOTE is this multiple random starting values (for each nest) or one random starting value?
    #@#print("logistic of random starting value for program MARK:", s, s.dtype)
    # the logistic function tends to overflow if it's a normal float; make it np.float128
    ret = prog_mark(s, ndata)
    #@#print("ret=", ret)
    return ret

# -----------------------------------------------------------------------------
#   THE LIKELIHOOD FUNCTION
# -----------------------------------------------------------------------------

# This function computes the overall likelyhood of the data given the model parameter estimates.
#
# The model parameters are expected to be received in the following order:
# - a_s   = probability of survival during non-storm days
# - a_mp  = conditional probability of predation given failure during non-storm days
# - a_mf  = conditional probability of flooding given failure during non-storm days
# - a_ss  = probability of survival during storm days
# - a_mfs = conditional probability of predation given failure during storm days
# - a_mps = conxditional probability of predation given failure during storm days
# - sM    = for program MARK?

#def like_old(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData, stormDays, surveyDays, obs_int):
def like_old(argL, obsFreq, nestData, surveyDays, stormDays ):
  
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
# try to keep these in numpy:
def like(perfectInfo, hatchTime, argL, numNests, obsFreq, nestData, surveyDays):
    # perfectInfo == 0 or 1 to tell you whether you know all nest fates or not
# NEST DATA COLUMNS: 
# 0) ID number              | 4) flooded? (T/F)        | 8) j (last active)  
# 1) initiation date        | 5) surveys til discovery | 9) k (last checked) 
# 2) end date               | 6) discovered? (T/F)     | 10) assigned fate
# 3) hatched? (T/F)         | 7) i (first found)       | 11) num obs intervals 
# 12) final int storm (T/F) | 13) num other storms 
    # ---------------------------------------------------------------------------------------------------
    # 1. Unpack:
    #    a. Initial values for optimizer:
    a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL

    # b. Observation history values from nest data:
    #@#print(">> nest data going into likelihood function:\n", nestData, nestData.shape)
    # NOTE NOTE should this be the assigned fate or true fate?
    flooded = nestData[:,4]
    hatched = nestData[:,3] # actual hatched nests (after storms accounted for)
    ff      = nestData[:,7]
    la      = nestData[:,8]
    lc      = nestData[:,9]
    numInt  = nestData[:,11]
    # print("number of obs intervals:", numInt)

    # obsDays  = int(la - ff) # doesn't work bc it's an array
    obsDays  = la - ff 
    finalInt = lc - la
    
    #ff, la, lc = nestData[:,8:10] # doesn't work bc they aren't lists
    # print(
    #     "i, j, k:", ff, la, lc, 
    #     # "observation days:", obsDays, type(obsDays),
    #     "final interval length:", finalInt)
    # print(">>>>> likelihood: hatched:", hatched.sum())
    # if perfectInfo == 0:
        # numIntTotal = nestData[:,11]
        #print(">> number of observation intervals for each nest:\n", numIntTotal)
    # else:
        # numIntTotal = hatchTime # what was this for???
        # numIntTotal = 
        #print(">> number of observation intervals for each nest, frequency = 1 day:\n", numIntTotal)
    # NOTE this SHOULD be number of obs (I think) but it's giving number of 
    #      obs intervals, so I'll roll with it

    # ---------------------------------------------------------------------------------------------------
    # 2. Initialize the overall likelihood counter; Decimal gives more precision
    logLike = Decimal(0.0)         
    # ---------------------------------------------------------------------------------------------------
    # 3. Define state vectors (1x3 matrices) - all the possible nest states 
    #       [1 0 0] (alive)   [0 1 0] (fail-predation)   [0 0 1] (fail-flood) 
    stillAlive = np.array([1,0,0]) 
    mortFlood  = np.array([0,1,0])
    mortPred   = np.array([0,0,1])
    # ---------------------------------------------------------------------------------------------------
    # 4. Create arrays to hold state vectors for all nests:
    #    a. state of nest on date nest was first found (stateFF)
    #    b. state of nest on date nest was last checked (stateLC) - this is the fate as observed
    # Could also calculate bassed on nest fate value
    # FOR THE INITIAL STATE (stateFF), just one vector (see notebook) - later in code
    stateEnd    = np.empty((numNests, 3))     # state at end of normal interval
    stateLC     = np.empty((numNests, 3)) 
     # > use broadcasting - fill doesn't work with arrays as the fill value:
    stateEnd[:] = stillAlive
    stateLC[:]  = mortPred   # default is still depredation

    stateLC[flooded==True] = mortFlood  # flooded status gets flooded state vector
    stateLC[hatched==True] = stillAlive # hatched nests stay alive the entire time

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
    trMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
    # print(">> transition matrix\n:", trMatrix, trMatrix.shape)
    TstateI = np.transpose(stillAlive) 
    ##print(">> transpose of initial state vector:", TstateI, TstateI.shape)
    #                                             this will depend on how many storms/when they are
    # NOTE need the transpose of each row, not entire matrix. 
    pwr = np.linalg.matrix_power(trMatrix, obsFreq) # raise the matrix to the power of the number of days in obs int
    pwrStm = np.linalg.matrix_power(trMatrix, obsFreq*2) # power equation for storm intervals (longer obs int)
    # print(">> transition matrix raised to the obs int power:\n", pwr, pwr.shape)

    normalInt = stateEnd@pwr@TstateI
    # print(
    #     ">> likelihood of one normal interval:\n",
    #     normalInt,
    #     normalInt.shape,
    #     normalInt.dtype)

    # The final interval is one of these two (ends in final state):
    normalFinal = stateLC@pwr@TstateI
    # print("final interval:", normalFinal, "and -log likelihood:", -np.log(normalFinal))
    stormFinal  = stateLC@pwrStm@TstateI

    logLik   = np.ones(numNests, dtype=np.longdouble) # this should give it enough precision & avoid errors
    logLik      = logLik * np.log(normalInt) * -1 # dtype changes to float64 unless you multiply it by itself
    # print("-log likelihood of 1 interval:", logLik)
    
    # logLikFin    = np.ones(numNests, dtype=np.longdouble)
    logLikFin = np.empty(numNests, dtype=np.longdouble)
    # logLikFin.fill(-np.log(normalFinal))
    logLikFin= -np.log(normalFinal)
    logLikFin[hatched == True] = 0
    logLikFin[flooded == True] = -np.log(stormFinal[flooded==True])
    # logLikFin[]    = logLikFin * (-np.log(normalFinal))
    # logLikFinStm = logLikFinStm * (-np.log(stormFinal))
    # print(">> log likelihood final interval, updated with storms/hatch:\n", logLikFin)

    stormDuringFin = nestData[:,12] # was there a storm during the final interval?
    logLikelihood  = (logLik*numInt) + (logLikFin)
    # print("log likelihood of each nest history:", logLikelihood)

    # for x in range(numNests):
        # print(">> likelihood equation: (",logLik[x],"*",numIntNorm[x],")+(",logLikStm[x],"*",numIntStm[x],")+(",logLikFin[x],"**(1 -",stormDuringFin[x],")+(", logLikFinStm[x],"**",stormDuringFin[x])
    #    print(
    #        f">> likelihood equation for nest {x}: " 
    #     #    f"{numInt[x]:.0f} * {logLik[x]:.5f} + "
    #        f"({numInt[x]:.0f} * {logLik[x]:.5f}) + {logLikFin[x]:.5f} ="
    #     #    f"{logLikFinStm[x]:.5f} * (1-{stormDuringFin[x]:.0f}) + " 
    #     #    f"{logLikFinStm[x]:.2f} * {stormDuringFin[x]:.2f} = "
    #        f"{logLikelihood[x]:.2f}")


    logLike        = np.sum(logLikelihood)
    # print("overall log likelihood:", logLike)
    return(logLike)

# this function does the matrix multiplication for a SINGLE interval of length intElt days 
 # during observaton, nest state is assessed on each visit to form an observation history 
 # the function calculates the negative log likelihood of one interval from the observation history 
 # these can then be multiplied together to get the overall likelihood of the observation history 

# -----------------------------------------------------------------------------
#   THE LIKELIHOOD WRAPPER FUNCTION
# -----------------------------------------------------------------------------
# The values are log-transformed before running them thru the likelihood 
# function, so the values given to the optimizer are the untransformed values,
# meaning the optimizer output will also be untransformed.
# >> Therefore, need to transform the output as well.
def like_smd( 
        x, perfectInfo, hatchTime, nestData, obsFreq, 
        stormDays, surveyDays, whichRet):

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

    numNests = nestData.shape[0]
    #@#print(">> number of nests:", numNests)

    # call the likelihood function:
    argL = np.array([s2,mp2,mf2,ss2,mps2,mfs2, sM])
    #ret = like(argL, ndata, obs, storm, survey)
    #ret = like(argL, nestData, obsFreq, stormDays, surveyDays)
    if whichRet == 1:
        ret = like(perfectInfo, hatchTime, argL, numNests, 
                   obsFreq,  nestData, surveyDays)
        # print('like_smd(): Msg : ret = ', ret)
    
    # else:
    elif whichRet == 2:
        ret = like_old(argL, obsFreq, nestData, surveyDays, stormDays)
        # print('like_smd(): Msg : using old function; ret = ', ret)
    
    else:
        print(' argument whichRet is invalid ')
    
    # rets = np.array([ret, ret2])

    return(ret)

# ---------------------------------------------------------------------------------------
def ansTransform(ans):
        
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
    # Choose random initial values for the optimizer
    # These will be log-transformed before going through the likelihood function
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

#def run_params(paramsList, dirName):
# NOTE need to create storms after params are chosen
# NOTE is there even a reason to have this in a function?

with open(likeFile, "wb") as f:
# with open(likeFile, "ab") as f: # changing this to append didn't help...
    # append shouldn't matter if the file is just open the whole time

    parID     = 0

    for i in range(0, len(paramsList)): # for each set of params
        #par = paramsList[i]
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(
            "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ "
            "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ")
        print(">>>>>>>>> param set number:", parID)
        # paramsArray is a 2d array; loop through the rows 
        # each row is a set of parameters we are trying
        par        = paramsArray[i] 
        # in an array, all values have the same numpy dtype (float in this case) 
        # after selecting the row, unpack the params & change dtype as needed:
        numNests   = par[0].astype(int)
        pSurv      = par[1]
        pSurvStorm = par[2]
        freq       = par[4].astype(int)
        dur        = par[3].astype(int)
        hatchTime  = par[5].astype(int)
        obsFreq    = par[6].astype(int)
        pMFlood    = par[7]
        brDays     = par[8]
        pDisc      = par[9]
        stormF     = par[10]
        fateCues   = 0.6 if obsFreq > 5 else 0.66 if obsFreq == 5 else 0.75
        # fateCues   = par[10]
        #survival   = par[15]
        # Generate random list of storm days based on real weekly probabilities
        #  > Then create a list of survey days 
        # NOTE do you want new storms/survey days for each replicate 
        #      or each parameter set?
        stormDays  = stormGen(freq, dur)
        surveyDays = mk_surveys(stormDays, obsFreq, brDays)
        surveyInts = survey_int(surveyDays)
        repID      = 0  # keep trackof replicates
        numOut     = 21 # number of output params

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
                # empty arrays to store data for this replicate:
                likeVal =  np.zeros(shape=(numOut), dtype=np.longdouble)
                nd = np.zeros(shape=(numNests, 3), dtype=int)
                nd2 = np.zeros(shape=(numNests, 7), dtype=int)
                # -------------------------------------------------------------
                # make the nests:
                nData          = mk_nests(par, initProb, nd)
                nestPeriod     = mk_per(nData[:,1], (nData[:,1]+nData[:,2]))
                stormsPerNest  = storm_nest(nestPeriod, stormDays)
                flooded        = mk_flood(par, stormsPerNest)
                hatched        = nData[:,2] >= hatchTime
                nestFate       = mk_fates(numNests, hatched, flooded)
                # -------------------------------------------------------------
                # observer:
                par2           = [numNests, obsFreq, pDisc, stormF]
                obs            = observer(par2, fateCues, nestFate, surveyDays, nData, nd2)
                # -------------------------------------------------------------
                # concatenate to make data for the nest models:
                #   0. nest ID                  6. # obs int                
                #   1. initiation               7. first found       
                #   2. survival (w/o storms)    8. last active              
                #   3. fate                     9. last checked   
                #   4. num storms               10. assigned fate
                #   5. survey til disc
                #   
                nestData = np.concatenate((nData, nestFate[:,None], obs), axis=1)
                # not sure where indexError will show up again
                # try: # create nest/observer data
                #     # np.save(n, nestData) # make sure this is correct kind of save
                # except IndexError as error:
                #     print(
                #         ">> !! IndexError in nest data:", 
                #         error,
                #         ". Go to next replicate.")
                #     continue
                
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # hatched = nestData[:,3] # true hatched (storms accounted for)
                #hatchProp = sum(hatched)/numNests
                # expDays = nestData[0,14]
                # expDays = exposure()
                # trueDSR = 1 - ((numNests - hatched.sum()) / expDays)
                ###########################################################
                # Keep only discovered nests, then count them:
                # nestData   = nestData[np.where(nestData[:,6]==1)] 
                nestData   = nestData[np.where(nestData[:,5]!=999)] 
                discovered = nestData.shape[0]   

                if debug: print(">> proportion of nests assigned hatch fate:", 
                                np.sum((nestData[:,3]==0)[discovered==True])/(sum(discovered==True)),
                                "vs period survival:", 
                                pSurv**hatchTime)


                if debug:
                    print(
                        ">> nests w/ only 1 obs while active:",
                        np.where(nestData[:,7] == nestData[:,8]),
                        "& nests w/ unknown fate:",
                        np.where(nestData[:,10] == 7)
                        ) 
                exclude  = ((nestData[:,10] == 7) | 
                            (nestData[:,7]==nestData[:,8]))                         
                # if there's only one observation, firstFound will == lastActive
                excluded = sum(exclude) # exclude = boolean array; sum = num True
                analyzed = discovered - excluded
                failed   = nestData.shape[0] - sum(nestData[:,3]) # num nests - num hatched
                expDisc  = sum(nestData[:,15])
                trueDSR_disc = 1 - ((nestData.shape[0] - sum(nestData[:,3])) / 
                                    sum(nestData[:,15]))
                
                nestData    = nestData[~(exclude),:]    # remove excluded nests 
            
                trueDSR_analysis = 1 - ((nestData.shape[0] - sum(nestData[:,3])) / 
                                          sum(nestData[:,15]) )

                if debug:
                    print(
                        "> DISCOVERED NESTS - total | analyzed: hatched:", 
                        discovered, "|", analyzed,
                        # "excluded from analysis:", excluded,
                        "failed:", failed, "|",
                        nestData.shape[0] - sum(nestData[:,3]),
                        "exposure days:", expDisc, "|", sum(nestData[:,15])
                        # "true DSR:", trueDSR_disc, "|", trueDSR_analysis
                        )
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
                res = ansTransform(ans)
                if debug:
                    print(
                        "\n>> assigned DSR:",
                        pSurv,
                        "true DSR of all nests:", 
                        trueDSR, 
                        "discovered nests:",
                        trueDSR_disc,
                        "and nests used in analysis:", 
                        trueDSR_analysis
                        )

                #OPTIMIZER: MARK function
                # inp = disc[:,np.r_[0,7:11]] # doesn't include index 11
                markProb = mark_probs(srand, nData)
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
                        trueDSR,trueDSR_analysis,pSurv, pSurvStorm,pMFlood,
                        # hatchTime,numNests,obsFreq,discovered,excluded,ex
                        hatchTime,numNests,obsFreq,discovered,excluded
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
                    #np.savetxt(f, like_val, delimiter=",")
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
            

    # sudo vim -o file1 file2 [open 2 files] 
    # BLAH
    # NOTE 5/16/25 - The percent bias responds more like I would expect when I use
    #                the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
    #                BUT I still don't know why the calculated DSR is consistently low.

    from datetime import datetime
    import decimal
    from decimal import Decimal
    from itertools import product
    import itertools
    import numpy as np 
    from os.path import exists
    import os
    from pathlib import Path
    from scipy import optimize