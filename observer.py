#!/usr/local/bin/python
import numpy as np
import pprint
import pandas as pd
from itertools import groupby
from operator import itemgetter
import plotext as plt # plot ASCII plots in the terminal window
from makeNests import mk_nests, mk_fates, mk_flood, storm_nest
# from rsettings import config, rng, initFromFile
from helpers import expDecay, searchSorted2,print
from print_func import  arrPrint, print_prop, dfPrint
np.set_printoptions(precision=3, linewidth=120)

# def mk_surveys(stormDays, obsFreq, breedingDays, conf, complicate=False):
def mk_surveys(stormDays, obsFreq, breedingDays, conf, complicate=True, db=1):
  """
    This function creates the list of survey days by taking a random start date 
    from the first 5 breeding days and creating a range with step size determined
    by observation frequency. Then remove storm days.

    If complicate=True, surveys restart 2 days after storm finishes
    If False, storm days are simply excluded 
    
    surveyInts = interval between survey days. surveyInts[0]=0

    -------
    RETURNS:
      A tuple of surveyDays & surveyInts

    -----
    NOTES:
      no surveys should be <2 days after a storm day
  """
  # +> first day of each week because the initiation probability is weekly 
  # +> the upper value should not be == to the total number of season days 
  # +> because then nests end after season is over 

  # NOTE this function runs once per replicate; for loops shouldn't matter much

  stormDays   = np.sort(stormDays)
  # start       = rng.integers(1,high=5) # +> random 1st svy from 1st 5 br days      
  start       = 1
  end         = start + breedingDays
  surveyDays  = np.arange(start, end, step=obsFreq)
  stormSurvey = np.isin(surveyDays, stormDays) 
  if conf.debugObs>=5:
    print(f"\t\t>> before adding storms: {len(surveyDays)=} ; \n{surveyDays=}")
    print(f"\t\t{stormSurvey=}")
  if(complicate):
    splits      = np.where(np.diff(stormDays)!=1)[0] +1 # print(f"{splits=}")
    storms      = np.split(stormDays, splits)
    if conf.debugObs>=4: print(f"\t\t{storms=}")
    if len(stormDays) > 0: #+> only if there are storms
      for s in storms:
        lastDay = np.max(s)
        stormPos    = np.searchsorted(surveyDays, lastDay)
        # if conf.debugObs>=3: print(f"\t\t{lastDay=} ; {stormPos=}")
        #+> left and right side give same result:
        # stormPosR    = np.searchsorted(surveyDays, lastDay, side="right")
        sDiff   =  surveyDays[stormPos] - lastDay #sDiff   = lastDay+2 - surveyDays[stormPos]
        # print(f"{lastDay=} ; {stormPosL=} ; {stormPosR=} ; {sDiff=}")
        # if conf.debugObs>=4: print(f"\t\t{lastDay=} ; {stormPos=} ; {sDiff=}")
        # if surveyDays[stormPos] < lastDay + 2:
        if sDiff < 2:
          # mask = surveyDays >= surveyDays[stormPos-1]
          # mask = surveyDays >= lastDay
          # surveyDays[surveyDays >= lastDay] += 2
          # print("add more time after storm")
          # if conf.debugObs>=4: print(f"\t\t{surveyDays[stormPos]=}")
          # surveyDays[surveyDays >= lastDay] += sDiff
          surveyDays[surveyDays >= lastDay] += 2
          # if conf.debugObs>=4:
            # print(f"\t\t{surveyDays[stormPos]=}; all survey days:")
            # arrPrint(surveyDays)
        # elif sDiff >= 2:
          # if conf.debugObs>=3: print(f"\t\t{surveyDays[stormPos-1]=}")
          # surveyDays[stormPos-1] = lastDay +2
          # if conf.debugObs>=3: print(f"\t\t{surveyDays[stormPos]=}")
          # surveyDays[stormPos] = lastDay +2
          # if conf.debugObs>=4:
            # print(f"\t\t{surveyDays[stormPos]=}; all survey days:")
            # arrPrint(surveyDays)
        # np.which()
        # if conf.debugObs>=4: print(f"\t\t\t{stormPos=}")

  # if conf.debugObs>=3: arrPrint(surveyDays)
  surveyDays  = surveyDays[np.isin(surveyDays, stormDays) == False]
  # surveyDays  = [s + 2 if s > x ]
  # survey interval for first obs is 0:
  # if conf.debugObs>=2: print("\t\t|>make survey intervals")
  # surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
  # surveyInts  = np.array( [surveyDays[n+1] - surveyDays[n] for n in range(0, len(surveyDays)-1) ] + [0])
  # surveyInts  = np.array( [surveyDays[n+1] - surveyDays[n] for n in range(0, len(surveyDays)-1) ])
  surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)) ])
  # surveyInts  = np.array([1] + [surveyDays[n] - surveyDays[n-1] for n in range(2, len(surveyDays)) ] )
  # if conf.debug: 
  if db>=1:
    print(f"\n\t>-> all survey days, minus storms (len {len(surveyDays)}):")
    # indPrint(surveyDays) 
    arrPrint(surveyDays, abbr=False) 
  if db>=2:
    print(f"\t\t{surveyInts=}{len(surveyInts)}", end=" ")
    print(f"\t\t\tsurvey ints > obs int: {np.sum(surveyInts>obsFreq)}")
  # print("\n")
  # surveyInts  = np.append(surveyInts, )

  return(surveyDays, surveyInts, stormSurvey)

def mk_per(start, end, con):

  nestPeriod = np.stack((start, end)) # +> create array of tuples
  # NOTE need the double parentheses so it knows output is tuples
  nestPeriod = np.transpose(nestPeriod) # +> an array of start,end pairs 
  return(nestPeriod)

# @profile
# def assign_fate(assignVal, pWrong, trueFate, numNests, obsFr, intFinal, stormFate, cn):
#+> print
# def assign_fate(assignVal, decRate, trueFate, numNests, obsFr, intFinal, stormFate, cn):
# def assign_fate( par, trueFate,  intFinal, cn):
def assign_fate( par, rng, stormFin,trueFate,  intFinal, cn):
  """
  Observer assigns correct or incorrect fate based on some conditions:
    The observer assigns the correct fate based on a comparison of a set 
    probability to random draws from a uniform distribution. If observer is 
    incorrect, then they assign a fate of unknown unless stormFate==True, in
    which case all nests that ended in a period that contained a storm are 
    assumed to have failed due to the storm.

    Arguments: 
    assignVal = the value given for incorrect fates
  
    fateCuesProb=random values to compare
    fateCuesPres=probability of fate cues being present
      >-> created within the function using exp decay & final int length
    if fate percentages are fixed, fateCuesPres should be the same (=1) for all
  
    Returns: vector w/ assigned fate for each nest
  """
  
  # assignedFate = np.zeros(numNests) # if there was no storm in the final interval, correct fate is assigned 
  # rng = np.random.default_rng(seed=cn.rngSeed)
  # if cn.debug>=3: print(f"{stormFin=}")
  assignedFate=np.empty(par.numNests)
  assignedFate.fill(7) # +> default=unk; fill w/ known fate if field cues allow
  fateCuesPresent   = expDecay(n0=1, k=par.decayRate, t=intFinal)
  fateProb = rng.uniform(low=0, high=1, size=par.numNests)
  assignedFate[fateProb < fateCuesPresent] = trueFate[fateProb < fateCuesPresent] 

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.debugObs>=5:
    # print(f"\t\t\tCORRECT FATE: {np.where(fateProb<fateCuesPresent)=}")
    # print(f"\t\t\t{intFinal=} {par.obsFreq=}")
    print(f"\t\t\tCORRECT FATE: np.where( {fateProb=} < {fateCuesPresent=}\n)"
          f"\t\t\t\t{(np.sum(fateProb<fateCuesPresent))=}")
    print(f"\t\t\t {assignedFate.shape=} {assignedFate=}")
  if cn.debugObs>=3:
    # print(f"\t\t\t\t{np.where(fateProb<fateCuesPresent)=} {len(np.where(fateProb<fateCuesPresent))=}")
    print(f"\t\t\t\t{np.where(fateProb<fateCuesPresent)=}")
  if cn.debugObs>=2:
    print(f"\n\t\t{np.sum(fateProb<fateCuesPresent)=} | "
          f"\n\t\t{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=} |"
          f" {np.sum(assignedFate==0)=}")
  if cn.debugObs>=4:
    # print(f"{stormFin=}")
    print(f"\t\t\t{intFinal=} {len(intFinal)} ")#"\n\tcorrect={np.sum(intFinal<=par.obsFreq)=}")
    print(f"\t\t\t{fateCuesPresent=}")
    print(f"\t\t\t{fateProb=}")
    print(f"\t\t\t{fateProb<fateCuesPresent=}")
    print(f"\t\t\t{assignedFate=} ;\n\t\t{assignedFate.shape=}")
    arrPrint(assignedFate)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  if par.propMC>0 or par.propUnk>0:
    assignedFate = add_misclass(par,rng,trueFate,assignedFate,cn,db=cn.debugObs)
    #-~~
    # if cn.debugObs>=2: print(f"\t>> mis-assigning fate based on proportions")
  # if par.MCtype == "none":

  else:
    if par.stormFate: assignedFate[stormFin] = 2
    #-~~
    # if cn.debugObs>=2: print(f"\t>> mis-assigning fate based on storm activity")
    # if cn.debugObs>=2: print(f"\t>> mis-assigning fate based on storm activity & adding small num misclassified")
    # if cn.debugObs>=4: print(f"\t\t{par.obsFreq=} | {intFinal=}")

    # if par.stormFate: assignedFate[intFinal > par.obsFreq] = 2
    # if par.stormFate: assignedFate[flooded] = 2
    # assignedFate[intFinal > par.obsFreq] = 2 if par.stormFate else 7
    # assignedFate = add
    # assignedFate = add_misclass(par,rng,trueFate,assignedFate,cn,db=cn.debugObs)

    # if cn.debugObs>=3: print(f"\t\t\tSTORM NESTS: {np.where(intFinal > par.obsFreq)=}")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if cn.debugObs>=4:
    #   print(f"\t\t\tSTORM NESTS: {np.where(stormFin)=}")
    #   print(f"\t\t\t{assignedFate=} ; {assignedFate.shape=}")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # NOTE fate cues prob should affect all nest fates equally, not just failures
    #-~~
  if cn.debugObs>=2:
    print(f"\t\t{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=} | {np.sum(assignedFate==0)=}")
    # print(f"\t\t\t {assignedFate.shape=}")
  if cn.debugObs>=3: arrPrint(assignedFate)
  return(assignedFate)

#-----------------------------------------------------------------------------

# def add_misclass(par,assignedFate, propReplace):
# def add_misclass(par,nData,db=0):
# def add_misclass(par,trueFate,assignedFate,db=0):
def add_misclass(par,rng,trueFate,assignedFate,cn,db=0):
  #+> has to be discovered nests!
  # nReplace = propReplace * len(assignedFate)
  # assignedFate = nData[:,7]
  # cond = [
  #     (par.MCtype == "hatch2fail"),
  #     (par.MCtype == "fail2hatch"),
  #     (par.MCtype == "unknown"),
  #     ]
  # vals = [0,[1,2],]
  #NOTE don't reset seed each time; pass from main script as arg
  # rng = np.random.default_rng(seed=cn.rngSeed)
  val   = [0,9,11,112,13,1131,1211] if par.MCtype == "hatch2fail" else [1,2]
  mcVal =  2 if par.MCtype == "hatch2fail" else 0
  uVal = 7
  # eqval = np.isin(assignedFate,val)
  eqval = np.isin(trueFate,val)
  # if db>=3: print(f"\t\t{eqval=}   {len(eqval)=} ")
    #-~~
  # if db>=4: print(f"\t\t\t\t{eqval=}   {np.sum(eqval)=} ")
  
  if par.MCtype!="none":
    nMisclass = int(np.round(par.propMC * np.sum(eqval)))
    nUnknown  = int(np.round(par.propUnk * np.sum(eqval)))
  else:
    nMisclass = int(np.round(0.02*np.sum(eqval)))
    nUnknown = int(np.round(0.0*np.sum(eqval)))
  tot       = nMisclass + nUnknown
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if db>=2: print("\t\t\t NUMBER TO MISCLASSIFY:", nMisclass, end=" ")
  # if db>=2: print("\t\t\t NUMBER TO MARK UNKNOWN:", nUnknown, end=" " )
  # if db>=2: print(f"\t\t\t\t{val=} , {mcVal=} , {uVal=} , {tot=}")
  #
  # if db>=4: print(f"\t\t\tBEFORE <{len(assignedFate)}> : {assignedFate=}")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # ind = np.where(assignedFate in val)
  ## NOTE for some reason, returns a tuple with 1 array instead of just an array
  ##+> select random indices to replace at 
  ind = np.where(eqval)
  ind = ind[0]
  # if db>=5: print(f"\t\t\t{ind=} {len(ind)=}", end=" ")
  # ind = np.isin(assignedFate,val)
  # mask = rng.choice(len(assignedFate), size=nMisclass+nUnknown, replace=False)
  # mask = rng.choice(ind, size=nMisclass+nUnknown, replace=False)
  mask = rng.choice(ind, size=tot, replace=False)
  # if db>=5: print(f"\t{mask=}", end=" ")
  rep_vals  =[np.repeat(mcVal,nMisclass), np.repeat(uVal,nUnknown)]
  rep_vals  = np.concatenate(rep_vals).tolist()
  # if db>=5: print(f"\t{rep_vals=}")
  assignedFate[mask] = rep_vals
  # if db>=4: print(f"\t\t\tAFTER: {assignedFate=}")
  return assignedFate
  # nData[:,7] = assignedFate
  # return nData
  # mcMask = rng.choice(len(assignedFate), size=nMisclass, replace=False)
  # uMask = rng.choice(len(assignedFate), size=nUnknown, replace=False)
  # fVals = [0, 1, 2]
  # failVals = [1, 2] ## +> select one randomly to add to the fate
  # hatch    = 0
  # repVals  = rng.choice(addVals, size=nReplace)
  # assignedFate[replace] = 

#-----------------------------------------------------------------------------

# def svy_position(initiation, nestEnd, surveyDays, cn=config):
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

# @profile
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ---- NEST DISCOVERY & OBSERVATION ----------------------------------------
# -----------------------------------------------------------------------------
# def observer(nData, par, rng, stormFin,surveys, stormDays, out, conf):
def observer(nData, par, rng,surveys, stormDays, out, conf):
  """
    The observer searches for nests on survey days.
    Surveys til discovery (success) are calculated as random draws from a
    negative binomial distribution with daily success probability of discProb.
    If surveys til discovery is less than total number of surveys while nest
    is active, then nest is discovered. The observer then assigns fate in
    assign_fate. 
    -----
    ARGS
      nData = id, init, end, fate
      par = params for this set
      surveys = survey days, intervals, storm survey
      stormDays = storm days
      out = premade array for output
    -------
    RETURNS
      ndarray w/ nrows=numNests. 
      [columns = i, j, k, assigned fate, num obs int, intFinal]
    ---------
    NOTES
      Remember, pos[0] is the first survey after initiation, and pos[1] is the first survey after end.

  """
  # if conf.debugObs>=6:
  #   print("nest data!")
  #   arrPrint(nData)
  # rng = np.random.default_rng(seed=conf.rngSeed)
  # print(f"{type(conf)=}  {conf=}")
  initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
  surveyDays, surveyInts,stormSurvey = surveys

  pos = svy_position(initiation, end, surveys[0], cn=conf)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   print("\t\t\t\t>>>> init dates:",end=" ")
  #   arrPrint(nData[:,1])
  #   print("\t\t\t\t>>>> position of initiation date in survey day list:",end=" ") 
  #   arrPrint( pos[0])
  #   print("\t\t\t\t>>>> end dates:",end=" ")
  #   arrPrint(nData[:,2])
  #   print("\t\t\t\t>>>> position of end date in survey day list:",end=" ")
  #   arrPrint( pos[1]) 
  #   print("\t\t\t\t>> survey days with index number:",end=" ")
  #   arrPrint( surveyDays,abval=20)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## +> number of surveys that occur while nest is active?
  tot_svy      = pos[1] - pos[0]   
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if conf.debugObs>=3:
  #   print("\t\t\t|>total num surveys for each nest:", end=" ")
  #   arrPrint(tot_svy)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  svysTilDiscovery = rng.negative_binomial(n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
  discovered     = svysTilDiscovery < tot_svy
  hatched        = nData[:,3] == 0
  # disc_ind       = pos[0] + svysTilDiscovery #+> index of survey day when nest was discovered
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.debugObs>=3:
    print("\t\t\t|> surveys til discovery = ", end=" ")
    arrPrint(svysTilDiscovery)
    print(f"\t\t\t|> num discovered  = {sum(svysTilDiscovery < tot_svy)}", end=" ")
    print("\t\t\t|> nest discovered? (svysTilDiscovery < num_svy)")
    arrPrint(discovered)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  num_obsTrue = tot_svy - svysTilDiscovery 
  # num_svy = tot_svy - svysTilDiscovery -1 # -1 for final survey
  # num_svy[fate==0] = num_svy[fate==0] + 1 # add it back for hatched bc j=k
  num_obsTrue[~discovered] = 0
  # print(f"{num_obsTrue=}")
  out[:,6] = num_obsTrue
  num_obs = num_obsTrue
  num_obs[~hatched] -= 1
  # print(f"{num_obs=}")
  # if conf.debugObs>=3:
  #   print("\t\t\t|>num obs while active:", end=" ")
  #   arrPrint(num_obs)
  intFinal  = surveyInts[pos[1]] # +> actual length of final int for each nest
  # intFinal[fate==0] = 0 #+> final int for hatched is 0
  # +> need actual final int bc during storms, hatched nests can be marked as flooded 
  # +> if the obs int is longer than usual
  iVal = surveyDays[pos[0]+svysTilDiscovery] # i
  kVal = surveyDays[pos[1]]
  jVal = surveyDays[pos[1]-1]
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if conf.debugObs>=6:
  #   print("\t\t\t\tj & k before:")
  #   arrPrint(jVal)
  #   arrPrint(kVal)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  jVal[fate==0] = kVal[fate==0]
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if conf.debugObs>=5:
  #   print("\t\t\t\tj & k:")
  #   arrPrint(jVal)
  #   arrPrint(kVal)
  # if conf.debugObs>=4:
  #   print(f"\t\t\t{iVal=} \n\t\t\t{jVal=} \n\t\t\t{kVal=}")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #+> num storms befor final interval
  # num_storms = 
  # num_storms = np.sum(np.isin(np.arange(iVal,jVal), stormDays))
  # if conf.debugObs >= 2:
  #   print("\n|> number of storms in interval from i to j:")
  #   arrPrint(num_storms)
  out[:,0][discovered] = iVal[discovered]
  out[:,1][discovered] = jVal[discovered] 
  out[:,2][discovered] = kVal[discovered]
  finalInterv  = mk_per(out[:,1], out[:,2],con=conf)
  ## was there a storm in final interval?
  stormLoc = storm_nest(par.stormFrq, finalInterv, stormDays, conf)
  # stormFinal = any(stormLoc==1)
  stormFinal = (stormLoc==1).any(axis=1)
  # if conf.debugObs>=2:
    # print(f"{finalInterv=}")
    # print(f"{stormLoc=}")
    # print(f"{stormFinal=}")

  # if config.fateType=="fixed":
    # out[:,3] = assign_fixed(pWrong=par.pWrong, wrongVal=par.wType, trueFate=fate, numNests=numNests)
  # else:
  # out[:,3] = assign_fate(par.wType, par.pWrong, cues, fate, par.numNests, par.obsFreq, intFinal, par.stormFate, cn=conf)
  # out[:,3] = assign_fate(par.wType, par.pWrong, fateCues, fate, par.numNests, par.obsFreq, intFinal, par.stormFate, cn=conf)
  # out[:,3] = assign_fate(par.wType, par.decayRate, fate, par.numNests, par.obsFreq, intFinal, par.stormFate, cn=conf)
  # out[:,3] = assign_fate(par,fate,intFinal,cn=conf)
  out[:,3] = assign_fate(par,rng,stormFinal,fate,intFinal,conf)
  # out[:,4] = num_svy - svysTilDiscovery # +> num obs for the nest
  ##+> num_svy - svysTilDisovery is the total num obs (incl final interval)
  #+> could jsut subtract 1 if fate isn't 0

  #+> this one could be easily calculated later:
  # out[:,4][discovered] = jVal[discovered] - iVal[discovered] # +> num obs for the nest
  #+> need number of obs between i and j
  out[:,4] = num_obs 
  out[:,5] = intFinal.astype(int) # length of final interval - transform to integer for the ndarray
  # out[:,6] = num_storms
  # if conf.debugObs>=3:
  #   print("\t\tout=")
  #   dfPrint(out)

  return(out)

#-----------------------------------------------------------------------------
# @profile
     # [9]:len(final int)....[10]:num storms......[11]:num obs total....
# def make_obs(par, storm, survey, conf, nw=2, inff=True, pandas=False):
def make_obs(par,rng, storm, survey, conf, initDat, nw=2, inff=True, pandas=False):
# def make_obs(par, storm, survey, conf, nw=2, inff=initFromFile, pandas=False):
  """
    1. Call functions mk_nests, mk_per, storm_nest, mk_flood, mk_fates, & observer

    2. Combine the output into an array: 

     [0]:nest ID...........[1]:initiation.......[2]:end date..........
     [3]:true fate ........[4]:first found......[5]:last active.......
     [6]:last checked......[7]:assigned fate....[8]:num obs active.... 
     [9]:len(final int)....[10]:num obs total....

    --------
    RETURNS:
      numpy ndarray containing nest & observation data (column indices above)
      
      Can also uncomment lines to save nest data to .npy file
      
    And other lines to make nest data that's compatible with the old script.
  """
  # nd     = np.zeros(shape=(par.numNests, 3), dtype=int)
  # nd2    = np.zeros(shape=(par.numNests, 6), dtype=int)
  # rng = np.random.default_rng(seed=conf.rngSeed)
  nd     = np.zeros(shape=(par.numNests, 3), dtype=np.int16)
  nd2    = np.zeros(shape=(par.numNests, 7), dtype=np.int16)
  colnames = ['ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs']
  # if conf.debugObs>=3: print(f"\t\t{len(colnames)=}")

  # +> fateCues directly correlates to obsFreq, so doesn't need to be param
  # fateCues   = 0.71 if par.obsFreq > 5 else 0.76 if par.obsFreq == 5 else 0.8
  # if par.pWrong > 0: fateCues=1
  # if conf.debug: print("\t|>|>pWrong:", par.pWrong,"& probability that fate cues are present:", fateCues)
  # NOTE should I make sure all nests live for at least a day?

  # +> ---- make the nests: ---------------------------------------------------
  # nData      = mk_nests(par=par, nestData=nd, conf=conf)
  if conf.debugNests>=2: print("\n\t[*] [*] [*] [*] [*] making nests [*] [*] [*] [*] [*] [*] [*] [*] ")
  nData      = mk_nests(par, rng, nd, conf, initDat, initff=inff, nWeek=nw)
  nestPeriod   = mk_per(nData[:,1], (nData[:,2]), con=conf) # changed output of mk_nests 
  if conf.debugNests>=3: print(f"\t\tINIT DATES: {nData[:,1]}")
  # finalInterv  = mk_per(nData[:,5], nData[:,6],con=conf)
  # ## was there a storm in final interval?
  # stormFinal = storm_nest(par.stormFrq, finalInterv, storm, conf)
  # if conf.debugObs>=4:
    # print( f"\t\t\t\t>> start & end of nest period:")
    # arrPrint(nestPeriod)
  stormOut     = storm_nest(par.stormFrq, nestPeriod, storm, con=conf)
  stormDat     = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests, con=conf,rng=rng)
  # flooded    = stormDat[:,2] # need more than just whether nest flooded; need date
  hatched    = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
  # if conf.debugObs>=6: 
  #   print(f"\t\t\t|>end - init >= hatch time:")
  #   for x in range(5):
  #     print(f"\t\t\t\t{nData[x,2]}-{nData[x,1]}>={par.hatchTime}" )
  nData    = mk_fates(nData, par.numNests, hatched, stormDat, storm, con=conf)
    # expoList   <- logex$calc_daily_expo(numNests=nNest, surveyDays=survey[[1]],
                                     # surveyInts=survey[[2]], firstDay=nestData$i,
                                     # lastDay=nestData$k, db=config$debugNests)
  # print(f"{nData=}")
  # if conf.debugObs>=3:
  #   print("\t\tnestdata:")
  #   dfPrint(nData, nprint=30)

  # +> ---- observer: ---------------------------------------------------------
  if conf.debugObs>=2: print("\n\t[*] [*] [*] [*] [*] observer [*] [*] [*] [*] [*] [*] [*] [*] ")
  obs = observer(nData,par,rng,surveys=survey,stormDays=storm,out=nd2,conf=conf)

  # +> ---- concatenate to make data for the nest models: ---------------------
  nestData = np.concatenate((nData, 
                 obs
              #  stormOut[0][:,None] # storms per nest
                 ), axis=1)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.debugObs>=3:
    print(f"\t\tnestData {nestData.shape}:")
    dfPrint(nData, nprint=30)
    # dfPrint(nestData)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if pandas: nestData = pd.DataFrame(nestData,columns=colnames)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.debugObs>=3:
    disc = nestData[nestData[:,8]>0]
    # print(f"\t{disc.sum()=}")
    assignedFate = disc[:,7]
    trueFate     = disc[:,3]
    print( f"\t|>true fates (discovered only):",
        f" H:{sum(trueFate==0)}|D:{sum(trueFate==1)}"
        f"|Fl:{sum(trueFate==2)}|U:{sum(trueFate==7)}", end=" "
        )
    print(
        f"\t\t|>assigned fates (discovered only):",
        f" H:{sum(assignedFate==0)}|D:{sum(assignedFate==1)}"
        f"|Fl:{sum(assignedFate==2)}|U:{sum(assignedFate==7)}"
        )
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   arrPrint(assignedFate)
  # if conf.debugObs>=6:
  #   print_prop(nestData, "all")
  #   print_prop(nestData, "disc")
  # nestData[:,3] = (nestFate==0)
  # nestData[:,4] = (nestFate==2)
  # nestData[:,5] = sTilDisc
  # nestData[:,6] = disc1
  # nestData[:,12] = nestData[:,12] > 3
  # nestData[:,14] = 416 # exposure dayso
  # nestData[:,15] = nestData[:,2] - nestData[:,1]
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  # np.savetxt("nestdata_afterflood.csv", nestData, delimiter=",")
  # np.save("nest_data.npy", nestData)
  # np.save(nestfile, nestData)
  return(nestData)

