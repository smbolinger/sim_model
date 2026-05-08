#!/usr/local/bin/python
import numpy as np
import pprint
from pathlib import Path
# from helpers import load_config, init_from_csv, sprob_from_csv, searchSorted2
import matplotlib.pyplot as plt
from helpers import init_from_csv, sprob_from_csv, searchSorted2, print
from print_func import arrPrint,dfPrint
# from settings import config, rng
# from rsettings import config, rng
np.set_printoptions(precision=3)

# debug=config.debugNest

# def stormGen(frq, dur, stormDat=stormDat):
def stormGen(frq, dur, config, rng, stormDat, stFromFile=True):
  """
    generate a list of days where storms happened.

    the probabilities and week start dates used are read from csv outside the 
    function to streamline it.

    for rng.choice: a=array of values to choose from, p=associated probabilities
    ----
    RETURNS:
      a numpy array of values
  """
  # stormDat=sprob_from_csv(storm_init) # is evaluated later, can account for wsl filenames
  # rng = np.random.default_rng(seed=config.rngSeed)
  if stFromFile:
    # stormDat=sprob_from_csv(config.stormInit) # is evaluated later, can account for wsl filenames
    stormProb = [0.006,0.019,0.044,0.025,0.069,0.044,0.050,0.044,0.025,0.038,
                 0.050,0.057,0.031,0.069,0.025,0.069,0.038]
                 # 0.050,0.057,0.031,0.069,0.025,0.069,0.038,0.082,0.031,0.038,
                 # 0.063,0.050,0.031]
    stormProb = stormProb/np.sum(stormProb)
    print(f"\t\tstormProb by week & week start day [{len(stormProb)=}]:")
    # dfPrint(np.array([stormProb]), names=list(np.arange(23)))
    # stormWeek = np.arange(23)
    stormWeek = np.arange(1,18,1)
    weekStart = stormWeek*7
    dfPrint(np.array([stormProb]), names=list(weekStart))
    # out = rng.choice(a=[*stormDat], size=frq, replace=False, p=list(stormDat.values()))
    # print("\t\t|>choosing storm weeks from list:", end=" ")
    out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    # print(f"\t\t\t{out=}")
    rand =  rng.choice(7, size=len(out))
    out = out + rand
    # print(f"\t\t\tadd a random number to week start: {rand=} ; {out=}")
    # out3 = out + rng.integers(7)
    # print(f"{out3=}")
    # print("|>choosing storm days from imported data ", end=" ")
  else:
    out = rng.choice(40, size=frq,replace=False)
    print("|>SMALL: choosing storm days from np.arange(40)", end=" ")
  dr = np.arange(0, dur, 1)
  stormDays = [out + x for x in dr] # add sequential storm days when dur>1
  stormDays = np.array(stormDays).flatten()
  # splits      = np.where(np.diff(stormDays)!=1)[0] +1 # print(f"{splits=}")
  # storms      = np.split(stormDays, splits)
  # print(f"\t\t{storms=}")
  print(f"\t\t{stormDays=}")
  # print("\t\t>> storm days:", stormDays)
  # NOTE: should i move part of the storm creation out of mk_survey_days to here?
  # arrPrint(stormDays)
  return(stormDays)
# -----------------------------------------------------------------------------
def mk_init(numNests,config, rng, initDat, hTime="all", nWeek=0, initFromFile=True):
  """
    make initiation dates

    initFromFile & nWeek are used when number of breeding
    days is < 180 & we can't use init probs from file (too long)
  """
  # rng = np.random.default_rng(seed=config.rngSeed)
  if initFromFile:
    # initDat=init_from_csv(config.stormInit) # this will evaluate after storm_init has been changed for wsl
    if hTime==16:
      coniInit = [2,11,9,4,22,18,14,7,11,2,6,2,20]
      initProb = coniInit/np.sum(coniInit)
      coniWeek = np.arange(4,16,1)
      weekStart = coniWeek * 7
      initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
    elif hTime==20:
      leteInit = [4,74,67,48,51,33,42,41,34,28,36,10,7,96]
      initProb = leteInit/np.sum(leteInit)
      leteWeek = np.arange(3,16,1)
      weekStart = leteWeek * 7
      initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
    elif hTime==28:
      wiplInit = [1,7,18,7,8,6,1,5,11,3,1,6]
      initProb = wiplInit/np.sum(wiplInit)
      wiplWeek = np.arange(1,12,1)
      weekStart = wiplWeek * 7
      initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
    else:
      inits = [1,7,22,83,86,63,56,60,71,58,42,39,38,16,9]
      initProb = inits/np.sum(inits)
      weeks = np.arange(1,16,1)
      weekStart = weeks*7
      initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
      # initWeek = rng.choice(a=[*initDat], size=numNests, p=list(initDat.values()))  # random starting weeks; len(a) must equal len(p)
    # print(f"\t|> init dates from file")
  else:
    weeks = np.arange(1,nWeek)
    # print(f"\t|>SMALL: init weeks from 1 to {nWeek}", end=" ")
    initWeek = rng.choice(a=weeks, size=numNests)
  initiation = initWeek + rng.integers(7)          # add a random number from 1 to 6 (?) 
  # if debug:
  # if config.debugNests >= 4:
  #   print(">> initiation week start days:\n", initWeek) 
  return(initiation)

#-----------------------------------------------------------------------------
def mk_surv(numNests, hatchTime, pSurv, con,rng): #+>print
  """
  Decide how long each nest is active

  >> use a negative binomial distribution - distribution of number of
      failures until success 
  
    >> in this case, "success" is actually the nest failing,
        so use 1-pSurv (the failure probability) 
    >> gives you number of days until the nest fails (survival)
    >> if survival > incubation time, then the nest hatches 

  >> then use survival to calculate end dates for each nest
      (end = initiation + survival)
  >> set values > incubation time to = incubation time (nest hatched)
      (need to because you are summing the survival time)
  >> once nest reaches incubation time (+/- some error) it hatches
      and becomes inactive

  """

  # rng = np.random.default_rng(seed=con.rngSeed)
  survival = np.zeros(shape=(numNests), dtype=np.int32)
  survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests) 
  # survival = survival - 1 # but since the last trial is when nest fails, need to subtract 1
  survival[survival > hatchTime] = hatchTime # add some amt of error?
  # if con.debugNests>=3:
  #   print("\t\t\t|> survival in days:", end=" ") 
  #   arrPrint(survival)
  return(survival)
# -----------------------------------------------------------------------------
def mk_nests(par, rng, nestData, conf, initDat, initff=True, nWeek=2):  #+>print
  """
    nestData is an empty np array to be filled.
      |> try subtracting 1 from survival time
    
    Returns:
    -------
    3 columns: nest ID, initiation date, end date
  
    Notes 
    -----
    1. Unpack necessary parameters - some have only 1 member, but they are still treated as arrays, not scalars
    2. Assign values to the dataframe
    NOTE: why not call mk_fates from within this function?
  
  """
  # rng = np.random.default_rng(seed=conf.rngSeed)
  nestData[:,0] = np.arange(par.numNests) # column 1 = nest ID numbers 
  nestData[:,1] = mk_init(par.numNests, conf, rng, initDat,initFromFile=initff, nWeek=nWeek)                # record to a column of the data array
  # if conf.debugNests==3:
  #   print(f"init dates: {nestData[:,1]}")
  survival = mk_surv(par.numNests, par.hatchTime, par.probSurv, con=conf,rng=rng)
  # if conf.debugNests==3: print(f"\t\t\t{survival.sum()=} - total nest days")
  nestData[:,2] = nestData[:,1] +survival
  # nestData[:,2] = nestData[:,1] +survival -1
  # if conf.debugNests==3:
  #   print("\n\t\t\t>> ID, init, & end:\n")
  #   arrPrint(nestData[0:5,:])
  #   print("\n\t\t\t\t. . . . . .\n")
  #   arrPrint(nestData[-5:,:])
  # # if conf.debugNests>=6:
  #   # print("\n\t\t\t>> ID, init, & end:\n")
  #   # arrPrint(nestData)
  # ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
  # # NOTE Remember that int() only works for single values 
  return(nestData)
# ---- FLOODING & SUCH -------------------------------------------------------
def storm_nest(stormFreq, nestPeriod, stormDays, con):
  """
  Returns:
  -------
  a list containing numStorms & stormNestIndex

  Background:
  ----------
  >> stormNestIndex searches for storm days w/in active period of each nest
    - returns index where storm day would be within the active interval: 0 = before init; 2 = after end; 1 = within interval
    - fate cues should become harder to interpret after storms
  """
  stormNestIndex = np.zeros((len(nestPeriod), stormFreq))
  stormNestIndex = searchSorted2(nestPeriod, stormDays)
  # stormNest = np.any(stormNestIndex == 1, axis=1) 
  # numStorms = np.sum(stormNestIndex==1, axis=1) # axis=1 means summing over rows?

  # return([numStorms, stormNestIndex])
  # return(numStorms)
  return(stormNestIndex)
# -----------------------------------------------------------------------------
def mk_flood( stormDays, pMortFl, stormIndex, numNests, con,rng):
  """
  NEED TO KNOW WHICH STORM SO CAN CHANGE END DATE
  Decide which nests fail from flooding:
    1. Create a vector of random probabilities drawn from a uniform dist
    2. Compare the random probs to pfMort
    3. If flooded=1 and it was during a storm, then nest flooded
    
  Arguments:
    all storm days; prob mort from flood; 
    stormIndex=output from storm_nest()-each storm w/in interval or not?
    number nests; config

  Creates:
    numStorms - vector telling how many storm periods intersected with 
          active period, for each nest
    whichStorm - during which storm is flood first true?
    flP    - random probabilities, 1 per storm

  Returns: list:
    ............[0] num storms.......[1] which storm first flooded
    ............[2] T/F nest flooded AND during storm? 
  """
  # pfMort = params[2]     # prob of surviving (not flooding) during storm
  # print("prob of failure due to flooding:", pfMort)
  # pflood = rng.uniform(low=0, high=1, size=numNests) 
  # numStorms, stormIndex= stormOut
  # stormIndex = stormIndex.astype(int)
  # NOTE: is this the best way to get number of storms??
  numStorms = np.sum(stormIndex==1, axis=1) # axis=1 means summing over rows?
  # stormNest = numStorms >= 1
  # snCount   = sum(stormNest)
  flooded = np.zeros(numNests, dtype=np.int32) ## keep track of flooded nests
  #can be zeros, but need to remember 0 is also an index
  whichStorm = np.zeros(numNests, dtype=np.int32)
  # totStormDays = 
  # flP = rng.uniform(low=0, high=1, size=sum(numStorms>0)) # not quite right because prob of flooding stays the same or each nest in different storms
  flP = rng.uniform(low=0, high=1, size=sum(numStorms)) 
  if con.debugFlood>=2:
    # print(f"\t\t\t|>{len(stormIndex)=} {stormIndex=}")
    print("\t\t\t|>stormIndex:")
    arrPrint(stormIndex,abval=5)
    print(f"\t\t\t|>{len(stormDays)=} {stormDays=}")
    print("\t\t\t|>prob of flooding:", pMortFl)
    # arrPrint(pMortFl)
  if con.debugFlood>=1:
    print(f"\t\t\t|>random probabilities, one per storm ({len(flP)=}) :")
    arrPrint(flP)
  x=0
  # np.savetxt("storm_index.csv",stormIndex, delimiter=",")
  for n in range(numNests):
    # storms = np.zeros(len(stormDays))
    # for s in range(numStorms[n]):
    if numStorms[n] > 0:
      flood = np.zeros(len(stormDays), dtype=np.int32)
      if con.debugFlood>=2: print(f"\t\t\t\t\t{len(flood)=}) :",end=" ")
      for s in range(len(flood)): ##for each storm day when nest active:
        # if con.debugFlood>=3: print(f"{range(len(flood))=}", end=" ")
        if con.debugFlood>=2: print(f"{s=}", end=" ")
        if stormIndex[n,s] == 1:
          flood[s] = flP[x] < pMortFl # I changed how pMortFL was defined.
          if con.debugFlood>=3: print(f"{stormIndex[n,s]=} ; {flP[x]=:.3f} < {pMortFl=} ? {flood[s]=} ; {x=} ", end=" ")
          x=x+1 ##
      if any(flood.astype(bool)):
        flooded[n] = 1 # but if default val is 0, could be confused for index 0...
        whichstorm = np.where(flood==1)[0] # first index where val==True
        # if con.debugFlood>=3: print(f"{whichStorm=}")
        if con.debugFlood>=3: arrPrint(whichStorm, ind=10)
        whichStorm[n] = stormDays[whichstorm[0]]
        if con.debugFlood>=3: arrPrint(whichStorm, ind=10)
        # if con.debugFlood>=2: print(f"{whichStorm=}")
      
  # stormInfo = np.concatenate((stormInfo, stormIndex), axis=1)
  # need to check whether this is the correct distribution 
  # NOTE: still needs to be conditional on nest having failed already...  
  # NOTE np.concatenate joins existing axes, while np.stack creates new ones
  # np.savetxt("storm_out.csv", np.concatenate(stormInfo, stormIndex))
  # flooded = np.where(pflood>pMortFl, 1, 0) # if pflood>pfMort, flooded=1, else flooded=0 
  # and/or/not don't work bc it's a vector; since it's 1 and 0, can use arithmetic: 
  stormNest = numStorms >= 1
  stormInfo = np.zeros((numNests, 3))
  stormInfo[:,0] = numStorms
  stormInfo[:,1] = whichStorm # stormInfo[:,1] = stormDays[whichStorm]
  stormInfo[:,2] = flooded # true number flooded

  return(stormInfo)
# -----------------------------------------------------------------------------

def mk_fates(nestDat, numNests, hatched,stormInfo, stormDays, con): #+>print
  """
    Want number flooded to derive organically from the storm activity, instead 
    of being a preset value

    Runs mk_flood() to update end dates to account for storms. 
    Then adds a column for true fate to nest data.

    hatched = nests that exceeded the incubation time threshold

    stormInfo = output from mk_flood()

  Returns:
    Nest data with true fate added and end dates for storm nests updated.
  """
  # NOTE probably easier to make debug arg for each function and then set it to the config value when called...
  
  trueFate = np.empty(numNests) 
  trueFate.fill(1) # nests that didn't flood or hatch were depredated 
  flooded = stormInfo[:,2].astype(int)
  whichStorm = stormInfo[:,1].astype(int) # now this is the actual storm DAY, not the index
  trueFate[hatched == True] = 0 # was nest discovered?  
  trueFate[flooded == True] = 2  # should override the nests that "hatched" that were actually during storm
  if con.debugNests>=3:
    print( "\t\t\t|>|> end date before storms accounted for:", end=" ")
    arrPrint(nestDat[:,2])

  ## change end date for flooded nests
  nestDat[:,2][flooded==True] = whichStorm[flooded==True]
  if con.debugNests>=1: print("\t\t\t|>|> hatch?", sum(hatched), end=" ")
  if con.debugNests>=3: arrPrint(hatched)
  if con.debugNests>=1: print( "\t\t\t|>|> flood?", sum(flooded))
  if con.debugNests>=3:
    arrPrint( flooded)
    print( "\t\t\t|>|> end date after storms accounted for:", end=" ")
    arrPrint(nestDat[:,2])
  
  nestDat = np.concatenate((nestDat, trueFate[:,None]), axis=1)
  if con.debugNests>=3: print("\t\t\t[*] [*] creating fates [*] [*] [*]")
  if con.debugNests>=3: print(f"\t\t\t\t{flooded=} | {whichStorm=} | {hatched=}")
  if con.debugNests>=2:
    print("\t\t>>> true final nest fates:", end=" ")# # ---- TRUE DSR ------------------------------------------------------------
    print(
        f"  H:{sum(trueFate==0)}|D:{sum(trueFate==1)}|Fl:{sum(trueFate==2)}",
        end=" "
        )
    arrPrint(trueFate)
  #OH, but I don't ever return nestDat anyway. so maybe this should be a function that ADDS true fate to nestDat.

  return(nestDat)

# +>old:
  # # Calculate proportion of nests hatched and use to calculate true DSR
  # #   daily mortality = num failed / total exposure days
  # #   (num failed =  total-num hatched) 
  # #   (total exposure days = add together survival periods)
  # #   DSR = 1 - daily mortality
  # trueHatch = trueFate==0 # true/false did nest hatch (after storms accounted for)?
  # nestData[:,3] = trueHatch.astype(int)
  
  # trueDSR2 = 1 - ( (numNests - trueHatch.sum()) / survival.sum() ) 
  # if debug: print(">>>> total exposure days (unobserved):", survival.sum())
  # if debug: print(">>>>> and true DSR, calculated correctly:", trueDSR2)
# How does the observer assign nest fates? 
