#!/usr/lecal/bin/python
import sys
import numpy as np
import pprint
from pathlib import Path
# from helpers import load_config, init_from_csv, sprob_from_csv, searchSorted2
import matplotlib.pyplot as plt
from helpers import init_from_csv, sprob_from_csv, searchSorted2, print
from print_func import arrPrint,dfPrint
# from settings import config, rng
# from rsettings import config, rng
# np.set_printoptions(precision=3)

# def stormGen(frq, dur, stormDat=stormDat):
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
  # stormDat=sprob_from_csv(storm_init) # is evaluated later, can account for wsl filenames
  # rng = np.random.default_rng(seed=config.rngSeed)
  stormProb,weekStart = stormDat
  if config.debugNests>=4: dfPrint(np.array([stormProb]), names=list(weekStart))
  if stFromFile:
    # stormDat=sprob_from_csv(config.stormInit) # is evaluated later, can account for wsl filenames
    # out = rng.choice(a=[*stormDat], size=frq, replace=False, p=list(stormDat.values()))
    # print("\t\t|>choosing storm weeks from list:", end=" ")
    out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    if db>=3: print(f">> stormGen: {weekStart=} {stormProb=}") 
    # print(f"\t\t\t{out=}")
    rand =  rng.choice(7, size=len(out))
    out = out + rand
    # print(f"\t\t\tadd a random number to week start: {rand=} ; {out=}")
    # out3 = out + rng.integers(7)
    # print(f"{out3=}")
    # print("|>choosing storm days from imported data ", end=" ")
  else:
    out = rng.choice(40, size=frq,replace=False)
    if db>=1: print("|>SMALL: choosing storm days from np.arange(40)", end=" ")
  dr = np.arange(0, dur, 1)
  stormDays = [out + x for x in dr] # add sequential storm days when dur>1
  stormDays = np.array(stormDays).flatten()
  # splits      = np.where(np.diff(stormDays)!=1)[0] +1 # print(f"{splits=}")
  # storms      = np.split(stormDays, splits)
  # print(f"\t\t{storms=}")
  # if config.debugNests>=4: print(f"\t\t{stormDays=}")
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
  # probInit, weekStart = initDat
  initProb, weekStart = initDat # weekStart = weeks * 7
  if initFromFile:
    initWeek = rng.choice(a=weekStart, size=numNests, p=initProb)  # random starting weeks; len(a) must equal len(p)
    if config.debugNests >= 3:
      print(f">> mk_init: {weekStart=}") 
      print(f">> mk_init: {initProb=}") 
      # initWeek = rng.choice(a=[*initDat], size=numNests, p=list(initDat.values()))  # random starting weeks; len(a) must equal len(p)
  else:
    weeks = np.arange(1,nWeek)
    print(f"\t|>SMALL: init weeks from 1 to {nWeek}", end=" ")
    initWeek = rng.choice(a=weeks, size=numNests)
  initiation = initWeek + rng.integers(1, 7, size=numNests)          # add a random number from 1 to 6 (?) 
  if config.debugNests >= 3:
    print(">> mk_init: initiation week start days:\n", initWeek) 
    print(">> mk_init: plus random integer:\n", initiation) 
  return(initiation)

# def mk_per(start, end, con):
#
#   nestPeriod = np.stack((start, end)) # +> create array of tuples
#   # NOTE need the double parentheses so it knows output is tuples
#   nestPeriod = np.transpose(nestPeriod) # +> an array of start,end pairs 
#   return(nestPeriod)

#-----------------------------------------------------------------------------
# def mk_surv(numNests, hatchTime, pSurv, con,rng,plusone=True): #+>print
def mk_surv(numNests, hatchTime, pSurv, con,rng,plusone=False): #+>print
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
  Note: can add 1 because this is days survived til failure; don't count initial day

  """
  survival = np.zeros(shape=(numNests), dtype=np.int32)
  while np.any(survival==0):
    mask = (survival==0)
    survival[mask] = rng.negative_binomial(n=1, p=(1-pSurv), size=np.sum(mask)) 
    # survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests) 

  # survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests+50) 
  # survival = survival[survival>0]
  # survival = survival[:numNests]
  if plusone:
    survival = survival+1 ## add 1 so all nests survive >0 days
  # # survival = survival - 1 # but since the last trial is when nest fails, need to subtract 1
  survival[survival > hatchTime] = hatchTime # add some amt of error?
  if con.debugNests>=3:
    print("\t\t\t|> mk_surv: survival in days:", end=" ") 
    arrPrint(survival)
  return(survival)
# -----------------------------------------------------------------------------

def mk_set_nests(num_nest, prop_hatch,rng,conf):
  """
  """
  # print(f"{num_nest=}")
  nestData = np.zeros(shape=(num_nest,4))
  # print(f"{nestData=}")
  nestData[:,0] = np.arange(num_nest)
  nestData[:,1] = rng.choice(a=np.arange(1,90),size=num_nest)
  # print(f"{nestData[:,1]=}")
  nestData[:,3] = np.ones(shape=(num_nest))
  # chng = np.random.permutation(num_nest)[:(num_nest/2)]
  num_change = int(num_nest/2)
  mask = np.zeros(shape=(num_nest), dtype=bool)
  chng = rng.choice(np.arange(num_nest), size=(num_change), replace=False)
  mask[chng] = True
  alive = rng.choice(a=np.arange(1,19),size=(num_change))
  # print(f"{alive=}{len(alive)}")
  ## change half of fates to hatch
  nestData[:,3][mask] = 0
  # print(f"{nestData[:,2]=}")
  nestData[:,2][mask] = nestData[:,1][mask] + 20
  # print(f"{nestData[:,3]=}")
  # print(f"{len(nestData[:,3][chng])=}{len(nestData[:,3][~chng])=}")
  # nestData[:,3][~chng] = nestData[:,1][~chng] + alive
  nestData[:,2][~mask] = nestData[:,1][~mask] + rng.choice(a=np.arange(1,19),size=(num_change))
  # if conf.testing=="yes":
  #   print(f"{chng=}{len(chng)}")
  #   print(f"{len(nestData[:,2][mask])=}{len(nestData[:,2][~mask])=}")
  #   # print(f"{nestData[:,2][~mask]=}")
  #   print(f"{nestData[:,2]=}")

    # print(f"{nestData[nestData[:,3]==0]=}")
    # print(f"{nestData[nestData[:,3]==1]=}")
  # print(f"{nestData=}")
  return nestData


# def mk_nests(par, rng, nestData, conf, initDat, initff=True, nWeek=2):  #+>print
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
  # mk_set_nests(par.numNests,0.50, rng, conf)
  # rng = np.random.default_rng(seed=conf.rngSeed)
  nestData[:,0] = np.arange(par.numNests) # column 1 = nest ID numbers 
  nestData[:,1] = mk_init(par.numNests, conf, rng, initDat,initFromFile=initff, nWeek=nWeek)                # record to a column of the data array
  survival = mk_surv(par.numNests, par.hatchTime, par.probSurv, con=conf,rng=rng)
  # if conf.debugNests==3: print(f"\t\t\t{survival.sum()=} - total nest days")
  nestData[:,2] = nestData[:,1] +survival

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.debugNests>=4:
    with np.printoptions(threshold=sys.maxsize):
      print(f"\n\t\t\t|> mk_nests: init dates: ",end=" ")
      arrPrint(nestData[:,1],abval=20)
      print("\t\t\t|> mk_nests: survival in days:", end=" ") 
      arrPrint(survival,abval=20)
  # nestData[:,2] = nestData[:,1] +survival -1
  if conf.debugNests==3:
    print("\n\t\t\t>> mk_nests: ID, init, & end:\n")
    print(f"\n\t\t\t\t{nestData[0:9,:].T}{nestData[-9:,:].T}")
    # arrPrint(nestData[0:9,:])
    # print("\n\t\t\t\t. . . . . .\n")
    # arrPrint(nestData[-9:,:])
  # if conf.debugNests>=6:
    # print("\n\t\t\t>> ID, init, & end:\n")
    # arrPrint(nestData)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
  # stormFinal = np.zeros(numNests, dtype=np.int32)
  # totStormDays = 
  # flP = rng.uniform(low=0, high=1, size=sum(numStorms>0)) # not quite right because prob of flooding stays the same or each nest in different storms
  flP = rng.uniform(low=0, high=1, size=sum(numStorms)) 

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if con.debugFlood>=3:
    # print(f"\t\t\t|>{len(stormIndex)=} {stormIndex=}")
    print("\t\t\t|> mk_flood: stormIndex:")
    arrPrint(stormIndex,abval=5)
    print(f"\t\t\t| mk_flood: >{len(stormDays)=} {stormDays=}")
    # arrPrint(pMortFl)
  if con.debugFlood>=2:
    print("\t\t\t| mk_flood: >prob of flooding:", pMortFl, end=" ")
    print(f"\t\t\t| mk_flood: >random probabilities, one per storm ({len(flP)=}) :")
    arrPrint(flP)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  x=0
  # np.savetxt("storm_index.csv",stormIndex, delimiter=",")
  for n in range(numNests):
    # storms = np.zeros(len(stormDays))
    # for s in range(numStorms[n]):
    if numStorms[n] > 0:
      flood = np.zeros(len(stormDays), dtype=np.int32)
      # if con.debugFlood>=2: print(f"\t\t\t\t\t{len(flood)=}) :",end=" ")
      for s in range(len(flood)): ##for each storm day when nest active:

        #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if con.debugFlood>=4: print(f"\t>>mk_flood: {range(len(flood))=}", end=" ")
        if con.debugFlood>=3: print(f"\t>>mk_flood: {s=}", end=" ")
        #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if stormIndex[n,s] == 1:
          flood[s] = flP[x] < pMortFl # I changed how pMortFL was defined.
          
          #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          if con.debugFlood>=3: print(f"\t>>mk_flood: {stormIndex[n,s]=} ; {flP[x]=:.3f} < {pMortFl=} ? {flood[s]=} ; {x=} ", end=" ")
          #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          
          x=x+1 ##
      if any(flood.astype(bool)):
        flooded[n] = 1 # but if default val is 0, could be confused for index 0...
        whichstorm = np.where(flood==1)[0] # first index where val==True
        # if con.debugFlood>=3: print(f"{whichStorm=}")
        # if con.debugFlood>=3: arrPrint(whichStorm, ind=10)
        whichStorm[n] = stormDays[whichstorm[0]]
        # if con.debugFlood>=3: arrPrint(whichStorm, ind=10)

        if con.debugFlood>=1: print(f"\t>>mk_flood: {whichStorm=}")
    # else:
    #   ## a few random storm nests even w/o storms
    #   randN = rng.uniform(low=0, high=numNests-1,size=9)
    #   flooded[randN] = 1
    #   whichStorm[randN] = 1
      
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
  # stormInfo[:,3] = stormFinal # true number flooded

  return(stormInfo)
# -----------------------------------------------------------------------------

def mk_fates(nestDat, numNests, hatched,stormInfo, stormDays, con): #+>print
  """
    Want number flooded to derive organicarrrArddd  dzly from the storm activity, instead 
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

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if con.testing=="yes":
    if con.debugNests>=3:
      # print("\t\tnum storms, which storm, flooded T/F:")
      # print(f"\t\t\t{stormInfo=}")
      stormname = ["num_storm", "which_storm", "flooded?"]
      dfPrint(stormInfo, nprint=20, names=stormname)
    if con.debugNests>=4:
      print( "\t\t\t|>|> mk_fates: end date before storms accounted for:", end=" ")
      arrPrint(nestDat[:,2], abval=20)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ## change end date for flooded nests
  nestDat[:,2][flooded==True] = whichStorm[flooded==True]

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if con.testing=="yes":
    # if con.debugNests>=3: print( f"\t\t\t> {whichStorm[flooded==True]=}")
    # if con.debugNests>=2: print("\t\t\t|>|> # hatched:", sum(hatched), end=" ")
    # if con.debugNests>=3: print("\t\t\t|>|> true fates:", len(trueFate==0), end=" ")
    # if con.debugNests>=3: arrPrint(trueFate)
    if con.debugNests>=2: print("\t\t\t|>|> mk_fates: # hatched:", sum(trueFate==0), end=" ")
    if con.debugNests>=4: arrPrint(hatched, abval=20)
    # if con.debugNests>=2: print( "\t\t\t|>|> # flooded:", sum(flooded))
    if con.debugNests>=2: print( "\t\t\t|>|> mk_fates: # flooded:", sum(trueFate==2))
    if con.debugNests>=4:
      arrPrint( flooded)
      print( "\t\t\t|>|> mk_fates: end date after storms accounted for:", end=" ")
      arrPrint(nestDat[:,2], abval=20)
    if con.debugNests>=5: print("\n\t[*] [*] [*] [*] creating fates [*] [*] [*] [*] [*] [*]\n")
    # if con.debugNests>=4: print(f"\t\t\t\t{flooded=} | {whichStorm=} | {hatched=}")
    if con.debugNests>=4:
      print("\t\t>>> mk_fates: true final nest fates (printed as integer):", end=" ")# # ---- TRUE DSR ------------------------------------------------------------
      print(
          f"  H:{sum(trueFate==0)}|D:{sum(trueFate==1)}|Fl:{sum(trueFate==2)}",
          end=" "
          )
      arrPrint(trueFate.astype(int), abval=20)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  nestDat = np.concatenate((nestDat, trueFate[:,None]), axis=1)

  # OH, but I don't ever return nestDat anyway. so maybe this should be a function that ADDS true fate to nestDat.

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

# def mk_histories(nd, storm, inits, par, rng, conf, inff, nw):
#   """
#     Make nest histories separate from observations
#   """
#   if conf.debugNests>=2: print("\n\t[*] [*] [*] [*] [*] making nests [*] [*] [*] [*] [*] [*] [*] [*] ")
#   nd     = np.zeros(shape=(par.numNests, 3), dtype=np.int16)
#   nData      = mk_nests(par, rng, nd, conf, inits, initff=inff, nWeek=nw)
#   nestPeriod   = mk_per(nData[:,1], (nData[:,2]), con=conf) # changed output of mk_nests 
#   stormOut     = storm_nest(par.stormFrq, nestPeriod, storm, con=conf)
#   stormDat     = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests, con=conf,rng=rng)
#   hatched    = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
#   ## make true nest fates:
#   nData    = mk_fates(nData, par.numNests, hatched, stormDat, storm, con=conf)
#   if conf.other=="setn":
#     nData = mk_set_nests(par.numNests,0.50, rng, conf)
#   return nData
