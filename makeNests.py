#!/usr/local/bin/python
import numpy as np
import pprint
# from helpers import load_config, init_from_csv, sprob_from_csv, searchSorted2
from helpers import init_from_csv, sprob_from_csv, searchSorted2
# config = load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)
# from datsim import config
from settings import config, rng
initDat=init_from_csv(config.stormInit) # this will evaluate after storm_init has been changed for wsl
stormDat=sprob_from_csv(config.stormInit) # is evaluated later, can account for wsl filenames
# debug=config.debugNest

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
  print("\t>> storm days:", stormDays)
  return(stormDays)
# -----------------------------------------------------------------------------
def mk_init(numNests):
  initWeek = rng.choice(a=[*initDat], size=numNests, p=list(initDat.values()))  # random starting weeks; len(a) must equal len(p)
  initiation = initWeek + rng.integers(7)          # add a random number from 1 to 6 (?) 
  # if debug: print(">> initiation week start days:\n", initWeek) 
  return(initiation)

#-----------------------------------------------------------------------------
def mk_surv(numNests, hatchTime, pSurv, con):
  """
  Decide how long each nest is active

  >> use a negative binomial distribution - distribution of number of failures until success 
  
    >> in this case, "success" is actually the nest failing, so use 1-pSurv (the failure probability) 
    >> gives you number of days until the nest fails (survival)
    >> if survival > incubation time, then the nest hatches 

  >> then use survival to calculate end dates for each nest (end = initiation + survival)

  >> set values > incubation time to = incubation time (nest hatched) (need to because you are summing the survival time)
      
       >> once nest reaches incubation time (+/- some error) it hatches and becomes inactive

  """
  survival = np.zeros(shape=(numNests), dtype=np.int32)
  survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests) 
  survival = survival - 1 # but since the last trial is when nest fails, need to subtract 1
  # if debug: print(">> survival in days:\n", survival, len(survival)) 
  
  survival[survival > hatchTime] = hatchTime # add some amt of error?
  if con.debugNests>=3: print("\t>> survival in days:\n", survival, len(survival)) 
  # hatched = survival >= hatchTime # the hatched nests survived for >= hatchTime days 
  # if con.debugNests: print("hatched (no storms):", hatched, hatched.sum())
  ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
  # if debug: print("real hatch proportion:", hatched.sum()/numNests)
  return(survival)
# -----------------------------------------------------------------------------
# def mk_nests(par, init, weekStart, nestData): 
def mk_nests(par, nestData, conf): 
  """
  nestData is an empty np array to be filled.
    
  Returns:
  -------
  3 columns: nest ID, initiation date, end date
  
  Notes 
  -----
  1. Unpack necessary parameters - some have only 1 member, but they are still treated as arrays, not scalars
  2. Assign values to the dataframe
  
  """

  print("\n[*] [*] [*] [*] [*] making nests [*] [*] [*] [*] [*] [*] [*] [*] ")
  nestData[:,0] = np.arange(par.numNests) # column 1 = nest ID numbers 
  nestData[:,1] = mk_init(par.numNests)                # record to a column of the data array
  # if conf.debug: print(">> end dates:\n", nestEnd, len(nestEnd)) 
  survival = mk_surv(par.numNests, par.hatchTime, par.probSurv, con=conf)
  nestData[:,2] = nestData[:,1] +survival
  ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
  # NOTE Remember that int() only works for single values 
  # if conf.debugNests>=1 & conf.debug<3: print("\t>> ID, init, & end:\n", nestData[0:5,:], "\n. . . . . .\n", nestData[-5:,:])
  if conf.debugNests==2:
    print("\n\t>> ID, init, & end:\n")
    print("\t\t",nestData[0:5,:])
    # pprint.pprint(nestData[0:5,:],indent=4)
    print("\n\t. . . . . .\n")
    print("\t\t",nestData[-5:,:])
  if conf.debugNests>=4:
    print("\n\t>> ID, init, & end:\n")
    print("\t\t",nestData)
  # if conf.debugNests>=3: pprint.pprint(nestData, indent=4)
  return(nestData)
# ---- FLOODING & SUCH -------------------------------------------------------
# def storm_nest(nestPeriod, surveysDays, stormDays, con=config):
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
def mk_flood( stormDays, pMortFl, stormIndex, numNests, con):
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
  numStorms = np.sum(stormIndex==1, axis=1) # axis=1 means summing over rows?
  # stormNest = numStorms >= 1
  # snCount   = sum(stormNest)
  flooded = np.zeros(numNests, dtype=np.int32) ## keep track of flooded nests
  #can be zeros, but need to remember 0 is also an index
  whichStorm = np.zeros(numNests, dtype=np.int32)
  # totStormDays = 
  # flP = rng.uniform(low=0, high=1, size=sum(numStorms>0)) # not quite right because prob of flooding stays the same or each nest in different storms
  flP = rng.uniform(low=0, high=1, size=sum(numStorms)) 
  if config.debugFlood>=2: print("\t|>prob of flooding:",pMortFl)
  if config.debugFlood>=4:
    print("\t|>random probabilities, one per storm:", flP)
  x=0
  # np.savetxt("storm_index.csv",stormIndex, delimiter=",")
  for n in range(numNests):
    # storms = np.zeros(len(stormDays))
    # for s in range(numStorms[n]):
    if numStorms[n] > 0:
      # flood = np.zeros(len(stormIndex[n]), dtype=np.int32)
      flood = np.zeros(len(stormDays), dtype=np.int32)
      if config.debugFlood>=3:
        print(f"\n\tnest {n} experienced >=1 storm", end=" ")
      # flood = list(range(len(stormIndex[n])))
      for s in range(len(flood)): ##for each storm day when nest active:
        if config.debugFlood>=4: print("s=", s)
        # print("s=",s)
        if stormIndex[n,s] == 1:
          # flP = rng.uniform(low=0, high=1, size=1)
          # flood[s] = flP[x] > pMortFl
          flood[s] = flP[x] < pMortFl # I changed how pMortFL was defined.
          x=x+1 ##
          if config.debugFlood>=3:
            print(f"\tnest flooded day {s}?", flood[s], end=" ")
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
  if con.debugFlood>=1: 
    print("\t|>prob of failure due to flooding:", pMortFl, end="")
    # print("\tflooded nests:", sum(flooded))
    # if con.debugFlood>=3: print(flooded)
    print("\t\t|>storm nests:", sum(stormNest), end=" ")
    # if con.debugFlood>=3: print( stormNest)
    if con.debugFlood>=3: print( stormNest[:10])
    if con.debugFlood>=3: print( stormNest[-10:])
    # print("flooded and during storm:", floodFail, floodFail.sum())
    print("\t\t|>flooded & storm:", flooded.sum())
    if con.debugFlood>=3: print(flooded[:10])
    if con.debugFlood>=3: print(flooded[-10:])
    # print(f"ID, #storms, which, fl, stormIndex 1-5:\n", stormInfo)
    if con.debugFlood>=1: print(f"\n\tID, num storms, which, fl, stormIndex 1-5:")
    ## enumereate adds a counter to an iterable; can get index & value
    if con.debugFlood>=1 & con.debugFlood<3:
      # for i,row in enumerate(np.concatenate((stormInfo[:10],stormIndex[:10]),axis=1)): 
      for i,row in enumerate(np.concatenate((stormInfo[:5],stormIndex[:5]),axis=1)): 
        print(f"\t\t{i}: {row}")
      print("\t\t\t . . . . . . . ")
      for i,row in enumerate(np.concatenate((stormInfo[-5:],stormIndex[-5:]),axis=1)): 
        print(f"\t\t{i}: {row}")
    if con.debugFlood>=3:
      for i,row in enumerate(np.concatenate((stormInfo,stormIndex),axis=1)): 
        print(f"\t\t{i}: {row}")
  # nestData[:,4] = floodFail.astype(int) 
  # return(floodFail, whichStorm)
  # return(whichStorm) # if value >0, then nest failed during storm
  return(stormInfo)
# -----------------------------------------------------------------------------
# def mk_fates(nestDat, numNests, hatched, flooded, con=config):
# def mk_fates(nestDat, numNests, hatched, whichS, stormDays, con=config):
def mk_fates(nestDat, numNests, hatched,stormInfo, stormDays, con):
  """
  Want number flooded to derive organically from the storm activity, instead 
  of being a preset value

  Runs mk_flood() to update end dates to account for storms. 
  Then adds a column for true fate to nest data.

  stormInfo = output from mk_flood()

  Returns:
  Nest data with true fate added and end dates for storm nests updated.
  """
  
  trueFate = np.empty(numNests) 
  # if con.debugNests>=1: print(">> hatched:", sum(hatched))
  # if con.debugNests>=3: print( hatched)
  trueFate.fill(1) # nests that didn't flood or hatch were depredated 
  flooded = stormInfo[:,2].astype(int)
  whichStorm = stormInfo[:,1].astype(int) # now this is the actual storm DAY, not the index
  trueFate[hatched == True] = 0 # was nest discovered?  
  trueFate[flooded == True] = 2  # should override the nests that "hatched" that were actually during storm
  if con.debugNests>=4: print("\t|>|> end date before storms accounted for:\n", nestDat[:,2])
  # had to add the ==True for some reason
  nestDat[:,2][flooded==True] = whichStorm[flooded==True]
  if con.debugNests>=1: print("\t|>|> hatch?", sum(hatched), end=" ")
  if con.debugNests>=3: print(hatched)
  if con.debugNests>=1: print( "\t\t|>|> flood?", sum(flooded))
  if con.debugNests>=3: print( flooded)
  
  nestDat = np.concatenate((nestDat, trueFate[:,None]), axis=1)
  #OH, but I don't ever return nestDat anyway. so maybe this should be a function that ADDS true fate to nestDat.

  if con.debugNests>=2: print(">>>>> true final nest fates:\n", trueFate, len(trueFate))
  # return(trueFate)
  return(nestDat)

  # # ---- TRUE DSR ------------------------------------------------------------

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
# -----------------------------------------------------------------------------
# ---- NEST DISCOVERY & OBSERVATION ----------------------------------------
# -----------------------------------------------------------------------------
# How does the observer assign nest fates? 
