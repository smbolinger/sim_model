#!/usr/local/bin/python
import numpy as np


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
def mk_init(numNests):
    initWeek = rng.choice(a=[*initDat], size=numNests, p=list(initDat.values()))  # random starting weeks; len(a) must equal len(p)
    initiation = initWeek + rng.integers(7)                    # add a random number from 1 to 6 (?) 
    # if debug: print(">> initiation week start days:\n", initWeek) 
    return(initiation)
# -----------------------------------------------------------------------------
def mk_surv(numNests, hatchTime, pSurv, con=config):
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
    
    Notes 
    -----
    1. Unpack necessary parameters - some have only 1 member, but they are still treated as arrays, not scalars
    2. Assign values to the dataframe
    
    """

    nestData[:,0] = np.arange(par.numNests) # column 1 = nest ID numbers 
    nestData[:,1] = mk_init(par.numNests)                              # record to a column of the data array
    #s if debug: print(">> end dates:\n", nestEnd, len(nestEnd)) 
    survival = mk_surv(par.numNests, par.hatchTime, par.probSurv)
    nestData[:,2] = nestData[:,1] +survival
    ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT
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
    trueFate.fill(1) # nests that didn't flood or hatch were depredated 
    flooded = stormInfo[:,2].astype(int)
    whichStorm = stormInfo[:,1].astype(int) # now this is the actual storm DAY, not the index
    trueFate[hatched == True] = 0 # was nest discovered?  
    trueFate[flooded == True] = 2  # should override the nests that "hatched" that were actually during storm
    if con.debugNests: print(">> end date before storms accounted for:", nestDat[:,2])
    # had to add the ==True for some reason
    nestDat[:,2][flooded==True] = whichStorm[flooded==True]
    if con.debugNests: print("did nest hatch?\n", hatched, sum(hatched), "\ndid nest flood?\n", flooded, sum(flooded))
    
    nestDat = np.concatenate((nestDat, trueFate[:,None]), axis=1)
    #OH, but I don't ever return nestDat anyway. so maybe this should be a function that ADDS true fate to nestDat.

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
# -----------------------------------------------------------------------------
# How does the observer assign nest fates? 
