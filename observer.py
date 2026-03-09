#!/usr/local/bin/python
import numpy as np
from makeNests import mk_nests, mk_fates

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
# @profile
def assign_fate(assignVal, pWrong, fateCuesPresent, trueFate, numNests, obsFr, intFinal, stormFate, cn=config):
    """
    The observer assigns the correct fate based on a comparison of a set 
    probability to random draws from a uniform distribution. If observer is 
    incorrect, then they assign a fate of unknown unless stormFate==True, in
    which case all nests that ended in a period that contained a storm are 
    assumed to have failed due to the storm.
    
        fateCuesProb=random values to compare
        fateCuesPres=probability of fate cues being present
            has different value if there was a storm in final interval.
        if fate percentages are fixed, fateCuesPres should be the same (=1) for all
    
    Returns: vector w/ assigned fate for each nest
    """
    
    # assignedFate = np.zeros(numNests) # if there was no storm in the final interval, correct fate is assigned 
    assignedFate=np.empty(numNests)
    assignedFate.fill(7) # default is unknown; fill with known fates if field cues allow

    fateProb = rng.uniform(low=0, high=1, size=numNests)
    fateCuesPres = np.zeros(numNests)
    fateCuesPres.fill(fateCuesPresent)
    fateCuesPres[intFinal > obsFr] = 0.1 # nests with longer final interval have lower chance of cues
        
    assignedFate[fateProb < fateCuesPres] = trueFate[fateProb < fateCuesPres] 
    assignedFate[fateProb < pWrong] = assignVal # if fixed percentages turned off, pWrong == 0

    if cn.debugObs: print(">> true fates:", trueFate, len(trueFate))
    if cn.debugObs: print(">> assigned fates:", assignedFate, len(assignedFate))
    if stormFate: assignedFate[intFinal > obsFr] = 2
    if cn.debugObs: 
        print(">> compare random probs to fateCuesPresent:\n", 
              [fateProb,fateCuesPres], 
              fateProb.shape)
        print(f">> or to pWrong: {pWrong} with fill value: {assignVal}")
        print(">> nests with storm in final interval:", np.where(intFinal>obsFr))
        print(">> storm fate == True?", stormFate)
        print(">> assigned fates after incorrect fates assigned:", assignedFate, len(assignedFate))
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
    surveyDays, surveyInts = surveys

    pos = svy_position(initiation, end, surveys[0])
    num_svy          = pos[1] - pos[0]   
    svysTilDiscovery = rng.negative_binomial(n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
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
    # if config.fateType=="fixed":
        # out[:,3] = assign_fixed(pWrong=par.pWrong, wrongVal=par.wType, trueFate=fate, numNests=numNests)
    # else:
    out[:,3] = assign_fate(par.wType, par.pWrong, cues, fate, par.numNests, par.obsFreq, intFinal, par.stormFate)
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
       [0]:nest ID.............[1]:initiation.......[2]:survival(w/o storm).....
       [4]:first found.........[5]:last active......[6]:last checked............
       [7]:assigned fate.......[8]:num obs int......[9]:days in final interval..

    Returns:
        numpy ndarray containing nest & observation data (column indices above)
        
        Can also uncomment lines to save nest data to .npy file
        
        And other lines to make nest data that's compatible with the old script.
    """
    nd       = np.zeros(shape=(par.numNests, 3), dtype=int)
    nd2      = np.zeros(shape=(par.numNests, 6), dtype=int)
    # fateCues directly correlates to obsFreq, so doesn't need to be param
    fateCues   = 0.71 if par.obsFreq > 5 else 0.76 if par.obsFreq == 5 else 0.8
    if par.pWrong > 0: fateCues=1
    if debug: print("pWrong:", par.pWrong,"& probability that fate cues are present:", fateCues)
    # NOTE should I make sure all nests live for at least a day?
    # ---- make the nests: ---------------------------------------------------
    nData          = mk_nests(par=par, nestData=nd)
    nestPeriod     = mk_per(nData[:,1], (nData[:,2])) # changed output of mk_nests 
    stormOut       = storm_nest(par.stormFrq, nestPeriod, storm)
    stormDat       = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests)
    # inclFlood        = mk_flood(nData, par.pMortFl, stormsPerNest, numNests=par.numNests)
    # flooded        = stormDat[:,2] # need more than just whether nest flooded; need date
    hatched        = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
    if config.debugNests: print("hatched (before storms)=", hatched, sum(hatched))
    nData      = mk_fates(nData, par.numNests, hatched, stormDat, storm)
    # ---- observer: ---------------------------------------------------------
    # par2      = [par.numNests, par.obsFreq, par.discProb, par.stormFate, par.wType, par.pWrong]
    # obs       = observer(nData, par=par2, cues=fateCues, surveys=survey, out=nd2)
    obs       = observer(nData, par=par, cues=fateCues, surveys=survey, out=nd2)
    # ---- concatenate to make data for the nest models: ---------------------
    nestData = np.concatenate((nData, 
                            #    np.zeros((par.numNests,4)),
                               obs
                            #    stormOut[0][:,None] # storms per nest
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
