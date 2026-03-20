#!/usr/local/bin/python
import numpy as np
import pprint
# import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import plotext as plt # plot ASCII plots in the terminal window
from makeNests import mk_nests, mk_fates, mk_flood, storm_nest
from settings import config, rng
from helpers import expDecay, arrPrint, searchSorted2
np.set_printoptions(precision=3)

def mk_surveys(stormDays, obsFreq, breedingDays, conf):
  """
    This function creates the list of survey days by taking a random start date 
    from the first 5 breeding days and creating a range with step size determined
    by observation frequency. Then remove storm days.
    
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

  # stormDays   = stormDays.sort()
  stormDays   = np.sort(stormDays)
  # stormDays   = stormDays.tolist()
  start       = rng.integers(1,high=5) # +> random 1st svy from 1st 5 br days      
  end         = start + breedingDays
  surveyDays  = np.arange(start, end, step=obsFreq)
  stormSurvey = np.isin(surveyDays, stormDays) 
  # print(f"\t\t{stormSurvey=}")
  # stormPos    = searchSorted2(surveyDays, stormDays)
  # stormPos    = np.searchsorted(surveyDays, stormDays)
  splits      = np.where(np.diff(stormDays)!=1)[0] +1
  storms      = np.split(stormDays, splits)
  if len(stormDays) > 0: #+> only if there are storms
    for s in storms:
      lastDay = np.max(s)
      stormPos    = np.searchsorted(surveyDays, lastDay)
      sDiff   =  surveyDays[stormPos] - lastDay
      # sDiff   = lastDay+2 - surveyDays[stormPos]
      # if surveyDays[stormPos] < lastDay + 2:
      if sDiff < 2:
        # mask = surveyDays >= surveyDays[stormPos-1]
        # mask = surveyDays >= lastDay
        # surveyDays[surveyDays >= lastDay] += 2
        surveyDays[surveyDays >= lastDay] += sDiff
      # np.which()
      # print(f"\t\t\t{stormPos=}")
    

  # print(f"{stormPos}")
  # for pos in stormPos:
    # print(f"{surveyDays[pos+1]}")
    
  #: stormSets   = []
  #+> group by difference between index & value to get consecutive groupings
  #+> for each group, jjh
  # grp = groupby(enumerate(stormDays.tolist()), key=lambda x:x[0]-x[1])
  # grp = groupby(enumerate(stormDays), key=lambda x: x[0] - x[1])
  # print(f"\t\t{list(grp)=}")
  # stormEnds = [i[-1] for i in grp]
  # stormSets = ([i[1] for i in g] for _, g in grp)
  # for k,g in groupby(enumerate(stormDays), lambda x:x[0]-x[1]):
  #
  #   grp = (map(itemgetter(1), g))
  #   grp = list(map(int,grp)) # +> make all grp members into ints?
  #   stormSets.append((grp[0],grp[-1])) # +> get start and end of subset

  # print(f"\t\t{list(stormSets)=}")
  # print(f"\t\t{list(stormEnds)=}")

     
    # surveyDays = 
    # print("survey days:", end=" ")
    # for i, s in enumerate(surveyDays):
    # for s in surveyDays:
      # print(s, end=" ")
      # print(s>x)
      # surveyDays[i] = s+2 if s >= x else s 
      # print(surveyDays[i], end=" ")
  # surveyDays = [[s+2 if s > x else s for s in surveyDays] for x in surveyDays[stormSurvey]]
  # surveyDays[stormSurvey] 
  # +> keep only values that aren't in storm_days: 
  # NOTE need to remove here bc for longer storm intervals, may not be removed
  surveyDays  = surveyDays[np.isin(surveyDays, stormDays) == False]
  # surveyDays  = [s + 2 if s > x ]
  # survey interval for first obs is 0:
  surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
  # surveyInts  = np.append(surveyInts, )

  return(surveyDays, surveyInts)

def mk_per(start, end, con):

  nestPeriod = np.stack((start, end)) # +> create array of tuples
  # NOTE need the double parentheses so it knows output is tuples
  nestPeriod = np.transpose(nestPeriod) # +> an array of start,end pairs 
  return(nestPeriod)

# @profile
def assign_fate(assignVal, pWrong, trueFate, numNests, obsFr, intFinal, stormFate, cn):
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
  assignedFate=np.empty(numNests)
  assignedFate.fill(7) # +> default=unk; fill w/ known fate if field cues allow

  fateCuesPresent   = expDecay(n0=1, k=0.1, t=intFinal)
  fateProb = rng.uniform(low=0, high=1, size=numNests)
  assignedFate[fateProb < fateCuesPresent] = trueFate[fateProb < fateCuesPresent] 

  if stormFate: assignedFate[intFinal > obsFr] = 2
  # NOTE fate cues prob should affect all nest fates equally, not just failures
  return(assignedFate)

#-----------------------------------------------------------------------------

def svy_position(initiation, nestEnd, surveyDays, cn):
  """ Finds index in surveyDays of iniatiation and end dates for each nest """
  position = np.searchsorted(surveyDays, initiation) 
  position2 = np.searchsorted(surveyDays, nestEnd)
  surveyDays = dict(zip(np.arange(len(surveyDays)), surveyDays))
  
  return((position, position2)) # +> return a tuple

# -----------------------------------------------------------------------------

# @profile
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ---- NEST DISCOVERY & OBSERVATION ----------------------------------------
# -----------------------------------------------------------------------------
def observer(nData, par, surveys, out, conf):
  """
    The observer searches for nests on survey days.
    Surveys til discovery (success) are calculated as random draws from a
    negative binomial distribution with daily success probability of discProb.
    If surveys til discovery is less than total number of surveys while nest
    is active, then nest is discovered. The observer then assigns fate in
    assign_fate. 

    RETURNS
    -------
      ndarray w/ nrows=numNests. 
      [columns = i, j, k, assigned fate, num *normal* obs ints, intFinal]

    NOTES
    ---------
      Remember, pos[0] is the first survey after initiation, and pos[1] is the first survey after end.

  """
  initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
  surveyDays, surveyInts = surveys

  pos = svy_position(initiation, end, surveys[0], cn=conf)
  num_svy      = pos[1] - pos[0]   
  svysTilDiscovery = rng.negative_binomial(n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
  discovered     = svysTilDiscovery < num_svy
  num_svy[~discovered] = 0

  intFinal  = surveyInts[pos[1]] # +> actual length of final int for each nest
  kVal = surveyDays[pos[1]]
  jVal = surveyDays[pos[1]-1]
  jVal[fate==0] = kVal[fate==0]
  out[:,0] = surveyDays[pos[0]+svysTilDiscovery] # i
  out[:,1][discovered] = jVal[discovered] 
  out[:,2][discovered] = kVal[discovered]
  # if config.fateType=="fixed":
    # out[:,3] = assign_fixed(pWrong=par.pWrong, wrongVal=par.wType, trueFate=fate, numNests=numNests)
  # else:
  # out[:,3] = assign_fate(par.wType, par.pWrong, cues, fate, par.numNests, par.obsFreq, intFinal, par.stormFate, cn=conf)
  # out[:,3] = assign_fate(par.wType, par.pWrong, fateCues, fate, par.numNests, par.obsFreq, intFinal, par.stormFate, cn=conf)
  out[:,3] = assign_fate(par.wType, par.pWrong, fate, par.numNests, par.obsFreq, intFinal, par.stormFate, cn=conf)
  out[:,4] = num_svy - svysTilDiscovery # +> num obs for the nest
  out[:,5] = intFinal.astype(int) # length of final interval - transform to integer for the ndarray

  return(out)

#-----------------------------------------------------------------------------
# @profile
def make_obs(par, storm, survey, conf):
  """
  1. Call functions mk_nests, mk_per, storm_nest, mk_flood, mk_fates, & observer
    2. Combine the output into an array: 
       [0]:nest ID...........[1]:initiation........[2]:survival(w/o storm)....
       [3]:true fate ........[4]:first found.......[5]:last active............
       [6]:last checked......[7]:assigned fate.....[8]:num obs int............
       [9]:days in final interval.............................................

    Returns:
      numpy ndarray containing nest & observation data (column indices above)
      
      Can also uncomment lines to save nest data to .npy file
      
    And other lines to make nest data that's compatible with the old script.
  """
  nd     = np.zeros(shape=(par.numNests, 3), dtype=int)
  nd2    = np.zeros(shape=(par.numNests, 6), dtype=int)

  # +> fateCues directly correlates to obsFreq, so doesn't need to be param
  # fateCues   = 0.71 if par.obsFreq > 5 else 0.76 if par.obsFreq == 5 else 0.8
  # if par.pWrong > 0: fateCues=1
  # if conf.debug: print("\t|>|>pWrong:", par.pWrong,"& probability that fate cues are present:", fateCues)
  # NOTE should I make sure all nests live for at least a day?

  # +> ---- make the nests: ---------------------------------------------------
  nData      = mk_nests(par=par, nestData=nd, conf=conf)
  nestPeriod   = mk_per(nData[:,1], (nData[:,2]), con=conf) # changed output of mk_nests 
  stormOut     = storm_nest(par.stormFrq, nestPeriod, storm, con=conf)
  stormDat     = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests, con=conf)
  # flooded    = stormDat[:,2] # need more than just whether nest flooded; need date
  hatched    = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
  nData    = mk_fates(nData, par.numNests, hatched, stormDat, storm, con=conf)

  # +> ---- observer: ---------------------------------------------------------
  obs     = observer(nData, par=par, surveys=survey, out=nd2, conf=conf)

  # +> ---- concatenate to make data for the nest models: ---------------------
  nestData = np.concatenate((nData, 
                 obs
              #  stormOut[0][:,None] # storms per nest
                 ), axis=1)
  # nestData[:,3] = (nestFate==0)
  # nestData[:,4] = (nestFate==2)
  # nestData[:,5] = sTilDisc
  # nestData[:,6] = disc1
  # nestData[:,12] = nestData[:,12] > 3
  # nestData[:,14] = 416 # exposure dayso
  # nestData[:,15] = nestData[:,2] - nestData[:,1]
  
  # np.savetxt("nestdata_afterflood.csv", nestData, delimiter=",")
  # np.save("nest_data.npy", nestData)
  # np.save(nestfile, nestData)
  return(nestData)

