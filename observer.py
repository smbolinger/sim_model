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
  if conf.debug>=2:
    print(f"\t\t-> {stormDays=}")
    print("\t\t>-->survey days before alteration:")
    arrPrint(surveyDays)
    # d = {ind: v for ind,v in enumerate(surveyDays)}
    # print(f"{d}")
    print(f"\t\t{storms=}")
  for s in storms:
    lastDay = np.max(s)
    stormPos    = np.searchsorted(surveyDays, lastDay)
    sDiff   =  surveyDays[stormPos] - lastDay
    # sDiff   = lastDay+2 - surveyDays[stormPos]
    if conf.debug>=2:
      print(f"\t|>{s=}", end=" ")
      print(f"\t|>{surveyDays[stormPos]=}", end=" ")
      # print(f"\t|>{(lastDay)=} ; {sDiff=}", end=" ")
      print(f"\t|> add {int(sDiff)} to survey days >= {(int(lastDay))} ")
      # print(f"\t|> ")
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
  if conf.debug>=2:
    print("\n\t\t>-->survey days after alteration:") # NOTE: need \n bc of prev
    arrPrint(surveyDays)
  # surveyDays  = [s + 2 if s > x ]
  # survey interval for first obs is 0:
  surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
  # surveyInts  = np.append(surveyInts, )
  if conf.debug: 
    print(f"\t\t>-> all survey days, minus storms (len {len(surveyDays)}):")
    # indPrint(surveyDays) 
    arrPrint(surveyDays) 

  return(surveyDays, surveyInts)

def mk_per(start, end, con):

  nestPeriod = np.stack((start, end)) # +> create array of tuples
  # NOTE need the double parentheses so it knows output is tuples
  nestPeriod = np.transpose(nestPeriod) # +> an array of start,end pairs 
  if con.debugNests>=5:
    print( f"\t\t\t>> start & end of nest period:\n")
    arrPrint(nestPeriod)
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
  # if cn.debugObs>=3: 
  #   timevals = np.arange(0,10)
  #   probvals = expDecay(n0=1, k=0.1, t=timevals)

    # print("\n\trange of prob vals for plot:",np.min(probvals), np.max(probvals))
    # fig = plt.plot(probvals,timevals) # maybe can't assign a plotext plot to an object?
    # plt.clt()
    # plt.clear_color()
    # plt.clf()
    # plt.plot(probvals, timevals)
    # # fig.savefig("figs/expDecay.png")
    # plt.show() # syntax for plotext is almost identical to matplotlib syntax
    #
  if cn.debugObs>=4:
    print("\t\t\t|> probability of fate cues:")
    arrPrint(np.round(fateCuesPresent,3))
  if cn.debugObs>=5:
    print("\t\t\t|>& final int (for comparison):")
    arrPrint(intFinal)

  fateProb = rng.uniform(low=0, high=1, size=numNests)
  if cn.debugObs >=4:
    print("\t\t\t|>random probs for fate:")
    arrPrint(np.round(fateProb,3))
  assignedFate[fateProb < fateCuesPresent] = trueFate[fateProb < fateCuesPresent] 

  if cn.debugObs>=2:
    print("\t\t\t>-> true fates (all nests, not just discovered):")
    arrPrint(trueFate)
    print("\t\t\t>-> assigned fates before (all nests, not just discovered):")
    arrPrint(assignedFate)
  if stormFate: assignedFate[intFinal > obsFr] = 2
  if cn.debugObs>=2:
    print(
        f"\t\t>=>=> true fate counts (proportion):"
        f"\tH:{sum(trueFate==0)}({sum(assignedFate==0)/numNests})|"
        f"D:{sum(trueFate==1)}({sum(assignedFate==1)/numNests})|"
        f"F:{sum(trueFate==2)}({sum(assignedFate==2)/numNests})"
        )
    print("\t\t\t>-> assigned fates after incorrect fates assigned(all nests):")
    arrPrint(assignedFate)
    
  if cn.debugObs>=4: 
    print("\t\t\t>-> compare random probs to fateCuesPresent:\n") 
    print("\t {[np.round(fateProb[f],3):np.round(fateCuesPresent[f],3) for f in 1:numNests]}")

    # print(f"\t>-> or to pWrong: {pWrong} with fill value: {assignVal}")
  if cn.debugFlood>=4:
    print("\t\t\t>-> nests with storm in final interval:", np.where(intFinal>obsFr))
    print("\t\t\t>-> storm fate == True?", stormFate)
  # NOTE fate cues prob should affect all nest fates equally, not just failures
  return(assignedFate)

#-----------------------------------------------------------------------------

def svy_position(initiation, nestEnd, surveyDays, cn):
  """ Finds index in surveyDays of iniatiation and end dates for each nest """
  position = np.searchsorted(surveyDays, initiation) 
  if cn.debugObs>=5:
    print("\t\t>> initiation dates:\n")
    arrPrint(initiation)
    print("\t\t>>>> position of initiation date in survey day list:\n") 
    arrPrint( position)
    print("\t\t>> end dates:\n")
    arrPrint(nestEnd)
  position2 = np.searchsorted(surveyDays, nestEnd)
  surveyDays = dict(zip(np.arange(len(surveyDays)), surveyDays))
  if cn.debugObs>=5:
    print("\t\t>>>> position of end date in survey day list:\n", position2, len(position2)) 
    print("\t\t>> survey days with index number:\n", surveyDays)
  
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
  print("\n\t\t[*] [*] [*] [*] [*] observer [*] [*] [*] [*] [*] [*] [*] [*] ")
  if conf.debugObs>=5:
    print("nest data!")
    arrPrint(nData)
  initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
  surveyDays, surveyInts = surveys

  pos = svy_position(initiation, end, surveys[0], cn=conf)
  num_svy      = pos[1] - pos[0]   
  if conf.debugObs>=4:
    print("\t|> num surveys for each nest:")
    arrPrint(num_svy)
  svysTilDiscovery = rng.negative_binomial(n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
  discovered     = svysTilDiscovery < num_svy
  if conf.debugObs>=3:
    print("\t\t\t|> nest discovered? (svysTilDiscovery < num_svy)")
    arrPrint(discovered)
  num_svy[~discovered] = 0
  if conf.debugObs>=4:
    print("\t\t\t|> surveys til discovery:")
    arrPrint(svysTilDiscovery)
    print("\t\t\t> total surveys while nest active:")
    arrPrint(num_svy)

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
  if conf.debugObs>=3:
    trueFate = nData[:,3]
    assignedFate = out[:,3][discovered]
    print(
        f"\t\t=>=> assigned fates - discovered (proportion):"
        f"\tH:{sum(assignedFate==0)}({sum(assignedFate==0)/sum(discovered)})|"
        f"D:{sum(assignedFate==1)}({sum(assignedFate==1)/sum(discovered)})|"
        f"Fl:{sum(assignedFate==2)}({sum(assignedFate==2)/sum(discovered)})"
        )
    print("\t\t\t>-> true fates (discovered nests):")
    arrPrint(trueFate[discovered])
    print("\t\t\t>-> assigned fates (discovered nests):")
    arrPrint(assignedFate)
  if conf.debugObs>=1: 
    print("\t\t>-> number discovered: ",
          np.sum(discovered==True),
          "\t\t>-> disc prob: ",
          par.discProb,
          )
    print("\t\t\t>-> assigned fates:")
    arrPrint(assignedFate)
    assHatch = ((out[:,3])==0)[discovered==True]
    nonUnk   = ((out[:,3])!=7)[discovered==True]
    trHatch  = ((nData[:,3]==0))[discovered==True]
    prop = np.sum(assHatch)/(np.sum(nonUnk))
    prop2 = np.sum(trHatch)/(np.sum(discovered==True))
    print(
      f"\t\t>> proportion non-unk nests assigned hatch fate"
      f" ({np.sum(assHatch)} / {np.sum(nonUnk)}):",
      np.round(prop,5),
      # np.sum(((out[:,3])==0)[discovered==True])/(sum(discovered==True)),
      # np.sum(((out[:,3])==0)[discovered==True])/(np.sum((out[:,3]!=7)[discovered==True])),
      "vs. period survival:",
      np.round(par.probSurv**par.hatchTime,5),
      f"vs. proportion of all nests: {prop2}"
      )
  if conf.debugObs==3 or conf.debugObs==2: 
    # trueFate = round(nData[:,3])
    trueFate = nData[:,3]
    assignedFate = out[:,3][discovered]
    print(
        "\t\t|>surveys til discovery; discovered T/F; total obs days;"
        " total active days; assigned fate; true fate:")
    active = nData[:,2]-nData[:,1]
    obsLen = (out[:,2]-out[:,0])
    for i in range(5):
      # print(f"\t\t\t{i:02}: {svysTilDiscovery[i]} | {discovered[i]} | {(out[:,2]-out[:,0])[i]}:03 | {(nData[:,2]-nData[:,1])[i]}")
      print(
          f"\t\t\t{i:03} : {svysTilDiscovery[i]:02} | {discovered[i]:>5} | {(obsLen[i]):>4} |"
          # f" {((nData[:,2]-nData[:,1])[i]):>4} | {assignedFate[i]} | {trueFate[i]}"
          f" {round(active[i]):>4} | {assignedFate[i]} | {round(trueFate[i])}"
          )
    for i in range(-5,0):
      print(
          f"\t\t\t{i:03} : {svysTilDiscovery[i]:02} | {discovered[i]:>5} | {(obsLen[i]):>4} |"
          f" {round(active[i]):>4} | {assignedFate[i]} | {round(trueFate[i])}"
          )
  if conf.debugObs>=3: 
    trueFate = nData[:,3]
    assignedFate = out[:,3]
    active = nData[:,2]-nData[:,1]
    obsLen = (out[:,2]-out[:,0])
    print(
        f"\t\t =>=> assigned fates - total (proportion):"
        f"\tH:{sum(assignedFate==0)}({sum(assignedFate==0)/par.numNests})|"
        f"D:{sum(assignedFate==1)}({sum(assignedFate==1)/par.numNests})|"
        f"F:{sum(assignedFate==2)}({sum(assignedFate==2)/par.numNests})"
        )
  if conf.debugObs>=4:
    print(
        "\t\t|>surveys til discovery; discovered T/F; total obs days;"
        " total active days; assigned fate; true fate:")
    for i in range(len(out)):
      print(
          f"\t\t\t{i:03} : {svysTilDiscovery[i]:02} | {discovered[i]:>2} | {(obsLen[i]):>4} |"
          f" {round(active[i]):>4} | {assignedFate[i]} | {round(trueFate[i])}"
          )
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
  if conf.debugNests>=4: print("\t\t|>hatched (before storms)=", hatched, sum(hatched))
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
  
  ndString = "\t\t\tID--init-end-fate---i---j---k---afate-nstm-fInt"
  if conf.debugSummary>=2: print(f"\nnestData:\n{ndString}\n", nestData[0:5,:], "\n. . . . . . \n", nestData[-5:,:])
  if conf.debugSummary>=3: print(f"\nnestData:\n{ndString}\n", nestData)
  # np.savetxt("nestdata_afterflood.csv", nestData, delimiter=",")
  # np.save("nest_data.npy", nestData)
  # np.save(nestfile, nestData)
  return(nestData)

