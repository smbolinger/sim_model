#!/usr/local/bin/python
import sys
import numpy as np
import pprint
import pandas as pd
from itertools import groupby
from operator import itemgetter
import plotext as plt # plot ASCII plots in the terminal window
# from makeNests import mk_nests, mk_fates, mk_flood, storm_nest,mk_set_nests, mk_per
from makeNests import storm_nest
# from rsettings import config, rng, initFromFile
from helpers import expDecay, searchSorted2,print, mk_per,svy_position
from print_func import  arrPrint, print_prop, dfPrint
from dsrCalc import calc_dsr
# np.set_printoptions(precision=3, linewidth=120)

# -----------------------------------------------------------------------------
# ---- SURVEYS ----------------------------------------------------------------
# -----------------------------------------------------------------------------
def mk_surveys(stormDays, obsFreq, breedingDays, conf, rng, complicate=True, db=1):
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
  start       = rng.integers(1,high=5) # +> random 1st svy from 1st 5 br days      
  # start       = 1
  end         = start + breedingDays
  surveyDays  = np.arange(start, end, step=obsFreq)
  stormSurvey = []
  if(complicate):
    splits      = np.where(np.diff(stormDays)!=1)[0] +1 # print(f"{splits=}")
    storms      = np.split(stormDays, splits)
    if db>=4: print(f"\t\t{storms=}")
    if len(stormDays) > 0: #+> only if there are storms
      for s in storms:
        stormSurvey = np.where(np.isin(surveyDays,stormDays))[0] ##index of storm survey
        if len(stormSurvey)>0:
          if db>=4: print(f"\t\t{[surveyDays[i] for i in stormSurvey]=}")
          for i in stormSurvey: ## have to do this in tandem with next step
            surveyDays[surveyDays>=surveyDays[i]]+=2
          if db>=3: print(f"\t\tnew {surveyDays=}")
        lastDay = np.max(s)
        stormPos    = np.searchsorted(surveyDays, lastDay)
        if db>=3: print(f"\t\t{stormPos=}",end=" ")
        #+> left and right side give same result:
        # stormPosR    = np.searchsorted(surveyDays, lastDay, side="right")
        sDiff   =  surveyDays[stormPos] - lastDay #sDiff   = lastDay+2 - surveyDays[stormPos]
        if db>=3: print(f"\t\t{surveyDays[stormPos]=} - {lastDay=} "
                                   f"=\t\t{sDiff=}")
        if sDiff < 3:
          if sDiff < 2:
            if sDiff < 1: surveyDays[surveyDays >= lastDay] += 1
            surveyDays[surveyDays >= lastDay] += 1
          surveyDays[surveyDays >= lastDay] += 1
        else:
          surveyDays
          if db>=3: print(f"\t\tsDiff<2; {surveyDays=}")
  else:
    if db>=2: print("\t\t|> !!! complicate=False")
    # if conf.debugObs>=2: print("\t\t|> complicate=False; storms = double obs int")
  surveyDays  = surveyDays[np.isin(surveyDays, stormDays) == False]
  surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)) ])
  if db>=1:
    with np.printoptions(threshold=sys.maxsize):
      print(f"\n\t>>survey days, w/o storms: {surveyDays} ({len(surveyDays)})")
  if db>=1:
    with np.printoptions(threshold=sys.maxsize):
      print(f"\t>>{surveyInts=} ({len(surveyInts)})", end=" ")
      print(f"\t\t\t{np.sum(surveyInts>obsFreq)=}")

  if any(surveyInts > 12) and db < 1: print("\t***extra long obs int!")
  if any(surveyInts > 12) and db < 1:
    print("\t***extra long obs int! investigate:")
    mk_surveys(stormDays, obsFreq, breedingDays, conf, rng, complicate=True, db=4)
    # mk_surveys(stormDays, obsFreq, )
  return(surveyDays, surveyInts, stormSurvey)


# -----------------------------------------------------------------------------
# ---- NEST DISCOVERY & OBSERVATION ----------------------------------------
# -----------------------------------------------------------------------------
def observer(nData, par, rng,surveys, stormDays, out,stormUnk, conf):
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
      [columns = i, j, k, assigned fate, num obs int, intFinal, stormFinal]
    ---------
    NOTES
      Remember, pos[0] is the first survey after initiation, and pos[1] is the first survey after end.

  """
  # if conf.debugObs>=6:
    # print("nest data!")
    # print(f"{type(nData)=}")
    # arrPrint(nData, abval=20) # if conf.debugObs>=4:
  initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
  # print(f"{initiation=}")
  surveyDays, surveyInts,stormSurvey = surveys
  # print(f"{surveyDays=}")
  pos = svy_position(initiation, end, surveys[0], cn=conf)
  # print(f"{pos=}")
  tot_svy      = pos[1] - pos[0]  #+> num surveys while nest is active
  svysTilDiscovery = rng.negative_binomial(n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
  discovered     = svysTilDiscovery < tot_svy
  hatched        = nData[:,3] == 0
  num_obsTrue = tot_svy - svysTilDiscovery #num_svy = tot_svy - svysTilDiscovery -1 # -1 for final survey
  num_obsTrue[~discovered] = 0 #+> not discovered = 0 observations
  out[:,6] = num_obsTrue
  num_obs = num_obsTrue
  num_obs[~hatched] -= 1
  intFinal  = surveyInts[pos[1]] # +> actual length of final int for each nest
  iVal = surveyDays[pos[0]+svysTilDiscovery] # i
  kVal = surveyDays[pos[1]]
  jVal = surveyDays[pos[1]-1]

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.testing=="yes":
    if conf.debugObs>=6:
      print("nest data!")
      arrPrint(nData, abval=20) # if conf.debugObs>=4:
      print("\t\t\t\t>>>> init dates:",end=" ")
      arrPrint(nData[:,1], abval=20)
      print("\t\t\t\t>>>> position of initiation date in survey day list:",end=" ") 
      arrPrint( pos[0], abval=20)
      print("\t\t\t\t>>>> end dates:",end=" ")
      arrPrint(nData[:,2], abval=20)
      print("\t\t\t\t>>>> position of end date in survey day list:",end=" ")
      arrPrint( pos[1], abval=20) 
    if conf.debugObs>=4:
      print("\t\t\t\t>> survey days with index number:",end=" ")
      arrPrint( surveyDays,abval=20)
    if conf.debugObs>=3:
      print("\t\t\t|>num obs while active:", end=" ")
      arrPrint(num_obs, abval=20)
      print("\t\t\t|>total num surveys for each nest:", end=" ")
      arrPrint(tot_svy, abval=20)
      print("\t\t\t|> surveys til discovery = ", end=" ")
      arrPrint(svysTilDiscovery, abval=20)
    if conf.debugObs>=3:
      print(f"\t\t\t|> num discovered  = {sum(svysTilDiscovery < tot_svy)}", end=" ")
      print("\t\t\t|> nest discovered? (svysTilDiscovery < num_svy)")
      arrPrint(discovered, abval=20)
      # print(f"\t\t\t{intFinal=}")
    if conf.debugObs>=6:
      print("\t\t\t\tj & k before:")
      arrPrint(jVal, abval=20)
      arrPrint(kVal, abval=20)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  jVal[fate==0] = kVal[fate==0]
  out[:,0][discovered] = iVal[discovered]
  out[:,1][discovered] = jVal[discovered] 
  out[:,2][discovered] = kVal[discovered]
  finalInterv  = mk_per(out[:,1], out[:,2],con=conf)
  stormLoc = storm_nest(par.stormFrq, finalInterv, stormDays, conf)
  stormFinal = (stormLoc==1).any(axis=1) #+>storm in final interval?
  out[:,7] = stormFinal.astype(int)

  #+> need number of obs between i and j
  out[:,4] = num_obs 
  out[:,5] = intFinal.astype(int) # length of final interval - transform to integer for the ndarray
  longFinal  = out[:,5] > par.obsFreq
  # longFinal  = out[:,5] > par.obsFreq * 2
  # longFinal  = out[:,5] > np.round(par.obsFreq * 1.3)
  arrays = [stormFinal,longFinal,intFinal,discovered]
  # out[:,3] = assign_fate(par,rng,stormFinal,longFinal,fate,intFinal,stormUnk,conf)
  out[:,3] = assign_fate(par,rng,arrays,fate,stormUnk,conf)
  # out[:,3][discovered] = assign_fate(par,rng,sfin2,longFinal,fate2,ifin2,stormUnk,conf)

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.testing=="yes":
  #   if conf.debugObs>=5:
  #     print("\t\t\t\tj & k:")
  #     arrPrint(jVal, abval=20)
  #     arrPrint(kVal, abval=20)
    if conf.debugObs>=4:
      print(f"\t\t\t{iVal.astype(int)=} \n\t\t\t{jVal.astype(int)=} \n\t\t\t{kVal.astype(int)=}")
      print(f"assigned fate:")
      arrPrint(out[:,3], abval=20)
      print(f"final interval length:")
      arrPrint(out[:,5], abval=20)
  #   if conf.debugObs>=4:
  #     print(f"\t\t\t{np.where(out[:,3]==7)=}")
  #     print(f"\t\t\t{np.where(out[:,3]==2)=}")
    if conf.debugObs>=5:
      print(f"\t\t\t{stormLoc=}")
      print(f"\t\t\t{stormFinal=}{stormFinal.astype(int)=}")
      print(f"\t\t\t{np.where(stormFinal)=}{np.sum(stormFinal)=}")
      print(f"\t\t\t{np.where(longFinal)=}{np.sum(longFinal)=}")
  #   if conf.debugObs>=5:
  #     # print(f"\t\t\t{finalInterv=}") ##NOTE prints annoyingly
  #     print(f"\t\t\t{stormLoc=}")
  #     print(f"\t\t\t{stormFinal=}") ## not always caught if nest ends or begins on day of storm?
  #     print(f"\t\t\t{longFinal=}")
    if conf.debugObs>=4:
      print("\t\tout=")
      dfPrint(out)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return(out)

#-----------------------------------------------------------------------------
# @profile
     # [9]:len(final int)....[10]:num storms......[11]:num obs total....
# def make_obs(par,rng, obsVarNum, storm, survey, conf, initDat,stormUnk, nw=2, inff=True, pandas=False):
# @profile
def assign_fate( par, rng,arr,trueFate, stormUnk,cn):
  """
    Arguments: 
    assignVal = the value given for incorrect fates
  
    fateCuesProb=random values to compare
    fateCuesPres=probability of fate cues being present
      >-> created within the function using exp decay & final int length
    if fate percentages are fixed, fateCuesPres should be the same (=1) for all
  
    Returns: vector w/ assigned fate for each nest

    Notes: The observer assigns the correct fate based on comparison of a set 
    probability to random draws from a uniform distribution. The set probability
    is determined by exponential decay, so it decreases as more days pass.

    If a storm occurs, the rate of decay of evidence is higher, meaning
    more "ambiguous" nests. Ambiguous nests are classified based on 
    stormFate: if True, then assigned "flooded"; if False, then "unknown"

  """

  stormFin,longFin,intFinal,discovered = arr
  
  numNests = len(trueFate)
  assignedFate=np.empty(numNests)
  assignedFate.fill(7) # +> default=unk; fill w/ known fate if field cues allow

  ## probability of finding fate cues decays over time:
  fateCuesPres   = expDecay(n0=1, k=par.decayRate, t=intFinal)

  ## higher decay rate for nests with storm in final interval:
  fateCuesPres[stormFin] = expDecay(n0=1,k=par.decayStorm,t=intFinal[stormFin])
  fateProb = rng.uniform(low=0, high=1, size=par.numNests)
  wrongFateMask = fateProb > fateCuesPres
  correctFate = fateProb < fateCuesPres
  assignedFate[correctFate] = trueFate[correctFate] 

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if cn.debugObs>=5:
      print(f"\t\t\tCORRECT FATE: np.where( {fateProb=} < {fateCuesPres=}\n)"
            f"\t\t\t\t{(np.sum(fateProb<fateCuesPres))=}")
      print(f"\t\t\t {assignedFate.shape=} {assignedFate=}")
    if cn.debugObs>=2:
      print(f"\t\t{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=} |"
            f" {np.sum(assignedFate==0)=}"
            f"\n\t\t{np.sum(fateProb<fateCuesPres)=} | ")
      # with pd.option_context("display.max_columns",None):
    if cn.debugObs>=5:
      # print(f"{stormFin=}")
      print(f"\t\t\t{intFinal=} {len(intFinal)} ")#"\n\tcorrect={np.sum(intFinal<=par.obsFreq)=}")
      # print(f"\t\t\t{fateCuesPresent=}")
      # print(f"\t\t\t{fateProb=}")
    if cn.debugObs>=3:
      print(f"\t\t\t\t{np.where(fateProb<fateCuesPres)=}")
      # print(f"\t\t\t{fateProb<fateCuesPresent=}")
      print(f"\t\t\t{assignedFate[discovered]=} ;\n\t\t{assignedFate[discovered].shape=}")
      arrPrint(assignedFate, abval=20)

  #--account for storms-------------------------------------------------------
  #----------------------------------------------------------------------------
  tfd = trueFate[discovered]
  afd = assignedFate[discovered]
  if par.propMC>0 or par.propUnk>0:
    # assignedFate = add_misclass(par,rng,trueFate,assignedFate,cn,db=cn.debugObs)
    assignedFate[discovered] = add_misclass(par,rng,tfd,afd,cn,db=cn.debugObs)
    # if cn.debugObs>=2: print(f"\t>> mis-assigning fate based on proportions")
  else:
    ## add small amount of misclassified nests 
    if cn.addMC:
      assignedFate[discovered] = add_misclass(par,rng,tfd,afd,cn,db=cn.debugObs)

    if par.stormFate:
      assignedFate[stormFin] = 2
      assignedFate[longFin] = 2
    if stormUnk and not par.stormFate:
      assignedFate[stormFin] = 7
      assignedFate[longFin] = 7

  #----------------------------------------------------------------------------

  # NOTE fate cues prob should affect all nest fates equally, not just failures

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if cn.debugObs>=4:
      print(f"\t\t{trueFate[stormFin]=}")
      print(f"\t\t{np.where(trueFate[stormFin])=}")
      # stormFates = trueFate[stormFin]
      print(f"\t\t{trueFate[longFin]=}")
      print(f"\t\t{np.where(trueFate[longFin])=}")
      # print(f"\t\t{assignedFate[wrongFateMask & (stormFin | longFin)]=}")
      # print(f"\t\t{np.where(assignedFate[wrongFateMask & (stormFin | longFin)])=}")
      # stormFates = trueFate[longFin]
    if cn.debugObs>=2:
      print(f"\n\t\t.:.:{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=}"
            f" | {np.sum(assignedFate==0)=}:.:.")
      # print(f"\n\t\tfateProb less than this number = correct fate "
      #       f"(unless long/storm final interval):\n\t{fateCuesPresent=}")
      # print(f"\twrong if cues_not_present==True and either long or storm final")
    if cn.debugObs>=3:
      print(f"\nMARK UNKNOWN IF CUES NOT PRESENT/LONG FINAL/STORM IN FINAL INTERVAL:")
      df2print = pd.DataFrame({
        # "init": 
        "true_fate": trueFate,
        "prob_of_cues": fateCuesPres,
        "comparison_prob": fateProb,
        "cues_not_present": fateProb>fateCuesPres,
        "long_final_int": longFin,
        "storm_final_int": stormFin,
        "assigned_fate": assignedFate,
        })
      # with pd.option_context("display.max_columns",60,"precision",2):
      with pd.option_context("display.max_columns",60,
                             'float_format', '{:.2f}'.format):
        print(df2print.T)
      # print(f"\t\t\t {assignedFate.shape=}")
    # if cn.debugObs>=3: arrPrint(assignedFate, abval=20)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(assignedFate)

#-----------------------------------------------------------------------------

def add_misclass(par,rng,trueFate,assignedFate,cn,db=0):
  #+> has to be discovered nests!
  #NOTE don't reset seed each time; pass from main script as arg
  
  fval = [0,1,2] ## fval = all possible replacement vals to choose from
  hatchVal = [0]
  failVal = [1,2]
  uVal = 7

  if par.MCtype!="none":
    # val   = [0] if par.MCtype == "hatch2fail" else [1,2] # values to replace
    ## if using a preset value, all fates should be correct until misclassified
    # assignedFate = trueFate
    ## this should alreayd happen because decay rate is 0, but make it explicit
    val   = hatchVal if par.MCtype == "hatch2fail" else failVal # values to replace
    # mcVal =  [1,2] if par.MCtype == "hatch2fail" else 0  # replacement
    mcVal =  failVal if par.MCtype == "hatch2fail" else hatchVal  # replacement
    # fVal =  [1,2] if par.MCtype == "hatch2fail" else [0]
    # uVal = 7
    eqval = np.isin(trueFate,val) # where does fate == val?
    ## this way, even unknowns can become "mc" instead? no, shouldn't be any unknowns...
    nMisclass = int(np.round(par.propMC * np.sum(eqval)))
    nUnknown  = int(np.round(par.propUnk * np.sum(eqval)))
    mcReplace =   rng.choice(mcVal,size=nMisclass)
    # tot       = nMisclass + nUnknown
  else:
    # fval = [0,1,2]
    val = fval
    mcVal = fval
    # uVal = 7
    # eqval = assignedFate[assignedFate!=7] # any non-unknown nest
    eqval = assignedFate!=7 # any non-unknown nest
    # nMisclass = int(np.round(0.05*np.sum(eqval)))
    nMisclass = int(np.round(0.03*np.sum(eqval)))
    nUnknown = int(np.round(0.0*np.sum(eqval))) ## unknowns already happen
    mcReplace = [rng.choice(np.setdiff1d(mcVal, [v])) for v in range(nMisclass)]
    # tot       = nMisclass + nUnknown
    # val = rng.choice(fval, size=tot,replace=True)
  # val = rng.choice(fval, size=nMisclass,replace=True)
  # limit = np.max(fval)
  # limit = np.max(val)
  # add = rng.choice(np.arange(20), size=nMisclass, replace=True)
  # mcVal = (val + add) % limit ## add random number but loop w/in fval
  # mcVal =   
  tot       = nMisclass + nUnknown

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if db>=4: print(f"\t\t{par.MCtype=} {par.propMC=} {par.propUnk=}")
    # if db>=3: print(f"\t\t{eqval=}   {len(eqval)=} ")
    if db>=4:
      print(f"\t\t\t\t{eqval=}   {np.sum(eqval)=} ")
      print(f"\t\t\t\t", end=" ")
      eqvalStr = f""
      for ind,val in enumerate(eqval):
        # eqvalStr += f"{ind}:{val}\t"
        eqvalStr += f"{ind}:{val}  "
        # with np.printoptions(linewidth=100):
          # print(f"{ind}:{val} ", end=" ")
      pprint.pprint(eqvalStr, width=100)
    if db>=2: print("\t\t\t NUMBER TO MISCLASSIFY:", nMisclass, end=" ")
    if db>=2: print("\t\t\t NUMBER TO MARK UNKNOWN:", nUnknown, end=" " )
    if db>=2: print(f"\t\t\t\t{val=} , {mcVal=} , {uVal=} , {tot=}")
    # if db>=2: print(f"\n\t\t\t{mcReplace=}")

    # if db>=4: print(f"\t\t\tBEFORE <{len(assignedFate)}> : {assignedFate=}")

  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## NOTE for some reason, returns a tuple with 1 array instead of just an array

  ind = np.where(eqval) ##+> select random indices to replace at 
  if db>=4: print(f"\t\t\t{ind=} {len(ind)=}", end=" ")
  ind = ind[0] ##  since np.where returns multi-dim obj?
  mask = rng.choice(ind, size=tot, replace=False)
  if db>=4: print(f"\t{mask=}", end=" ")
  rep_vals  =[mcReplace, np.repeat(uVal,nUnknown)]
  rep_vals  = np.concatenate(rep_vals).tolist()
  if db>=4: print(f"\t{rep_vals=}")
  if db>=4: print(f"\tBEFORE:{assignedFate[mask]=}")
  assignedFate[mask] = rep_vals

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    # if db>=2: print(f"\n\t\t\t{mcReplace=}")
    if db>=4: print(f"\t\t\t{ind=} {len(ind)=}", end=" ")
    # if db>=4: print(f"\t{mask=}", end=" ")
    # if db>=5: print(f"\t{rep_vals=}")
    # if db>=4: print(f"\t\t\tAFTER: {assignedFate=}")
    if db>=4: print(f"\t\t\tAFTER: {assignedFate[mask]=}")

  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return assignedFate

#-----------------------------------------------------------------------------

def make_obs(nData,par,rng, obsVarNum, storm, survey, conf, initDat,stormUnk, nw=2, inff=True, pandas=False):
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
  nd2    = np.zeros(shape=(par.numNests, 8), dtype=np.int16)
  colnames = ['ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs']
  # if conf.debugObs>=3: print(f"\t\t{len(colnames)=}")

  # NOTE should I make sure all nests live for at least a day?

  # +> ---- observer: ---------------------------------------------------------
  # if conf.debugObs>=3: print("\n\t[*] [*] [*] [*] [*] observer [*] [*] [*] [*] [*] [*] [*] [*] ")
  if conf.debugObs>=2: print(
      f"\n\t[*] [*] [*] [*] [*] observer ({par.stormFate=}) [*] [*] [*] [*] [*] [*] [*] [*] ")
  # obs = observer(nData,par,rng,surveys=survey,stormDays=storm,out=nd2,stormUnk=stormUnk,conf=conf)
  obs = observer(nData,par,rng,survey,storm,nd2,stormUnk,conf)

  # +> ---- concatenate to make data for the nest models: ---------------------
  nestData = np.concatenate((nData, 
                 obs #stormOut[0][:,None] # storms per nest
                 ), axis=1)

  if pandas: nestData = pd.DataFrame(nestData,columns=colnames)
  ## last day nests are active = max init date plus hatch time
  # maxNestDay = np.max(nestData[:,1]) + par.hatchTime
  # maxSurveyDay = maxNestDay + par.obsFreq


  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.testing=="yes":
    if conf.debugObs>=3:
      # print_prop(nestData, 8)
      print_prop(nestData, obsVarNum)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return(nestData)

def calc_nests(nestData1, par,rng,survey, obsCol,repID, parID, config, db=0):
  # rng = np.random.default_rng(seed=config.rngSeed) print("calling calc_nests")
  obsCol = int(obsCol)
  surveyInt = survey[1]
  longest_int = max(surveyInt)

  flooded  = sum(nestData1[:,3]==2)
  hatched  = sum(nestData1[:,3]==0)
  discover = nestData1[:,obsCol]>0 ## where num obs > 0

  nestData = nestData1[(discover),:] # +> remove undiscovered nests
  flood_dsc  = sum(nestData[:,3]==2)
  hatch_dsc  = sum(nestData[:,3]==0)
  short    = (nestData[:,2]<nestData[:,4]) ## where end<i
  unknown  = (nestData[:,7]==7)
  exclude = (unknown.astype(int) + short.astype(int)) > 0 # at least one is true
  if db>=5: print(f"\t\t\tcalc_nests: {exclude=}")
  misclass = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests

  nestData = nestData[~exclude,:] # +> remove undiscovered nests
  flood_an  = sum(nestData[:,3]==2)
  hatch_an  = sum(nestData[:,3]==0)
  misclass2 = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests
  avgFInt  = (nestData[:,9].sum()/len(discover))
  sNest    = nestData1[:,11].sum()
  avgK     = nestData[:,6].sum()/len(discover)
  maxI     = np.max(nestData[:,4])
  # srand = rng.uniform(0.00, 10.00) # +> random init val for MARK
  appDSR  = calc_dsr(nData=nestData1,
                      nestType="all",
                      calcType="apparent",
                      conf=config,
                      incTime=par.hatchTime,
                      psurv=par.probSurv,
                      debug=config.debugDSR)
  appPSR = appDSR ** par.hatchTime
  mayfDSR_an   =  calc_dsr(nData=nestData,
                           nestType="analysis",
                           calcType="mayfield",
                           conf=config,
                           incTime=par.hatchTime,
                           psurv=par.probSurv,
                           debug=config.debugDSR) 
  appDSR_an   = calc_dsr(nData=nestData,
                          nestType="analysis",
                          calcType="apparent",
                          conf=config,
                          incTime=par.hatchTime,
                          psurv=par.probSurv,
                          debug=config.debugDSR) 
  nestVals = np.array([
    parID,repID,flooded,hatched,flood_dsc,hatch_dsc,flood_an,hatch_an,
    sNest,discover.sum(),exclude.sum(),unknown.sum(),
    misclass.sum()-unknown.sum(),misclass2.sum(),avgFInt,avgK,maxI,longest_int,appDSR,appPSR,mayfDSR_an,appDSR_an])
  if db>=5: print(f"\t\t{nestVals=}")
  return nestVals
