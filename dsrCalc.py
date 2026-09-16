from datetime import datetime
import pandas as pd
import numpy as np
from MCmatrix import logistic
# from notr_MCmatrix import logistic
from observer import svy_position
from print_func import arrPrint, dfPrint
# from settings import config
# from rsettings import config
import itertools
from helpers import centerDat, print



# np.set_printoptions(precision=5, legacy='1.25', linewidth=120)
np.set_printoptions(precision=5, legacy='1.25', linewidth=999)
# debug = config.debug

def calc_exp(inp, cn, expPercent=0.5, debug=0): 
  """
    Calculate the exposure period for a nest (number of days observed)
      For each nest: 
        - known alive days plus estimate of alive days in final interval
        > alive days (before final int) + final int * expPercent
    ---------
    ARGUMENTS
      :param inp: = [ i,j,k] for all nests in set\n
      default expPercent is from Mayfield; Johnson recommended 0.4
    -------
    RETURNS
      ndarray. nrows=len(inp); cols=alive_days, final_int, exposure
    ----------
    MORE INFO
      - the ijk values should tell you failed vs hatched

    For the basic case where psurv is constant across all nests and times:
      1. count the total number of alive days when nest was observed
      2. count the number of days in the final interval (for failed nests)
      3. calculate the exposure
        - days obs before final int + (final int * expPercent)
      *expPercent* = percent of final interval nest is assumed alive
        - Mayfield used 50%, Johnson corrected it to 40%
        - final interval assumed to be 0 days for hatched nests, which
           were found after hatch (exposure of incubation period is over)
        - no nestling exposure bc precocial/semi-precocial chicks
          leave the nest so early 

    *NOTES* added debug function so it only prints outside optimizer
      > I think I couldn't get it to work as vectorized, so I used a loop
    ----------
  """
  # if cn.debugM>=2:
  #   np.save("out/inp.npy", inp)
  expo = np.zeros((len(inp), 3))
  # print(f"\t\t{expo.shape[0]=}")
  # if debug>=2:
  #   print("\t\t\t\tINP:  |> i:", end=" ")
  #   arrPrint(inp[:,0],abbr=False)
  #   print("\t\t\t\t\t\t\t|> j:", end=" ")
  #   arrPrint(inp[:,1],abbr=False)
  #   print("\t\t\t\t\t\t\t|> k:", end=" ")
  #   arrPrint(inp[:,2],abbr=False)
  #   # print("\t\t\t\t|>inp:")
  #   # arrPrint(inp[0:5,:], ind=8)
  #
  # for n in range(len(inp)-1): # want n to be the row NUMBER
  for n in range(len(inp)): # want n to be the row NUMBER
    #+> interval from first found - last active 
    #   +> all nests are KNOWN to be alive
    expo[n,0] = inp[n,1] - inp[n,0]
    # expo[n,0] = expo[n,0] - 1 # since this is essentially 1-day intervals, 
                  # need 1 fewer than total number? no?
    #+>interval from last active - last checked
    expo[n,1] = inp[n,2] - inp[n,1]
    # expo[n,1] = expo[n,1] - 1
    # +>if expo[n,1]!=0: expo[n,1] = expo[n,1]- 1 ; for hatched nests, stays 0

    # +>exposure = sum(alive days) + days in final int * expPercent
    expo[n,2]   = expo[n,0] + (expo[n,1]*expPercent)
    # NOTE need nests to be alive for at least one interval
  # if debug>=2: 
  #   print("\t\t\t\tEXPO:  |> aliv:", end=" ")
  #   arrPrint(expo[:,0],abbr=False)
  #   print("\t\t\t\t\t\t\t|> final int:", end=" ")
  #   arrPrint(expo[:,1],abbr=False)
  #   print("\t\t\t\t\t\t\t|> exposure:", end=" ")
  #   arrPrint(expo[:,2],abbr=False)
  #   print("\t\t\t\tEXPO: |> alive days:", expo[:,0].T)
  #   print("\t\t\t\t\t\t\t|> final int:", expo[:,1].T)
  #   print("\t\t\t\t\t\t\t|> exposure:", expo[:,2].T)
  #   print("\t\t\t\t|>expo:")
  #   arrPrint(expo[0:5,:], ind=8)
  # if cn.debugM>=2:
  #   np.save("out/exposure.npy", expo)
  return(expo)
#-----------------------------------------------------------------------------

def calc_dsr(nData, nestType, calcType, conf, incTime=0, psurv=0, debug=0):
  """ 
    Calculate exposure and DSR for a given set of nests. 
    ------

    - calculate daily mortality rate using Mayfield:
      if calc type == 'true':
        'exposure' = end - init
         may cause bias in 'true' val

      else:
        calculate with actual exposure days
        pass i,j,k from ndata to calc_exp()

    - pass exp and num_fail to mayfield()
    
    -----
    Returns DSR value (1-DMR). 
    -----
    NOTES
    Use debug argument so it only prints when not optimizing
    calc_exp doesn't work for all nests since many weren't discovered
  """

  if isinstance(nData,pd.DataFrame):
    nData = nData.to_numpy()
  nNests  = len(nData)
  # if calcType=="mayfield":
    ## +> pass i,j,k to calc_exp
    # expDays = calc_exp(nData[:,4:7], expPercent=0.4, cn=conf, debug=debug)
  # hatched = len(nData[:,3] == 0)
  # failed  = nNests-hatched
  
  # expDays = exposure(nestData[:,6:9], numNests=numN, expPercent=0.4)
  # expDays = calc_exp(nData[:,6:9], expPercent=0.4)
  # if nestType=="all":
  ## +> calc type 'true' means ???
  # if debug>=2: print(f"{nestType=} | {calcType=} |> ", end="   ")
    # if conf.debugDSR>=4:
    #   print(
    #       f"\t\t>> calculate mayfield DSR: {expDays=:.3f}|{nNests-hatched=}|"
    #       f"{dmr=:.3f}|{1-dmr=:.3f} "
    #       # f"| expected DSR={psurv}"
    #       )
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if calcType=="apparent" or calcType=="all":
    # if debug>=3: 
    #   print("\t> exposure days calc type = 'true'")
    #   print("\t> calculating exposure days from all nests")

    allDays = sum((nData[:,2]-nData[:,1]))
    avgExp  = allDays/nNests
    hatched = sum(nData[:,3] == 0)
    apparent = hatched/nNests
    appDSR   = 1-((nNests-hatched)/allDays) ## +> num failures/total days

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if conf.testing=="yes":
      if conf.debugDSR>=4:
        print(
              f"\t\t>> calculate apparent DSR: (1-(true_num_fail/total_days)):{appDSR:.3f} "
              f"\t\t1 - (({nNests}-{hatched}) / {allDays}) = {appDSR:.3f} "
    #           # f"| expected DSR: {psurv}"
              )
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if debug>=3: 
      # print(
            # f"\n\t\t\tapparent nest success s (hatched/total): {apparent:.3f} "
            # f"| expected PSR: {psurv ** incTime:.3f}"
            # f"| nrows of nest data: {nData.shape[0]}"
            # )
      # for n in range(5):
        # print(f"\t\t\t\texposure days: {nData[n,2]} - {nData[n,1]}")
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return(appDSR)
  # else:
  elif calcType=="mayfield":
    expDays = calc_exp(nData[:,4:7], expPercent=0.4, cn=conf, debug=0)
    expDays = expDays[:,2].sum()
    hatched = sum(nData[:,7] == 0)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if conf.testing=="yes":
      if conf.debugDSR>=2:
        # print(f"\n\t\t\t|>{nestType=}|{calcType=}|{incTime=}|{psurv=}|>", end=" ")
        print(f"\t\t>> calculate mayfield DSR: {expDays=:.3f}|{nNests-hatched=}|")
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dmr   = mayfield(num_fail=nNests-hatched, expo=expDays)

    ## +> now mayfield function prints instead..
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if conf.testing=="yes":
      # if conf.debugDSR>=4:
      #   print(
      #       f"\t\t>> calculate mayfield DSR: {expDays=:.3f}|{nNests-hatched=}|"
      #       f"{dmr=:.3f}|{1-dmr=:.3f} "
      #       # f"| expected DSR={psurv}"
      #       )
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return(1-dmr)
  # else:


# if debug: 
  ### duhhh, turn on if-else statements by defining a deparate dbug version,
  # BUT still has same issue as separate debug file (need to keep both updated)

def mayfield(num_fail, expo):
  """ 
    The Mayfield estimator of DSR 
    
    Mayfield's original estimator was defined as: 
        > DSR = 1 - (# failed nests / # exposure days)
    so if DSR = 1 - daily mortality, then:
        > daily mortality = # failed nests / # exposure days
    
    Arguments:
      num_fail = count of failed nests (total-hatched)
      expo   = sum of exposure days output of calc_exp() (out[:,2])

    Returns: the daily mortality 
    Note: I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
    expo needs to be a SUM
  """
  # print(expo, type(expo))
  # mayf = num_fail / (expo.sum())
  mayf = num_fail / (expo) # expo is already the sum?
  # if cn.debugM:
    # print(f"\t\t> mayfield DSR = ({num_fail=}) / ({expo=}) = {mayf}")
    # print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

  return(mayf)

#---------------------------------------------------------------------------------
# @profile
# def calc_daily_expo(numNests,surveyInts,surveyDays,firstDay,lastDay,config):
def calc_daily_expo(numNests,survey,firstDay,lastDay,config):
  """
    Calculate the daily exposure of each nest AND all survey days it's active
    ----
    ARGS:
      
    RETURNS:
      a list containing:
      1. a list of all exposure days across all nests
      2. a list of all survey days across all nests
  """
  # svyInd = svy_position(init, end, surveyDays)
  # print(f"{surveyInts=}")
  surveyDays, surveyInts = survey[0], survey[1]
  # db=config.debugLogEx
  db = config.debugDSR
  svyInd = svy_position(firstDay, lastDay, surveyDays,cn=config)
  initPos, endPos = svyInd
  # print(f"\t\t{initPos=} ; \n\t\t{endPos=}")
  # print(f"\t\t{surveyInts=}")

  # expo = [surveyInts[initPos[n]:endPos[n]] for n in range(numNests)]
  expo = []
  sdays = []
  # if(db>=3): print(f"\n\tSurvey ints that are zero: {surveyInts[surveyInts==0]}")
  for n in range(numNests):
    # NOTE becomes a list of arrays:
    # expo.append(surveyInts[initPos[n]:endPos[n]]) #+> probably slow
    #+> extend flattens the added arrays
    expo.extend(surveyInts[(initPos[n]+1):endPos[n]+1].tolist()) #+> probably slow
    # expo.extend(surveyInts[(initPos[n]):endPos[n]].tolist()) #+> probably slow
    sdays.extend(surveyDays[(initPos[n]+1):endPos[n]+1].tolist()) #+> probably slow
    # sdays.extend(surveyDays[(initPos[n]):endPos[n]].tolist()) #+> probably slow
    
    # expo.append(surveyInts[(initPos[n]+1):endPos[n]+1].tolist()) #+> probably slow

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if db>=3:
    # print(f"{numNests=} ")
    print(
        f"\t\t{firstDay=}\n {lastDay=} "
        f"\n\t\t{len(initPos)=} {initPos=} "
          f"\n\t\t{len(endPos)=} {endPos=}"
          )
    print(f"\t\t{len(surveyDays[initPos])=} {surveyDays[initPos]=} "
          f"\n\t\t{len(surveyDays[endPos])=} {surveyDays[endPos]=}"
          )
    print(f"\t{len(expo)=} {expo=} ")
    print(f"\t{len(sdays)=} {sdays=} ")
  # if db>=3:
    # print(f"\t{surveyInts[initPos+1:endPos+1]=} ")
    # print(f"\t{len(sdays2)=} {sdays2=} ")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ## These don't work:
  # sdays2 = np.concatenate([surveyDays[initPos:endPos]])
  # if db>=3: print(f"\t{len(sdays2)=} {sdays2=} ")
  # expo2 = np.concatenate([surveyInts[(initPos+1):(endPos+1)]])
  # if db>=3: print(f"\t{len(expo2)=} {expo2=} ")

  # return expo
  return [np.array(expo), sdays]

def make_daily_logex_df(obsData,
                        nObs,
                        # expos,
                        expoList,
                        survey,
                        config,
                        # covar1,
                        # saveDF=False,
                        alldiff=True,
                        zeroInd=True,
                        multiFate=False,
                        pandas=False,

                        exp1=False,
                        # ctr=False,
                        
                        # db=0,
                        ):
  """
    idate = date of initial obs
    leave alldiff = True and get both columns
    exp1 is for calculating true nest survival (exposure=1)
    obsData = ID, init, i, j, k, afate
    nObs = num obs (varies depening on which DSR is being calculated)

    calls make_df 
    ----
    RETURNS:
      pandas dataframe with 7 columns
        [ID, surv, expo, avDate, Date, avAge, Age]
  """
  #NOTE 04-Apr: getting errors about length of column replacements,
  #NOTE   but only for stormFate==True
  # db=config.debugLogEx
  db = config.debugDSR
  # if db>=2: print("\t>-> making daily obs df",end=" ")
  # if db>=3: print("\t\t>-> making daily obs df")
  # cols = ["id", "survive", "exposure", "idate", "date"]
  # cols = ["Nest.ID", "Surv", "Exposure", "ffDate", "Date", "Age", "propInit"]
  # cols = ["Nest.ID", "Surv", "Exposure", "avDate", "Date","avAge", "Age"]
  cols = ["Nest.ID","Surv", "Exposure", "avDate", "Date","avAge", "Age", "aFate"]

  ## CONVERT TO NUMPY ARRAY IF NOT ALREADY:
  if isinstance(obsData, pd.DataFrame):
    # if db>=3: print(f"\n\t\t\t{type(obsData)=}", end=" ")
    # if db>=3: print(f"\t\t{obsData=}")
    obsData = obsData.to_numpy()
  elif not isinstance(obsData, np.ndarray):
    # if db>=3: print(f"\n\t\t\t{type(obsData)=}", end=" ")
    # if db>=3: print(f"\t\t{obsData=}")
    obsData = np.array(obsData)
  # svyDay, svyInt = survey[1:2]
  # svyDays, svyInts,stormSvy = survey
  # if db>=4: print(f"\t\t{svyDay=}\n\t\t{svyInt=}")
    
  # if db>=4:
    # print(f"\t\t\t{type(obsData)=}\n\t\t{obsData=}")
    # print(f"\t\t >>> \t\t{obsData.shape=} {type(obsData)=}")
    # dfPrint(obsData, names=cols)
    # print(f"\t\t{cols=}")
  # nestID, init,ff, la, lc, afate, nObs = nestData.T
  # nestID, init,end,fate,ff, la, lc, afate = nestData.T
  ID, init,end,tfate,ff, la, lc, afate = obsData.T
  
  # if db>=4: print(f"\t\t|>{ID=}")
  nNest = obsData.shape[0]
  # nObs = nObs.astype(int) if exp1 else 
  # if db>=4: print(f"\t\t{np.sum(nObs)=} ; {nObs=}")
  if exp1:
    first,last,fate = init,end,tfate
  else:
    first,last,fate = ff,la,afate
  # nObs[afate!=0] +=1
  # print(f"nrows = {np.sum(nObs)=} ; {nObs=} ")
  # if alldiff:
  #   mat[:,3] = covar1
  # else:
  #   mat[:,3] = np.repeat(covar1, nObs)
  # obsDay    = svyDay[(svyDay>=first)&(svyDay<=last)]
  # expos    = np.diff(obsDay)
  # expos = [expo1, expo2]
  # if db>=4: print(f"\t\t{obsDay=}\n\t\t{expos=}")
  if multiFate==False: # +> make all failures "0"
    # afate = np.array([0 if i in [1,2,7] else 1 for i in afate])
    fate = np.array([0 if i in [1,2,7] else 1 for i in fate])

  # expoList   = calc_daily_expo(numNests=nNest, surveyDays=svyDays,
  #                                  surveyInts=svyInts, firstDay=first,
  #                                  lastDay=last, config=config,)
  # mat = make_df(ID,init,first,last,fate,nObs,nNest,expos,covar1,cols,db=db)
  # mat = make_df(ID,init,first,last,fate,nObs,nNest,expos,obsDay,cols,db=db)
  mat = make_df(ID,init,first,last,fate,nObs,nNest,expoList,cols,db=db)
  # mat = make_df(ID,init,first,last,fate,nObs,nNest,survey,cols,db=db)
  dfNew = pd.DataFrame(mat, columns=cols)

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if db>=4: print(f"\tpassed to make_df:")
  if db>=4: print(f"\t\t|>{first=}\n\t\t|>{last=}\n\t\t|>{fate=}")
  if db>=3: print(f"\t\t{mat.shape=}")
  if db>=4: print(f"\t\tBEFORE: {dfNew.shape=}, AFTER:", end=" ")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  dfNew['log_expo'] = np.log(dfNew['Exposure'])
  # cols = ["Nest.ID", "Surv", "Exposure", "avDate", "Date","avAge", "Age", "log_expo"]
  cols = ["Nest.ID", "Surv", "Exposure", "avDate", "Date","avAge", "Age", "aFate", "log_expo"]

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if db>=3:
    print(f"\t\t{dfNew.shape=}")
    dfPrint(dfNew, names=cols)
    print(f"\t\t\t{dfNew=}")
  if db>=4: dfPrint(dfNew)

  # NOTE save df in outer function
  # if saveDF:
  #   fn = f"_{parID:04}_{repID:03}.npy"
  #       prOut = Path(odir/f"pred{config.rngSeed}"/prFile)
  #       prOut.parent.mkdir(parents=True, exist_ok=True)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return dfNew

# def make_df(ID,init, first, last, fate, nObs, nNest, expos, covar, cols, db=0):
# def make_df(ID,init, first, last, fate, nObs, nNest, expos, obsDay, cols, db=0):
def make_df(ID,init, first, last, fate, nObs, nNest, expo, cols, db=0):
# def make_df(ID,init, first, last, fate, nObs, nNest,survey, cols, db=0):
  """
  PURPOSE
    calculate exposure, age, & date covariates for nest data

    for true DSR:
      first=init, last=end, fate=true fate,nObs=total days
    for obs DSR:
      first=i, last=k, fate=assigned fate, nObs=total observations
  """
  # nrows = np.sum(nObs)
  # svyDay, svyInt,stormSvy = survey
  nrows    = int(np.sum(nObs))
  initDay  = np.repeat(init, nObs)
  ageStart = first-init 
  ageEnd   = last-init
  avAge    = (ageEnd+ageStart)/2 
  # obsDay    = np.repeat(0,nObs)
  # obsDay    = svyDay[(svyDay>=first)&(svyDay<=last)]
  # expos    = np.diff(obsDay)
  expos,obsDay=expo
  expos,obsDay=expo

  #   print(f"\t\t\t{nrows=}{type(nrows)=}")
  #   print(f"\t\t\t{avAge=}")
  #   print(f"\t\t\t{initDay=}")
  # expo1, expo2=expos

  # if zeroInd:
  endDay = np.cumsum(nObs) -1 #+> zero-indexed
  # else:
    # endDay = np.cumsum(nObs) 
  endDay = endDay.astype(int)
  # if db>=3: print(f"\t\t\t{len(endDay)=} {endDay=}")

  mat = np.ones((nrows, len(cols) ))
  # if db>=3: print(f"\t\t{mat.shape=} | {len(expos)=}")
  # mat[:,0] = np.repeat(range(nNest), nObs) #+> repeat ID nObs times
  mat[:,0] = np.repeat(ID, nObs) #+> repeat ID nObs times
  mat[:,1][endDay] = fate ## nest status is 1 unless failed on last check
  ## where did I change the coding of final fate??
  mat[:,2] = expos
  # mat[:,2] = expo2
  # mat[:,3] = np.repeat(ff, nObs)
  mat[:,3] = np.repeat((first+last)/2, nObs) # avg observation day
  mat[:,4] = obsDay ## observation day
  mat[:,5] = np.repeat(avAge,nObs)
  mat[:,6] = obsDay - initDay
  mat[:,7] = np.repeat(fate,nObs)

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if conf.testing=="yes":
    # if db>=5: print(f"\t\t{svyDay=}\n\t\t{svyInt=}")
  if db>=2: print(f"\t\t\t{nObs=}{type(nObs)=}")
  if db>=2: print(f"\t\t\t{nrows=}{type(nrows)=}")
  if db>=5: print(f"\t\t\t>>make_df:{mat.shape=} \n\t{mat=}")
  if db>=4: print(f"\t\t\t>>make_df:{fate=} {len(fate)}")
  if db>=4: print(f"\t\t\t>>make_df:{expos=}{len(expos)}\n\t\t\t{obsDay=}{len(obsDay)}")
    # if db>=4: dfPrint(mat,nprint=20,names=cols)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return mat

def calc_nests(nestData1, par,rng,survey, obsCol,repID, parID, config, db=0):
  # rng = np.random.default_rng(seed=config.rngSeed) print("calling calc_nests")
  obsCol = int(obsCol)
  # print(f"{obsCol=}")
  surveyInt = survey[1]
  # print(f"{surveyInt=}")
  # print(f"{nestData1=}")
  longest_int = max(surveyInt)
  flooded  = sum(nestData1[:,3]==2)
  hatched  = sum(nestData1[:,3]==0)
  # discover = nestData1[:,8]>0 ## where num obs > 0
  # print(f"{nestData1[:,obsCol]=}")
  discover = nestData1[:,obsCol]>0 ## where num obs > 0
  # if db>=5: print(f"\t\t\t{discover=}")
  # discover = nestData1[:,10]>0 ## where num obs > 0
  nestData = nestData1[(discover),:] # +> remove undiscovered nests
  # if db>=5: print(f"\t\t\t{nestData[:,2]=}")
  # if db>=5: print(f"\t\t\t{nestData[:,4]=}")
  flood_dsc  = sum(nestData[:,3]==2)
  hatch_dsc  = sum(nestData[:,3]==0)
  # if db>=3: print("\t\tcalc_nests: using column 8 to determine discovered/not")
  # if db>=3: print(
  #     f"\t\tcalc_nests: using column {obsCol} to determine discovered/not")
  # if db>=5: print(f"\t\t>>calc_nests: discovered: {len(nestData)=}")
  # if db>=3: print(f"{(nestData[:,5]==nestData[:,6])=}")
  # short    = (nestData[:,4]==nestData[:,5]) ## where i==j
  short    = (nestData[:,2]<nestData[:,4]) ## where end<i
  # short    = np.zeros(len(nestData))
  # if db>=5: print(f"\t\t\t{short.astype(int)=}")
  # exclude  = ((nestData[:,7] == 7) or (nestData[:,5]==nestData[:,6]))
  unknown  = (nestData[:,7]==7)
  # if db>=5: print(f"\t\t\t{unknown.astype(int)=}")

  # print(f"{(unknown.astype(int) + short.astype(int))=}")
  # both = (unknown.astype(int) + short.astype(int))
  # exclude = unknown or short
  # can also use bitwise or (|) or np.logical_or():
  exclude = (unknown.astype(int) + short.astype(int)) > 0 # at least one is true
  # if db>=5: print(f"\t\t\t{exclude=}")
  misclass = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests
  nestData = nestData[~exclude,:] # +> remove undiscovered nests
  flood_an  = sum(nestData[:,3]==2)
  hatch_an  = sum(nestData[:,3]==0)
  misclass2 = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests
  # if db>=5: print(f"\t\t>>calc_nests: analyzed:{len(nestData)=}")
  # if db>=2: print(f"{nestData[:,11]=}")
  # misclass = misclass - unknown
  avgFInt  = (nestData[:,9].sum()/len(discover))
  sNest    = nestData1[:,11].sum()
  avgK     = nestData[:,6].sum()/len(discover)
  maxI     = np.max(nestData[:,4])
  srand = rng.uniform(0.00, 10.00) # +> random init val for MARK
  # mark_s = run_optim(minimizer="norm",
  #                    fun=mark_wrapper,
  #                    z=srand,
  #                    arg=(nestData, par.brDays, config),
  #                    met=config.optimizer
  #                    )
  appDSR  = calc_dsr(nData=nestData1,
                      nestType="all",
                      calcType="apparent",
                      conf=config,
                      incTime=par.hatchTime,
                      psurv=par.probSurv,
                      debug=config.debugDSR)
  # markPSR = mark_s ** par.hatchTime
  appPSR = appDSR ** par.hatchTime
  # lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
  #                survey=survey,config=config)
  # # llDSR = lVal[0]
  # llDSR,llPSR,llDFR = lVal


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
    # flooded,hatched,discover.sum(),exclude.sum(),unknown.sum(),
    # misclass.sum(), avgFInt, avgK, appDSR, mark_s, repID, parID])
    # parID,repID,flooded,hatched,sNest,discover.sum(),exclude.sum(),unknown.sum(),
    parID,repID,flooded,hatched,flood_dsc,hatch_dsc,flood_an,hatch_an,
    sNest,discover.sum(),exclude.sum(),unknown.sum(),
    # misclass.sum()-unknown.sum(),avgFInt,avgK,appDSR,appPSR,mayfDSR_an,appDSR_an])
    misclass.sum()-unknown.sum(),misclass2.sum(),avgFInt,avgK,maxI,longest_int,appDSR,appPSR,mayfDSR_an,appDSR_an])
    # misclass.sum(), avgFInt, avgK, appDSR,appPSR, mark_s,markPSR])
  if db>=5: print(f"\t\t{nestVals=}")
  return nestVals


#---------------------------------------------------------------------------------

def prog_mark(s, ndata, nocc, con):
  """
    Run the Program MARK algorithm
    1. Grab the data for the input for MARK 
        > First, grab only discovered nests
        > Then, only the needed columns
        (nest ID, first found, last active, last checked, assigned fate)
         inp[0] = ID | inp[1] = i | inp[2] = j | inp[3] = k | inp[4] = fate `
    2. Extract rows where j minus i does not equal zero (nest wasn't only 
       observed as active for one day)
        > Model requires all nests to have at least two observations while active
    ----
    NOTE
      
      **doesn't work** for obs_int=1

      The model used in Program MARK is based on Dinsmore (2002) -  
         allows for variance in DSR & use of covariates

      These functions are based on info in 'Program MARK: A Gentle Introduction' 

  """

  # prob, dof = probs
  # allp, alldof = mark_probs(s=s, ndata=ndata)
  # ALL IN ONE FUNCTION:
  s    = s.item() # EX makes singleton array into scalar
  # print(f"\t\t{s=}")
  # allp   = np.array(range(1,len(ndata)), dtype=np.longdouble) # all nest probabilities 
  allp   = np.array(range(0,len(ndata)), dtype=np.longdouble) # all nest probabilities 
  expo = calc_exp(inp=ndata[:,4:7], expPercent=0.4, cn=con)
  # print(f"{expo.shape[0]=} | {allp.shape[0]=}")
  for n in range(len(ndata)-1): # want n to be the row NUMBER
    # alive_days = expo[n,0] - 1
    # final_int  = expo[n,1] - 1
    alive_days = expo[n,0] 
    final_int  = expo[n,1]
  
    ##+> don't know why these equaations don't work when final_int = 1
    if final_int > 0: # final int for hatched nests == 0
      p   = (s**alive_days)*(1-(s**final_int)) 
    else:
      p   = s**alive_days
    allp[n]   = p # NOTE this line is throwing the Deprecation Warning
  nll = sum(-np.log(allp)) # +> sum of log = log of product & maybe faster
  # nll = -np.log(np.prod(allp))
  # NOTE these if statements take up lots of time, esp inside the optimizer

  #~----------------------------------------------------------------------------
  # if con.debugM>=2:
  #   s_arr = np.full(ndata.shape[0], s) ##+> length & fill value
  #   id_arr = ndata[:,0]
  #   fate_arr = ndata[:,7]
  #   mark_out = np.column_stack((allp, expo, s_arr, id_arr, fate_arr))
  #   np.save("out/MARK_print.npy", mark_out)
  return(nll)

# -----------------------------------------------------------------------------

def mark_wrapper(srn, ndata, nocc, conf):
  """
    This function calls the program MARK function when given a random starting 
    value (srn), number of occasions (nocc), and some nest data (ndata)

      > values given to optimizer are transformed then passed to MARK function
        > allows larger range of values for optimizer to work over w/o overflow
        > but values given to the function are still between 0 and 1, as required

      > Create vector to store the log-transformed values, then fill

    ---------
    RETURNS:
      output from prog_mark()

    -------
    NOTES:
      the logistic function tends to overflow if it's a normal float; make it np.float128

  """
  s = logistic(srn)
  #@#print("logistic of random starting value for program MARK:", s, s.dtype)
  ret = prog_mark(s, ndata, nocc, con=conf)
  #@#print("ret=", ret)
  return ret


# -----------------------------------------------------------------------------

def johnson(ndata, srn):
  """
    NOTE: Johnson (1979) provided a mathematical derivation that allowed the 
        calculation of variance for the estimate.
    He ALSO came to the conclusion that the Mayfield method is pretty much 
    equivalent to his ML estimator, w/ adjustment for long intervals.
    > for a single day:
       > probability of survival is s  
       > probability of failure is (1-s)
    > for interval of length k days:
       > prob of survival is s**k 
       > prob of failure is s**(1/2k-1)(1-s)
    > ex. - prob of a nest surviving three days and failing on the fourth is:
        s*s*s*(1-s) 
      > this assumes that a failed nest survived half (minus a day)
        of interval and then failed
    Johnson's rewriting of the Mayfield estimator:
         mortality = (f1 + sum(ft)) / (h1 + sum(t*ht) + f1 + 0.5*sum(t*ft)) 
    > created by differentiating the log-likelihood equation and setting to 
      zero (maximizing)
    > ht = hatched or survived til next visit; ft = failed by next visit
    > f1 and h1 represent an interval between visits of one day, which is not 
      used in our studies
      > so we end up with: sum(ft) / (sum(t*ht) + 0.5*sum(t*ft)) 
         where t = interval length, and 
         f and h represent number of failures and hatches, respectively
    Johnson's Mayfield-40 estimator: 
         mortality = sum(ft) / (sum(t*ht) + 0.4*sum(t*ft))
    Johnson's modified ML estimator:
         1/s*(sum(t*ht)) = sum( (t * ft * s^t-1) / (1 - s^t))
    ---------------------------------------------------------------------------
  """
  print("calculate Johnson estimator")
  # jEst = (1/srn) * sum()

# -----------------------------------------------------------------------------
