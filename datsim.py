#!/usr/local/bin/python


# sudo vim -o file1 file2 [open 2 files] 
# BLAH
# :/^[^#]/ search for uncommented lines
# /^[^#]*\s*print  or /^\s*print 
# > kernprof -l simdata_vect.py > 20sepprofile.out
# > python -m line_profiler .\simdata_vect.py.lprof
# NOTE 5/16/25 - The percent bias responds more like I would expect when I use
#        the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
#        BUT I still don't know why the calculated DSR is consistently low.

import numpy as np 
import scipy.stats as stats
import csv
import decimal
import itertools
import os
import pprint
import sys
import time
import yaml
import warnings
# warnings.simplefilter("always")

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
# from itertools import product
# import line_profiler
# import numexpr as ne
# from os.path import exists
from pathlib import Path
from scipy import optimize
from typing import Dict, Generator

from getClass import Params, Config
from settings import rng,config,atype,staticPar, pLists, now_short, now_long
from helpers import mk_param_list, mk_outdir, mk_fnames 
from print_func import arrPrint,dfPrint,printLL, print_all, print_nestdata,print_mark
from makeNests import stormGen
from observer import make_obs, mk_surveys
from dsrCalc import calc_dsr, mark_wrapper, mayfield
from MCmatrix import like_smd, triangle, logistic

np.set_printoptions(precision=3)
debug = config.debug
print("\t\t|>|>|>debug value:", debug, end=" ")


def randArgs():
  """
    Choose random initial values for the optimizer.
    These will be log-transformed before going through the likelihood function
    
    RETURNS:
      array of s, mp, ss, mps (for like_smd) and srand (for mark_wrapper)
    ----
    note that the estimates from these initial vals seem to be consistently
    biased positive. So maybe not an issue with the starting vals... but
    starting with the Mayfield estimate could make the process faster
  """
  # s   = rng.uniform(-10.0, 10.0)     
  # mp  = rng.uniform(-10.0, 10.0)
  s   = rng.uniform(0.0, 10.0)     
  mp  = rng.uniform(0.0, 10.0)
  # ss  = rng.uniform(-10.0, 10.0)
  # mps   = rng.uniform(-10.0, 10.0)
  # srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
  srand = rng.uniform(0.0, 10.0) # should the MARK and matrix MLE start @ same value?
  # z = np.array([s, mp, ss, mps, srand])
  z = np.array([s, mp])
  return(z)

# def mayfInit(mayfEstim, ):


# NOTE I have no real reason for choosing Nelder-Mead. Try another
# -----------------------------------------------------------------------------
#   CREATE NEST DATA AND RUN THE OPTIMIZER 
# -----------------------------------------------------------------------------
## +>loop thru param combinations; within loop, unpack params & run optimizer
# @profile
##+> add debug to print starting vals for basinhopping
def run_optim(minimizer, fun, z, arg, met='Nelder-Mead', db=False):
  """
    Run scipy.optimize.minimize on 'fun'. Will return value of -1 or -2 if exceptions occur.

    If all is well, transform the output (using ansTransform() for MCMC model, and
    logistic() for MARK model)

    Returns:
      the transformed output.
  """
  counter=0
  try:
    with warnings.catch_warnings(): #+> treeat warnings as exceptions here
      warnings.simplefilter("error")

      out = choose_alg(minimizer, fun, z, arg, met,debug=db)
      last_ex = 0.0
      # print("\t\t\tCOUNTER:",counter)
  except RuntimeWarning as error:
    last_ex=-10000
    counter+=1
    # print("\t\t>> Runtime warning:", error, "[count={counter}]")
    print("\t\t>> Runtime warning:"
          f"{error}; [err count={counter}]. Re-run.")
    out = choose_alg(minimizer, fun, z, arg, met,debug=db)
    if counter >3 :
      print("\t\t>> more than 3 errors; go to next replicate")
      return(last_ex)

  except decimal.InvalidOperation as error2:
    last_ex=-20000
    counter+=1
    print("\t\t>> Error: invalid operation in decimal:"
          f"{error2}; [err count={counter}]. Re-run.")
    out = choose_alg(minimizer, fun, z, arg, met,debug=db)
    if counter >3 :
      print("\t\t>> more than 3 errors; go to next replicate")
      return(last_ex)

  except OverflowError as error3:
    last_ex=-30000
    counter+=1
    print( "\t\t>> Error: overflow error:"
          f"{error3}; [err count={counter}]. Re-run.")
    out = choose_alg(minimizer, fun, z, arg, met,debug=db)
    if counter >3 :
      print("\t\t>> more than 3 errors; go to next replicate")
      return(last_ex)
  # print("Success?", out.success, out.message, "answer=", out.x)
  if fun==like_smd: 
    # print("Success?", out.success, out.message, "answer=", out.x)
    res = ansTransform(ans=out.x)
    # if res[1] < 0.6:
    #   print("run optimizer again with basinhopping")
    #   arg=
    #   try:
    #     out = optimize.minimize(fun, z, args=)
  else:
    # res=ansTransform(ans, unpack=False)
    # res=ansTransform(ans=out)
    res = logistic(out.x[0])
    # print("\t", res)
  return(res)

def choose_alg(minim, fun, z, arg, met, debug=False):
  if minim=="norm":
    minimizer = optimize.minimize(fun, z, args=arg, method=met) 
  elif minim=="bh":
    min_kwargs={"args": arg}
    minimizer = optimize.basinhopping(fun, z, minimizer_kwargs=min_kwargs)
    # if debug: print(f"{z=}", end=" ")
    
  return(minimizer)
  
def ansTransform(ans):
  """
    Transform the optimizer output so that it is between 0 and 1, and the 3 
    probabilities sum to 1. 

    'ans' is an object of type 'OptimizeResult', which has a number of components
  """
  # if unpack:
    # ans = ans.x  
  # s0   = ans.x[0]     # Series of transformations of optimizer output.
  s0   = ans[0]     # Series of transformations of optimizer output.
  mp0  = ans[1]     # These make sure the output is between 0 and 1, 
  # ss0  = ans[2]     # and that the three fate probabilities sum to 1.
  # mps0 = ans[3]

  s1   = logistic(s0)
  mp1  = logistic(mp0)
  # ss1  = logistic(ss0)
  # mps1 = logistic(mps0)

  ret2 = triangle(s1, mp1)
  s2   = ret2[0]
  mp2  = ret2[1]
  mf2  = 1.0 - s2 - mp2

  # ret3 = triangle(ss1, mps1)
  # ss2  = ret3[0]
  # mps2 = ret3[1]
  # mfs2 = 1.0 - ss2 - mps2
  
  # ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.longdouble)
  ansTransformed = np.array([s2, mp2, mf2], dtype=np.longdouble)
  return(ansTransformed)

# -----------------------------------------------------------------------------

# @profile
def rep_loop(par, nData, storm, survey, config, random=True):
  """
    For each data replicate, call this function, which:
      - takes the reduced nest data as input
      - calls the optimizer on like_smd() and mark_wrapper()
    
    Returns: like_val (daily survival and mortality values)
      [0]: program MARK (DSR).....[1]: MCMC (DSR).....[2]: MCMC (DMR)......
  """
  # +>---- empty array to store data for this replicate: ---------
  # like_val  = np.zeros(shape=(config.numOut), dtype=np.longdouble)
  # like_val  = np.zeros(shape=(3), dtype=np.longdouble)
  # perfectInfo = 0
  # whichL = par.whichLike
  ## +> np.r_ allows r-like indexing
  dat = nData[:, np.r_[0,4:10]] # doesn't include column index 10
  # print("check:")
  # arrPrint(dat)
  arg=(dat, par.obsFreq, par.useSMat, storm, survey, par.whichLike, config)
  # zargs = randArgs() if random==True else mayfInit()
  # TODO: integrate warning capture with function in helpers.py
  if config.optimizer=="global":
    res    = run_optim(minimizer="bh",
                       fun=like_smd,
                       z=randArgs(),
                       arg=arg,
                       )
  else:
    res    = run_optim(minimizer="norm",
                       fun=like_smd,
                       z=randArgs(),
                       arg=arg,
                       met=config.optimizer,
                       )
  if res[0] < 0.8:
    print(f"\t\t\t\t{res[0]=:.4f}; run optimizer again with basinhopping; ", end=" ")
    res = run_optim(minimizer="bh",fun=like_smd,z=randArgs(),arg=arg,db=True)
    # print(f"\t\t|>NEW {res[0]=:.4f}")
    if res[0] < 0.8:
      # discover = nData[nData[:,6]!=0]
      discover = nData[nData[:,8]>0] ## +> n obs > 0
      discovered = discover.shape[0]
      # excl = ((discover[:,7] == 7) | (discover[:,4]==discover[:,5]))
      # excl = ((discover[:,7] == 7) | discover[:,8]>0)
      excl = (discover[:,7] == 7)
      excluded  = np.sum(excl)            
      hatched = np.sum(discover[:,3]==0)
      unknown = np.sum(discover[:,7]==7)
      # print(f"\t\t\t\t\t{discovered=}|>{excluded=}&{hatched=}&{unknown=}", end=" ")
      # print(f"{res[0]=:.4f}; AGAIN with bh", end=" ")
      res0 = run_optim(minimizer="bh",fun=like_smd,z=randArgs(),arg=arg,db=True)
      res1 = run_optim(minimizer="bh",fun=like_smd,z=randArgs(),arg=arg,db=True)
      res2 = run_optim(minimizer="bh",fun=like_smd,z=randArgs(),arg=arg,db=True)
      # print(f"\t\t\t\t\t|>NEW {res0[0]=:.4f}{res1[0]=:.4f}{res2[0]=:.4f}", end=" ")
      # resList = np.array(res1, res2, res3)
      # resArr= np.concatenate((res1, res2, res3), axis=0)
      first_vals = [arr[0] for arr in [res0, res1, res2]]
      
      resName = "res" + str(first_vals.index(max(first_vals)))
      # res = eval(resName)
      res = locals()[resName]
      # res = resArr[np.argmax(resArr[:,0])]

      # res = np.max([res1,res2,res3])
      # print(f"\t\t\t\t\t|>NEW {res[0]=:.4f}")
  # srand = rng.uniform(-10.00, 10.00)
  if config.mayfStart:
        # mayfDSR_all  = calc_dsr(nData=nestData1,
        #                     nestType="all",
        #                     calcType="mayfield",
        #                     conf=config,
        #                     incTime=par.hatchTime,
        #                     psurv=par.probSurv,
        #                     debug=config.debugSummary)
    srand = calc_dsr(nData, nestType="disc", calcType="mayfield", conf=config,
                     incTime=par.hatchTime, psurv=par.probSurv)
  else:
    srand = rng.uniform(0.00, 10.00)
  # mark_s = run_optim(minimizer="norm", fun=mark_wrapper, z=srand, arg=(nData, par.brDays, config))
  if config.optimizer=="global":
    res    = run_optim(minimizer="bh",
                       fun=mark_wrapper,
                       z=srand,
                       arg=arg,
                       )
  else:
    mark_s = run_optim(minimizer="norm",
                       fun=mark_wrapper,
                       z=srand,
                       arg=(nData, par.brDays, config),
                       met=config.optimizer
                       )
  #NOTE ans2 is an "OptimizeResult" object; need to extract "x"
  # NOTE scott was probably right - mps doesn't make sense. and DSR includes storms already
  # so check whether mort flood probability goes up with more intense storms?
  s2, mp2 = res[0], res[1]
  like_val = np.array([ mark_s,s2,mp2], dtype=np.longdouble)
  #~#if config.debugLL>=2: print(f"\t\t>> like_val: MARK={like_val[0]}, MCMC-surv={like_val[1]}, MCMC-pred={like_val[2]}")
  return(like_val)
  
def main(fnUnique, testing, parLists, msg="", config=config, pStatic=staticPar):
  """
    If 'fnUnique'==True, filename is "uniquified" and includes H:M:S
      --> Otherwise, just the date.

    Output: a csv combining output from likelihood optimization (lVal- output 
    from rep_loop) & certain nest-related and optimizer-related values (nVal)
      Columns:
      [0] MARK estimate.....[1] MCMC estimate....[2] MCMC mortality estimate
      [3] Mayfield estimate - all nests [4] Mayfield estimate - analysis nests
      [5] num discovered....[6] num excluded.....[7] num unknown fate
      [8] num misclassified [9] num flooded.....[10] num hatched
      [11] num exceptions caught [12] replicate ID [13] parameter set ID
  """
  ### TODO: count the number of times the model returns "0.5000" or thereabouts 
  # counters <- range
  pList = parLists
  now_str = now_short
  odir  = mk_outdir(now_str)
  # print(f"\t|> output directory = {odir}")
  if len(msg) > 0:
    print("|> MSG: ", msg)
  print(f"\t\t<>CONFIG: {config}")
  if fnUnique:
    fname = mk_fnames(now_str, fdir=odir, suf=f"{atype}", uniq=True) 
  else:
    fname = mk_fnames(now_str,fdir=odir,suf=f"{atype}")
  likeFile = fname[0]
  if config.testing=="no":
    if os.path.exists(likeFile):
      print(f"filepath {likeFile} exists! renaming ")
      suf = datetime.today().strftime('%m%d%Y_%H%M%S')
      likeFile=Path(str(likeFile)+suf)
      print(f"{likeFile=}")
      return
  colNames = fname[1]
  with open(likeFile, "wb") as f: # NOTE not 'a' bc file stays open
    paramsArray = mk_param_list(parList=pList, fdir=odir, suf=f"{config.rngSeed}{atype}")
    print(
        f"\n\t|>|>|>{len(paramsArray)} param sets x {config.nreps} reps ="
        f" {len(paramsArray)*config.nreps} total rows"
        )
    parID     = 0
    summVars = ['dsc','ha','fl','unk','exc','mc','dsrT','dsrA','dsrD',
                'dsrC','diffA','diffD','diffC']
    summMat   = np.zeros(shape=(len(paramsArray), config.nreps, len(summVars)))
    # summMat   = np.zeros(shape=(len(paramsArray), config.nreps, 14))
    # print(f"dimensions of summary matrix: {summMat.shape}")
    for i in range(0, len(paramsArray)): # +> for each set of params
      par    = paramsArray[i] 
      par_merge  = {**par, **pStatic}
      par    = Params(**par_merge)
      print("\n\t<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
      print(f"\n>>>> param set #{parID} & {par=}\n")
      stormDays  = stormGen(frq=par.stormFrq, dur=par.stormDur)
      survey   = mk_surveys(stormDays, par.obsFreq, par.brDays, conf=config)
      # surveyDays, surveyInts = survey
      repID = numMC = nEx = 0 # +> num nests misclassified, num exceptions
      #~------------------------------------------------------------------------
      # likeVal  = np.zeros(shape=(config.nreps,config.numOut))
      # ndMatrix  = np.zeros(shape=(config.nreps,par.numNests,11)) #+> 2nd dim = ncol(nData)+1
      
      #~------------------------------------------------------------------------
      # if config.nreps==1:
      #   ndMatrix = np.zeros(shape=(par.numNests,10))
      # else:
      #   ##+> 3d array (harder to load into R):
      #   ndMatrix  = np.zeros(shape=(config.nreps,par.numNests,10))
      #------------------------------------------------------------------------
      for r in range(config.nreps): 
        print(f"\t>---->----> replicate  {parID}.{repID} >---->----> ")
        try:
          nestData1 = make_obs(par=par, storm=stormDays, survey=survey, conf=config) 
        except IndexError as error:
          print(
            "\t\t>> !!! IndexError in nest data:", 
            error,
            ". Go to next replicate")
          nEx = nEx + 1
          continue

        #~----------------------------------------------------------------------
        # +> save each rep as separate file-useful for loading into R
        # if config.saveNData:
        #   print("\n\t\t!!!!! SAVING THIS REPLICATE'S NEST DATA TO FILE")
        #   ndName = Path(f"{odir}/nd_{config.rngSeed}{atype}/nd_p{parID:02}_r{repID:02}.npy")
        #   #   ndName = Path (f"{odir}/nests{config.rngSeed}_{atype}/nd_par{parID:03}.npy")
        #   ndName.parent.mkdir(parents=True, exist_ok=True)
        #   # np.save(ndName, nestData1)
        #   # rep_col = np.full(len(nestData1), repID)
        #   # rep_col = rep_col[:,np.newaxis]
        #   # nd2 = np.hstack([nestData1, rep_col])
        #   # np.save(ndName, nd2)
        #   np.save(ndName, nestData1)
        #   # ndMatrix[r,:,:] = nd2
        #
        ## +> print all nest data
        if config.debug>=2:
          #   print("\t\t\t|> ALL NEST DATA"None)
          nm = ["ID","init","end","fate"," i "," j "," k ","afate","nobs","fint","nstm"]
          # print_nestdata(nestData1, names=nm,nprint=50) 
          print_nestdata(nestData1, names=nm, abbrv=False) 

        # if config.debug>=1: 
          # print("\n\t\t[*] [*] [*] [*] [*] calculating DSR [*] [*] [*] [*] [*] [*] [*] [*] ")

        ##+> matrix to store into about the nest data:
        # if config.nreps==1:
        #   ndMatrix[:,:] = nestData1 # +> now reps are a dimension, not a column
        # else:
        #   ndMatrix[r,:,:] = nestData1 # +> now reps are a dimension, not a column
        #----------------------------------------------------------------------

        ##+> ends up weird bc undiscovered nests have negative exposure:
        mayfDSR_all  = calc_dsr(nData=nestData1,
                            nestType="all",
                            calcType="mayfield",
                            conf=config,
                            incTime=par.hatchTime,
                            psurv=par.probSurv,
                            debug=config.debugSummary)

        ##+> decreases slightly in accuracy w/ lower vals of DSR:
        appDSR  = calc_dsr(nData=nestData1,
                            nestType="all",
                            calcType="apparent",
                            conf=config,
                            incTime=par.hatchTime,
                            psurv=par.probSurv,
                            debug=config.debugSummary)
        flooded  = sum(nestData1[:,3]==2)
        hatched  = sum(nestData1[:,3]==0)
        # print_prop(nestData[:,7], nestData[:,3], )
        # discover = nestData1[:,6]!=0
        discover = nestData1[:,8]>0
        nestData = nestData1[(discover),:] # +> remove undiscovered nests
        # exclude  = ((nestData[:,7] == 7) | (nestData[:,4]==nestData[:,5]))             
        exclude  = ((nestData[:,7] == 7))
        # print(f"\t\t\tXCLUDING: {sum(exclude)=}")
        # arrPrint(" {exclude}")
        unknown  = (nestData[:,7]==7)
        # TODO: make sure misclass is calculated correctly! 
        # TODO: add a summary of todos when a file is opened?
        misclass = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests

        mayfDSR_disc = calc_dsr(nData=nestData,
                                nestType="disc",
                                calcType="mayfield",
                                conf=config,
                                incTime=par.hatchTime,
                                psurv=par.probSurv,
                                debug=config.debugSummary)

        ##+> for discovered nests, these two estimates are very close
        appDSR_disc = calc_dsr(nData=nestData,
                                nestType="disc",
                                calcType="apparent",
                                conf=config,
                                incTime=par.hatchTime,
                                psurv=par.probSurv,
                                debug=config.debugSummary)

        nestData  = nestData[~(exclude),:]  # +> remove excluded nests 
        # print("\n\t\tnest data length after excluding:",nestData.shape[0])
        mayfDSR_an   =  calc_dsr(nData=nestData,
                                 nestType="analysis",
                                 calcType="mayfield",
                                 conf=config,
                                 incTime=par.hatchTime,
                                 psurv=par.probSurv,
                                 debug=config.debugSummary) 
        appDSR_an   = calc_dsr(nData=nestData,
                                nestType="analysis",
                                calcType="apparent",
                                conf=config,
                                incTime=par.hatchTime,
                                psurv=par.probSurv,
                                debug=config.debugSummary) 
        lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
                   survey=survey,config=config)
        llDSR = lVal[1]

        #~----------------------------------------------------------------------
        # if config.debugM>=2: ## +> matches the level for saving the info to print
        #   ##+> enable the saving in the prog_mark() definition
        #   prExp = True if config.debugM>=3 else False
        #   print_mark(print_exp=prExp)
        #
        # if config.debugLL>=2: ## +> matches the level for saving the info to print
        #   ##+> print LL equations for each nest:
        #   ##+> enable the saving inside hte logLike function definition
        #
        #   llArg = np.load('out/arg_PrintLL.npy') 
        #   printLL(len(llArg), *llArg.T) # +> tranpose so it is unpacked colwise
        #
        # if config.testing == "yes":
        #   print(f"\n\t\t\t\t{mayfDSR_all=:.3f}, {mayfDSR_disc=:.3f}, {mayfDSR_an=:.3f}")
        #   print(f"\t\t\t\t{appDSR=:.3f}, {appDSR_disc=:.3f}, {appDSR_an=:.3f}")
        #
        # if config.debug>=2: ## +> matches the level for saving the info to print
        #   ##+> print summary:
        #   sum_list = [
        #       hatched,
        #       flooded,
        #       discover,
        #       unknown,
        #       misclass,
        #       exclude,
        #       lVal[1],
        #       lVal[0],
        #       appDSR_an,
        #       appDSR,
        #       appDSR_disc,
        #       ]
        #   print_all(sum_list, nestData, par)
        #----------------------------------------------------------------------

        nVal = np.array([appDSR,        #4
                         appDSR_disc,
                         mayfDSR_disc,
                         mayfDSR_an,    #5
                         sum(discover),
                         sum(exclude),
                         sum(unknown),
                         sum(misclass),
                         flooded,
                         hatched,
                         nEx,
                         repID,
                         parID,
                         ])  
        like_val = np.concatenate((lVal, nVal))
        colnames=colNames # colnames=config.colNames

        if parID == 0 and like_val[12] == 0: #+> only 1st line gets the header
          np.savetxt(f, [like_val], delimiter=",", header=colnames)
          # if debug: print(">> ** saving likelihood values with header **")
        else:
          np.savetxt(f, [like_val], delimiter=",")
          # if debug: print(">> ** saving likelihood values **")
        # need to save it in the function where f was opened?
        # likeVal[r] = like_val

      #~----------------------------------------------------------------------
        ### +> save vals to summary matrix for printing:
        # if config.testing=="yes":
        #   # print("saving to matrix")
        #   # summVars = ['dsc','ha','fl','unk','exc','mc','dsrT','dsrA','dsrD','dsrC']
        #   # summVal =np.array([ discover, hatched, flooded, unknown, exclude, misclass, trueDSR, trueDSR_an, trueDSR_disc, lVal[1] ])
        #   summVal =np.array([
        #     discover.sum(),
        #     hatched.sum(),
        #     flooded.sum(),
        #     unknown.sum(),
        #     exclude.sum(),
        #     misclass.sum(),
        #     appDSR,
        #     appDSR_an,
        #     appDSR_disc,
        #     llDSR,
        #     llDSR-appDSR,
        #     llDSR-appDSR_an,
        #     llDSR-appDSR_disc,
        #     ])
        #   # print(f" {summMat[i,r,:].shape=} | {summVal.shape=}")
          #
          # summMat[i,r,:] = np.array(summVal)
      #----------------------------------------------------------------------

        ##NOTE don't comment! need to increment the repID
        repID = repID + 1
        
      #~----------------------------------------------------------------------
      # if debug>=2: arrPrint(ndMatrix)

      ##+> save nest data - takes up lots of disk space
      # if config.saveNData:
      #   print("\t>--> saving nest data to file")
      #   ndName = Path (f"{odir}/nests{config.rngSeed}_{atype}/nd_par{parID:03}.npy")
      #   ndName.parent.mkdir(parents=True, exist_ok=True)
      #   np.save(ndName, ndMatrix)
      #----------------------------------------------------------------------

      ##NOTE don't comment! need to increment the parID
      parID = parID + 1

    #~----------------------------------------------------------------------
    ##+> print the means from the matrix:
    # if config.testing=="yes":
    #   ## +> get means of summary matrix:
    #   meanMat = summMat.mean(axis=1)
    #   svars = ', '.join([str(x) for x in summVars]) # needs to be string
    #   pListN = mk_param_list(pList, listRet=True)
    #   # NOTE probably could make df out of a list of dicts anyway
    #   print(pListN)
    #   # pListNew = [pl[0,1,5,9] for pl in pListNew] #+> doesn't work!
    #   # pListNew = [pl[i] for pl in pListNew for i in [0,1,5,9]]
    #   # for pl in pListNew:
    #   pListNew = [[pl[0], pl[1], pl[5], pl[9]] for pl in pListN]
    #
    #   print(pListNew)
    #   # pnames = list(dir(Params)) ##+> this one gives all components of class
    #   # pnames = list(vars(par).keys()) ##+> this one includes staticPar
    #   # pnames = list(paramsArray[0].keys()) ##+> get the names from the dict?
    #   # pnames = [pnames[i] for i in [0,1,5,9]]
    #   # print(pnames)
    #   pnames = ["nNest", "pSurv", "obsInt", "hTime"]
    #
    #   fname_mean = f"out/mean_mat_{config.rngSeed}{atype}.csv"
    #   np.savetxt(fname_mean, meanMat, fmt='%.5f', delimiter=",", header=svars)
    #   # if config.debug>=2: arrPrint(summMat) ## +> print entire matrix
    #   print("==>> mean values for each param set: ")
    #   # arrPrint(meanMat)
    #   # dfPrint([meanMat,pListNew], names=[summVars,pnames])
    #   dfPrint([meanMat,pListNew], concat="cwise", names=[summVars,pnames])
    #----------------------------------------------------------------------

# startTime = time.perf_counter()

main(fnUnique=config.fnUnique, parLists=pLists, msg=config.msg, testing=config.testing)

# endTime = time.perf_counter()

# elapsed_time = endTime - startTime 

# if elapsed_time < 60:
    # print(f"Runtime: {elapsed_time:.2f} seconds") #
# elif elapsed_time < 3600:
    # minutes = elapsed_time / 60
    # print(f"Runtime: {minutes:.2f} minutes") #

