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
import yaml

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
from helpers import mk_param_list, mk_outdir, mk_fnames, arrPrint, printLL, print_all
from makeNests import stormGen
from observer import make_obs, mk_surveys
from dsrCalc import calc_dsr, mark_wrapper
from MCmatrix import like_smd, triangle, logistic

np.set_printoptions(precision=3)
debug = config.debug
print("\t\t|>|>|>debug value:", debug)


def randArgs():
  """
    Choose random initial values for the optimizer.
    These will be log-transformed before going through the likelihood function
    
    RETURNS:
      array of s, mp, ss, mps (for like_smd) and srand (for mark_wrapper)
  """
  s   = rng.uniform(-10.0, 10.0)     
  mp  = rng.uniform(-10.0, 10.0)
  # ss  = rng.uniform(-10.0, 10.0)
  # mps   = rng.uniform(-10.0, 10.0)
  srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
  # z = np.array([s, mp, ss, mps, srand])
  z = np.array([s, mp])
  return(z)

# -----------------------------------------------------------------------------
#   CREATE NEST DATA AND RUN THE OPTIMIZER 
# -----------------------------------------------------------------------------
## +>loop thru param combinations; within loop, unpack params & run optimizer
# @profile
def run_optim(minimizer, fun, z, arg, met='Nelder-Mead'):
  """
    Run scipy.optimize.minimize on 'fun'. Will return value of -1 or -2 if exceptions occur.

    If all is well, transform the output (using ansTransform() for MCMC model, and
    logistic() for MARK model)

    Returns:
      the transformed output.
  """
  
  try:
    out = choose_alg(minimizer, fun, z, arg, met)
    ex = 0.0
  except decimal.InvalidOperation as error2:
    ex=-1.0
    print("\t\t>> Error: invalid operation in decimal:", error2, "Go to next replicate.")
    return(ex)
  except OverflowError as error3:
    ex=-2.0
    print(
      "\t\t>> Error: overflow error:", 
      error3, 
      "Go to next replicate."
      )
    return(ex)
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

def choose_alg(minim, fun, z, arg, met):
  if minim=="norm":
    minimizer = optimize.minimize(fun, z, args=arg, method=met) 
  elif minim=="bh":
    min_kwargs={"args": arg}
    minimizer = optimize.basinhopping(fun, z, minimizer_kwargs=min_kwargs)
    
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
def rep_loop(par, nData, storm, survey, config):
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
  dat = nData[:,4:10] # doesn't include column index 10
  arg=(dat, par.obsFreq, par.useSMat, storm, survey, par.whichLike, config)
  res    = run_optim(minimizer="norm", fun=like_smd, z=randArgs(), arg=arg)
  if res[0] < 0.6:
    #~#if debug>=3: print(f"\t\t\tres={res[0]}; run optimizer again with basinhopping")
    res = run_optim(minimizer="bh", fun=like_smd, z=randArgs(), arg=arg)
  srand = rng.uniform(-10.00, 10.00)
  mark_s = run_optim(minimizer="norm", fun=mark_wrapper, z=srand, arg=(nData, par.brDays, config))
  #NOTE ans2 is an "OptimizeResult" object; need to extract "x"
  # NOTE scott was probably right - mps doesn't make sense. and DSR includes storms already
  # so check whether mort flood probability goes up with more intense storms?
  s2, mp2 = res[0], res[1]
  like_val = np.array([ mark_s,s2,mp2], dtype=np.longdouble)
  #~#if config.debugLL>=2: print(f"\t\t>> like_val: MARK={like_val[0]}, MCMC-surv={like_val[1]}, MCMC-pred={like_val[2]}")
  return(like_val)
  
def main(fnUnique, testing, parLists, config=config, pStatic=staticPar):
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
  # lf_suffix=""
  pList = parLists
  now_str = now_short
  odir  = mk_outdir(now_str)
  # dirs  = mk_outdir(now_str, suf=f"_{atype}")
  # dirs  = mk_outdir(now_str)
  # odir  = dirs[0]
  # ndir  = dirs[1]
  # if config.testing:
  #   odir  = mk_outdir(now_str, suf=f"{atype}")
  # else:
  #   odir  = mk_outdir(now_str, suf=f"{atype}")
  print(f"\t|> output directory = {odir}")
  print(f"\tCONFIG: {config}")
  if fnUnique:
    fname = mk_fnames(now_str, fdir=odir, suf=f"{atype}", uniq=True) 
  else:
    fname = mk_fnames(now_str,fdir=odir,suf=f"{atype}")
  # fdir  = fname[0].parent
  likeFile = fname[0]
  if config.testing=="no":
    if os.path.exists(likeFile):
      print("filepath exists! Exiting scipt.")
      return
  colNames = fname[1]
  with open(likeFile, "wb") as f: # NOTE not 'a' bc file stays open
    paramsArray = mk_param_list(parList=pList, fdir=odir, suf=f"{config.rngSeed}{atype}")
    print(
        f"\n\t|>|>|>{len(paramsArray)} param sets x {config.nreps} reps ="
        f" {len(paramsArray)*config.nreps} total rows"
        )
    parID     = 0
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
      # likeVal  = np.zeros(shape=(config.nreps,config.numOut))
      # ndMatrix  = np.zeros(shape=(config.nreps,par.numNests,11)) #+> 2nd dim = ncol(nData)+1
      
      ndMatrix  = np.zeros(shape=(config.nreps,par.numNests,10))
      for r in range(config.nreps): 
        print(f"\t>---->----> replicate  {repID} >---->----> ")
        try:
          nestData1 = make_obs(par=par, storm=stormDays, survey=survey, conf=config) 
        except IndexError as error:
          print(
            "\t\t>> !!! IndexError in nest data:", 
            error,
            ". Go to next replicate")
          nEx = nEx + 1
          continue

        # +> saves each rep as separate file:
        # ndName = f"{odir}/nd_p{parID}_r{repID}.npy"
        # np.save(ndName, nestData1)
        # rep_col = np.full(len(nestData1), repID)
        # rep_col = rep_col[:,np.newaxis]
        # nd2 = np.hstack([nestData1, rep_col])
        # ndMatrix[r,:,:] = nd2
        ndMatrix[r,:,:] = nestData1 # +> now reps are a dimension, not a column

        trueDSR  = calc_dsr(nData=nestData1, nestType="all", calcType="true", conf=config)
        flooded  = sum(nestData1[:,3]==2)
        hatched  = sum(nestData1[:,3]==0)
        # print_prop(nestData[:,7], nestData[:,3], )
        discover = nestData1[:,6]!=0
        nestData = nestData1[(discover),:] # +> remove undiscovered nests
        exclude  = ((nestData[:,7] == 7) | (nestData[:,4]==nestData[:,5]))             
        unknown  = (nestData[:,7]==7)
        misclass = (nestData[:,7]!=nestData[:,3])
        nestData  = nestData[~(exclude),:]  # +> remove excluded nests 
        trueDSR_an   = calc_dsr(nData=nestData, nestType="analysis",
                                calcType="true", conf=config) 
        lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
                   survey=survey,config=config)

        if config.debugLL>=2:
          #~#llArg = np.load('out/arg_PrintLL.npy') 
          #~#printLL(len(llArg), *llArg.T) # +> tranpose so it is unpacked colwise
          # ha, fl,dsc, unk, mc, ex, dsr_c, dsr_a, dsr_t = sums #+> unpack vals
          sum_list = [
              hatched,
              flooded,
              discover,
              unknown,
              misclass,
              exclude,
              lVal[1],
              trueDSR_an,
              trueDSR,
              ]
          print_all(sum_list, nestData, par)

        nVal = np.array([trueDSR, trueDSR_an, sum(discover), sum(exclude), sum(unknown), sum(misclass), flooded, hatched, nEx, repID, parID])  
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
        repID = repID + 1
        
      # if debug>=2: arrPrint(ndMatrix)
      if config.saveNData:
        print("\t>--> saving nest data to file")
        ndName = Path (f"{odir}/nests{config.rngSeed}_{atype}/nd_par{parID:03}.npy")
        ndName.parent.mkdir(parents=True, exist_ok=True)
        np.save(ndName, ndMatrix)
      parID = parID + 1

main(fnUnique=config.fnUnique, parLists=pLists, testing=config.testing)
