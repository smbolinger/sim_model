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
import numdifftools as ndt
from statsmodels.tools.numdiff import approx_hess2
import lmfit
import csv
import decimal
import itertools
import functools
import os
import pprint
import sys
import time
import traceback
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
# import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import rpy2.robjects as robjects
from scipy import optimize
from scipy.stats.kde import gaussian_kde
from scipy.differentiate import hessian
# from traceback import TracebackException
from typing import Dict, Generator

from getClass import Params, Config
# from settings import rng,config,atype,staticPar, pLists, now_short, now_long
# from rsettings import rng,config,atype,staticPar, pLists, now_short, now_long
from helpers import mk_param_list, mk_param_list_list, mk_outdir, mk_fnames,print
from print_func import arrPrint,dfPrint,printLL, print_all, print_nestdata,print_mark
# from makeNests import stormGen,initDat
from makeNests import stormGen
from observer import make_obs, mk_surveys
from dsrCalc import calc_dsr, mark_wrapper, mayfield
from MCmatrix import like_smd, triangle, logistic, logit,log_transform,tri_transform
# from notr_MCmatrix import like_smd, triangle, logistic
# from log_exposure import calc_daily_expo, make_daily_logex_df

# imported = sorted(list(sys.modules.keys()))
imported = sys.modules
calc_hess=True
randInit = True
# log_transform = False
r = robjects.r
# np.set_printoptions(suppress=True,precision=5)
# debug = config.debug
# print("\t\t|>|>|>debug value:", debug, end=" ")
#+> seearch for #\~ to find the dbug statements
## TODO:  the MCMC matrix is giving a low answer bc it doesn't

## TODO:   account for nsts that survivd storms.

## TODO: mark "storm nsts" and add an xtra day to one intrval?


def randArgs(config,rng):
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
  # rng =
  # rng = np.random.default_rng(seed=config.rngSeed)
  # s   = rng.uniform(-10.0, 10.0)     
  # mp  = rng.uniform(-10.0, 10.0)
  if randInit:
    if log_transform:
      s   = rng.uniform(2.0, 10.0)     
      mp  = rng.uniform(-5.0, 2)
      # s   = rng.uniform(0.0, 10.0)     
      # mp  = rng.uniform(0.0, 10.0)
    else:
      # s   = rng.uniform(0.6, 0.98)     
      mp  = rng.uniform(0.001, 0.1)
      mf = rng.uniform(0.001,0.1)

    # if tri_transform:
    #   z = np.array([s, mp], dtype=np.float32)
    # else:
    #   # mf = 1 - mp - s
    #   mf = logit(1-logistic(mp) - logistic(s)) if log_transform else 1 - mp - s
    #   z = np.array([s, mp, mf], dtype=np.float32)
  else:
    if log_transform:
      s = mp = mf = logit(0.33)
      z = np.array([s, mp, mf], dtype=np.float32)
    else: 
      s = mp = mf = 0.33
      z = np.array([s, mp, mf], dtype=np.float32)


  # ss  = rng.uniform(-10.0, 10.0)
  # mps   = rng.uniform(-10.0, 10.0)
  # srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
  # srand = rng.uniform(0.0, 10.0) # should the MARK and matrix MLE start @ same value?
  # z = np.array([s, mp, ss, mps, srand])
  z = np.array([mp, mf], dtype=np.float32)
  # print(f"randArgs: {z=} {type(z[0])=}")
  return(z)

def randArgs2(config,rng):
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
  # rng =
  # rng = np.random.default_rng(seed=config.rngSeed)
  # s   = rng.uniform(-10.0, 10.0)     
  # mp  = rng.uniform(-10.0, 10.0)
  s   = rng.uniform(5.0, 10.0)     
  mp  = rng.uniform(-4.0, 2.0)
  # ss  = rng.uniform(-10.0, 10.0)
  # mps   = rng.uniform(-10.0, 10.0)
  # srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
  # srand = rng.uniform(0.0, 10.0) # should the MARK and matrix MLE start @ same value?
  # z = np.array([s, mp, ss, mps, srand])
  # z = np.array([s, mp])
  # print("calling randArgs2 - create params")
  z = lmfit.Parameters()
  # z.add('s', value=rng.uniform(5.0,10.0))
  # z.add('mp', value=rng.uniform(-4.0,2.0))
  z.add('mp', value=rng.uniform(0.001,0.1))
  z.add('mf', value=rng.uniform(0.001,0.1))
  # z.add('s', value=rng.uniform(0.7,0.98),max=1,min=0)
  # print(f"{1-z['s']=}")
  # z.add('mp', value=rng.uniform(0.0001,1-z['s']),max=1,min=0)
  # z.add('mf', expr='1-s-mp',min=0,max=1)
  # print(f"randArgs: {z=} {type(z['s'])=}")
  return(z)

# def hess()
# def mayfInit(mayfEstim, ):
def calc_se(arg, an):
  print("\t\t>> calculating hessian")
  dat,frq,storm,survey,conf=arg
  # smd_obj = functools.partial(like_smd,*arg)
  smd_obj = functools.partial(like_smd,
                              obsData=dat,
                              obsFreq=frq,
                              stormDays=storm,
                              surveyDays=survey,
                              config=conf)
  # hess = hessian(lambda z: like_smd(z,*arg),out.x)
  # hess = hessian(smd_obj,an)
  # hess = ndt.Hessian(smd_obj)(out.x)
  hess = ndt.Hessian(smd_obj)(an)
  hess2 = approx_hess2(an, smd_obj)
  print(f"{hess=} {type(hess)=}")
  print(f"{hess2=} {type(hess2)=}")
  # try:
  #   with warnings.catch_warnings(): #+> treeat warnings as exceptions here
  #     warnings.simplefilter("error")
  #     se = np.sqrt(np.diag(np.linalg.inv(hess)))
  # except RuntimeWarning as warn:
  #   print(f"{warn} - use pseudo-inverse")
  #   # logit_se = np.sqrt(np.diag(np.linalg.pinv(hess)))
  #   se = np.sqrt(np.diag(np.linalg.pinv(hess)))
  se = np.sqrt(np.diag(np.linalg.inv(hess)))
  seS = np.sqrt(np.sum(np.sum(np.linalg.inv(hess))))

  # print(f"{logit_se=} {type(logit_se)=}")
  # lse1, lse2 = logit_se
  # se = [logistic(lse1), logistic(lse2)]
  print(f"\t{se=} {type(se)=}")
  print(f"\t{seS=} {type(seS)=}")
  # return logit_se
  # return se
  return (se[0],se[1],seS)


# NOTE I have no real reason for choosing Nelder-Mead. Try another
# -----------------------------------------------------------------------------
#   CREATE NEST DATA AND RUN THE OPTIMIZER 
# -----------------------------------------------------------------------------
## +>loop thru param combinations; within loop, unpack params & run optimizer
# @profile
##+> add debug to print starting vals for basinhopping
# def run_optim(minimizer, fun, z, arg, met='Nelder-Mead', db=False):
def run_optim(minimizer, fun, z, arg, met='nelder', db=False):
  """
    Run scipy.optimize.minimize on 'fun'. Will return value of -1 or -2 if exceptions occur.

    If all is well, transform the output (using ansTransform() for MCMC model, and
    logistic() for MARK model)

    Returns:
      the transformed output.
  """
  # print("calling run_optim")
  counter=0
  print(f">> run_optim: {met=}\t{minimizer=}\t{fun=}")
  try:
    with warnings.catch_warnings(): #+> treeat warnings as exceptions here
      warnings.simplefilter("error")
      # print("calling choose_alg")

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
    # print("\nSuccess?", out.success, out.message, f"{out.x=}")
    # # outmin = out[0]
    # # hess   = out[0]
    # # se   = out[1]
    # print(f"\n{arg=}")
    # print(f"{len(arg)=}")
      # print(lmfit.fit_report(out))
    # res = out.x
    if minimizer in ["norm_lmfit","bh_lmfit"]:
      an = (out.params['s'].value,out.params['mp'].value)
      # res = ansTransform(ans=an)
      if db:
        print(f"{out.params=}")
        print(f"{out.params['s'].stderr=}")
        print(f"{out.params['mp'].stderr=}")
        # print(f"{out.params['mf'].stderr=}")
        print(f"{out.residual=}")
    else:
      an = out.x
      # res = ansTransform(ans=an)
      print(f"\t\t{out.x=}")
      # print(f" {out.hess_inv=} {type(out.hess_inv)=}")
      # hess = out.hess_inv ## probably not good for calculating se
      # logit_se = np.sqrt(np.diag(np.linalg.inv(hess)))
      # print(f"{logit_se=} {type(logit_se)=}")
    # se = ansTransform(ans=out.)
    # print(f"{res=}")
    # print(f"{type(out.x)=}{out.x.shape=}")
    # print(f"{out.x[1]=}")
    # funArr  =  like_smd(z,*arg)
    # funArr  =  [z,*arg]
    # hessFun = ndt.Hessian(funArr,full_output=True)
    # hessFun = ndt.Hessian(funArr['x'],full_output=True)
    # hessFun = ndt.Hessian(funArr['x'],full_output=True)
    # hessFun = ndt.Hessian(like_smd,full_output=True)
    # hessFun = ndt.Hessian(lambda x: like_smd(z,*arg))
    # hessFun = ndt.Hessian(lambda x: like_smd(z,*arg))
    # print(f"{hessFun=} {type(hessFun)=}")
    # hess, info = hessFun(out['x'])
    # hess = ndt.Hessian(lambda x: like_smd(z,*arg))(out.x)
    # print("creating partial function")
    # if calc_hess:
    #   print("calculating hessian")
    #   dat,frq,storm,survey,conf=arg
    #   # smd_obj = functools.partial(like_smd,*arg)
    #   smd_obj = functools.partial(like_smd,
    #                               obsData=dat,
    #                               obsFreq=frq,
    #                               stormDays=storm,
    #                               surveyDays=survey,
    #                               config=conf)
    #   # hess = hessian(lambda z: like_smd(z,*arg),out.x)
    #   # hess = hessian(smd_obj,an)
    #   # hess = ndt.Hessian(smd_obj)(out.x)
    #   hess = ndt.Hessian(smd_obj)(an)
    #   print(f"{hess=} {type(hess)=}")
    #   try:
    #     with warnings.catch_warnings(): #+> treeat warnings as exceptions here
    #       warnings.simplefilter("error")
    #       logit_se = np.sqrt(np.diag(np.linalg.inv(hess)))
    #   except RuntimeWarning as warn:
    #     print(f"{warn} - use pseudo-inverse")
    #     logit_se = np.sqrt(np.diag(np.linalg.pinv(hess)))
    #
    #   print(f"{logit_se=} {type(logit_se)=}")
      # se = ansTransform(logit_se)
    # res = ansTransform(ans=outmin.x,se=se)
    # hess = ndt.Hessian()
    # if res[1] < 0.6:
    #   print("run optimizer again with basinhopping")
    #   arg=
    #   try:
    #     out = optimize.minimize(fun, z, args=)
  # else:
    # res=ansTransform(ans, unpack=False)
    # res=ansTransform(ans=out)
    # res = logistic(out.x[0])
    # print("\t", res)
  # res = ()
  # print(f"\t|> |> |> {res=}")
  return an
  # return(an,logit_se)
  # return(res,se)

def constr(x):
  return sum(x)-1 ## equality constraint - must equal zero

# def choose_alg(minim, fun, z, arg, met, debug=False):
def choose_alg(minim, fun, z, arg, met, debug=False):
  print(f">> choose_alg: - {minim=} {z=} {len(z)=}")
  
  constrs = ({'type': 'eq', 'fun': constr}) ## type=equality
  # bounds = [(0.00001, 1),(0.00001,1),(0.00001,1)]
  bounds = [(0.000001,.1),(0.000001,.1)]
  if minim=="norm_scipy":
    
    if met in ["SLSQP", "trust-constr", "L-BFGS-B"]:
      # minimizer = optimize.minimize(fun,z,args=arg,method=met,bounds=bounds,constraints=constrs)
      minimizer = optimize.minimize(fun,z,args=arg,method=met,bounds=bounds)
    else:
      minimizer = optimize.minimize(fun, z, args=arg, method=met) 
    res=minimizer
    # if debug: print("\nSuccess?", minimizer.success, minimizer.message)
    # minimizer = optimize.minimize(fun,minimizer1.x,args=arg,method="BFGS")
    # if debug: print("\nSecond run - success?", minimizer1.success, minimizer1.message)
  elif minim=="norm_lmfit":
    minimizer = lmfit.minimize(fun, z, args=arg, method=met,calc_covar=True) 
    if debug: print("\nSuccess?", minimizer.success, minimizer.message)
    # with np.printoptions(precision=12): print(f"{minimizer.x=}{minimizer.params=}")
    with np.printoptions(precision=12): print(f"{minimizer.params=}")
    # with np.printoptions(precision=12): print(f"{minimizer.x=}")
    if debug: print(lmfit.fit_report(minimizer))
    try:
      with np.printoptions(precision=12): print(lmfit.conf_interval(lmfit.minimize,minimizer))
    except lmfit.minimizer.MinimizerException as error:
      print(f"ERROR: {error}")
    # with np.printoptions(precision=12): print(lmfit.conf_interval(lmfit.minimize,minimizer))
    ##NOTE doesn't work bc 'leastsq' requires len(out) >= len(inp)
    # z = minimizer.params
    # print(f"{z=} {len(z)=}")
    # minimizer = lmfit.minimize(fun, z, args=arg, method='leastsq') 
  elif minim=="bh_scipy":
    if met in ["SLSQP", "trust-constr","L-BFGS-B"]:
      # min_kwargs={"args": arg, "method": met, "bounds": bounds, "constraints": constrs} ## will run the local optimization 
      min_kwargs={"args": arg, "method": met, "bounds": bounds} ## will run the local optimization 
    else:
      min_kwargs={"args": arg, "method": met} ## will run the local optimization 
    minimizer = optimize.basinhopping(fun, z, minimizer_kwargs=min_kwargs)
    res = minimizer.lowest_optimization_result
    
    # if debug: print("\n<bh>Success?", minimizer.success, minimizer.message)
    # minimizer = optimize.minimize(fun,minimizer1.x,args=arg,method="BFGS")
    # if debug: print("\n<bh>Second run - success?", minimizer1.success, minimizer1.message)
  elif minim=="bh_lmfit":
    # print(f"{z=} {len(z)=}")
    minimizer = lmfit.minimize(fun, z, args=arg, method='basinhopping',calc_covar=True) 
    try:
      with np.printoptions(precision=12): print(lmfit.conf_interval(lmfit.minimize,minimizer))
    except lmfit.minimizer.MinimizerException as error:
      print(f"ERROR: {error}")

    # z = minimizer.params
    # print(f"{z=} {len(z)=}")
    # minimizer = lmfit.minimize(fun, z, args=arg, method='leastsq') 
    # if debug: print(f"{z=}", end=" ")
    

  # funArr  = 
  # hessFun = ndt.Hessian(fun,full_output=True)
  # # print(f"{hessFun=} {type(hessFun)=}")
  # hess, info = hessFun(minimizer['x'])
  # # print(f"{hess=} {type(hess)=}")
  # se = np.sqrt(np.diag(np.linalg.inv(hess)))
  # print(f"{se=} {type(se)=}")
  return res
  # return(minimizer)
  # return [minimizer,hess]
  # return (minimizer,hess)
  # return (minimizer,se)

  
def ansTransform(ans):
# def ansTransform(ans,se):
  """
    Transform the optimizer output so that it is between 0 and 1, and the 3 
    probabilities sum to 1. 

    'ans' is an object of type 'OptimizeResult', which has a number of components
  """
  # print("calling ansTransform")
  # if unpack:
    # ans = ans.x  
  # s0   = ans.x[0]     # Series of transformations of optimizer output.
  # ss0  = ans[2]     # and that the three fate probabilities sum to 1.
  # mps0 = ans[3]
  print(f"ansTransform: transform output {ans=}")
  # est, se = ans
  #
  # if log_transform:
  #   s0   = est[0]     # Series of transformations of optimizer output.
  #   mp0  = est[1]     # These make sure the output is between 0 and 1, 
  #   s1   = logistic(s0)
  #   mp1  = logistic(mp0)
  #   sse0   = se[0]     # Series of transformations of optimizer output.
  #   mpse0  = se[1]     # These make sure the output is between 0 and 1, 
  #   sse1   = logistic(sse0)
  #   mpse1  = logistic(mpse0)
  # else:
  #   s1   = est[0]     # Series of transformations of optimizer output.
  #   mp1  = est[1]     # These make sure the output is between 0 and 1, 
  #   sse1   = se[0]     # Series of transformations of optimizer output.
  #   mpse1  = se[1]     # These make sure the output is between 0 and 1, 

  if log_transform:
    s0   = ans[0]     # Series of transformations of optimizer output.
    mp0  = ans[1]     # These make sure the output is between 0 and 1, 
    s1   = logistic(s0)
    mp1  = logistic(mp0)
  else:
    s1   = ans[0]     # Series of transformations of optimizer output.
    mp1  = ans[1]     # These make sure the output is between 0 and 1, 

  # ss1  = logistic(ss0)
  # mps1 = logistic(mps0)

  ret2 = triangle(s1, mp1)
  # ret2 = triangle(s0, mp0)
  s2   = ret2[0]
  mp2  = ret2[1]
  mf2  = 1.0 - s2 - mp2

  # ret3 = triangle(ss1, mps1)
  # ss2  = ret3[0]
  # mps2 = ret3[1]
  # mfs2 = 1.0 - ss2 - mps2
  
  # ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.longdouble)
  ansTransformed = np.array([s2, mp2, mf2], dtype=np.longdouble)
  # ansTransformed = np.array([s2, mp2, mf2, sse1, mpse1], dtype=np.longdouble)
  # ansTransformed = np.array([s2, mp2, mf2, se], dtype=np.longdouble)
  return(ansTransformed)

# -----------------------------------------------------------------------------

#@profile
def rep_loop(par,rng, nData, storm, survey, config, random=True,to_r=False):
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
  if isinstance(nData, pd.DataFrame):
    nData = nData.to_numpy()
  dat = nData[:, np.r_[0,4:10]] # doesn't include column index 10
  # if config.debug>=2:
    # print("check:")
    # arrPrint(dat)
  # arg=(dat, par.obsFreq, par.useSMat, storm, survey, par.whichLike, config)
  arg=(dat, par.obsFreq, storm, survey, config)
  # zargs = randArgs() if random==True else mayfInit()
  # TODO: integrate warning capture with function in helpers.py
  if config.optimFunc=="scipy":
    init = randArgs(config,rng)
    minimSuff = "scipy"
  else:
    init = randArgs2(config,rng)
    minimSuff = "lmfit"
  bh_minim = "bh_" + minimSuff
  norm_minim = "norm_" + minimSuff
  # if config.optimizer=="global":
  if config.optimGlob:
    # res    = run_optim(minimizer="bh",
    # print(f">> rep_loop: RUN OPTIMIZER w/ {bh_minim}")
    print(f">> rep_loop: RUN OPTIMIZER w/ ", end= " ")
    out    = run_optim(minimizer=bh_minim,
                       fun=like_smd,
                       # z=randArgs(config,rng),
                       # z=randArgs2(config,rng),
                       z=init,
                       arg=arg,
                       met=config.optimizer,
                       )
  else:
    # print("\nrep_loop: RUN OPTIMIZER\n")
    # res    = run_optim(minimizer="norm",
    # print(f">> rep_loop: RUN OPTIMIZER w/ {norm_minim}")
    print(f">> rep_loop: RUN OPTIMIZER w/ ", end= " ")
    out    = run_optim(minimizer=norm_minim,
                       fun=like_smd,
                       # z=randArgs2(config,rng),
                       z=init,
                       arg=arg,
                       met=config.optimizer,
                       )
  # res = ansTransform(out)
  res = out
  print(res)
  if False:
    if res[0] < 0.8:
      print(f"\t\t\t\t>> rep_loop: {res[0]=:.4f}; run optimizer again with basinhopping; ", end=" ")
      if config.optimFunc=="scipy":
        init = randArgs(config,rng)
      else:
        init = randArgs2(config,rng)
      # res = run_optim(minimizer="bh",fun=like_smd,z=randArgs2(config,rng),arg=arg,db=True)
      # res = run_optim(minimizer=bh_minim,fun=like_smd,z=init,arg=arg,met="basinhopping",db=True)
      out = run_optim(minimizer=bh_minim,fun=like_smd,z=init,arg=arg,met=config.optimizer,db=True)
      # res = ansTransform(out)
      res = out
      # print(f"\t\t|>NEW {res[0]=:.4f}")
      if res[0] < 0.8:
        # discover = nData[nData[:,6]!=0]
        # discover = nData[nData[:,8]>0] ## +> n obs > 0 - obs before fail
        # discover = nData[nData[:,6]>0] ## +> k > 0
        #+> but nobs is now num obs while active, so some failed nests have nobs=0
        ## but using total obs is somehoww leading to larger overestimate? or is it?
        discover = nData[nData[:,10]>0] ## +> TOTAL obs > 0
        discovered = discover.shape[0]
        # if config.debug>=3:
          # print(f"discovered nests ({discover.shape=}):")
          # dfPrint(discover)
        # excl = ((discover[:,7] == 7) | (discover[:,4]==discover[:,5]))
        # excl = ((discover[:,7] == 7) | discover[:,8]>0)
        excl = (discover[:,7] == 7)
        excluded  = np.sum(excl)            
        hatched = np.sum(discover[:,3]==0)
        unknown = np.sum(discover[:,7]==7)
        # print(f"\t\t\t\t\t{discovered=}|>{excluded=}&{hatched=}&{unknown=}", end=" ")
        print(f"{res[0]=:.4f}; AGAIN with bh", end=" ")
        # res0 = run_optim(minimizer="bh",fun=like_smd,z=randArgs2(config,rng),arg=arg,db=True)
        
        if config.optimFunc=="scipy":
          out0 = run_optim(minimizer=bh_minim,fun=like_smd,z=randArgs(config,rng),arg=arg,met=config.optimizer,db=True)
          out1 = run_optim(minimizer=bh_minim,fun=like_smd,z=randArgs(config,rng),arg=arg,met=config.optimizer,db=True)
          out2 = run_optim(minimizer=bh_minim,fun=like_smd,z=randArgs(config,rng),arg=arg,met=config.optimizer,db=True)
        else:
          out0 = run_optim(minimizer=bh_minim,fun=like_smd,z=randArgs2(config,rng),arg=arg,met=config.optimizer,db=True)
          out1 = run_optim(minimizer=bh_minim,fun=like_smd,z=randArgs2(config,rng),arg=arg,met=config.optimizer,db=True)
          out2 = run_optim(minimizer=bh_minim,fun=like_smd,z=randArgs2(config,rng),arg=arg,met=config.optimizer,db=True)
        # res0 = ansTransform(out0)
        # res1 = ansTransform(out1)
        # res2 = ansTransform(out2)
        res0 = out0
        res1 = out1
        res2 = out2
        # print(f"\t\t\t\t\t|>NEW {res0[0]=:.4f}{res1[0]=:.4f}{res2[0]=:.4f}", end=" ")
        # resList = np.array(res1, res2, res3)
        # resArr= np.concatenate((res1, res2, res3), axis=0)
        first_vals = [arr[0] for arr in [res0, res1, res2]]
        
        resName = "res" + str(first_vals.index(max(first_vals)))
        # res = eval(resName)
        res = locals()[resName]
        # res = resArr[np.argmax(resArr[:,0])]

        # res = np.max([res1,res2,res3])
        print(f"\t\t\t\t\t|>NEW {res[0]=:.4f}")
    # srand = rng.uniform(-10.00, 10.00)
    # s2, mp2, sse, mpse = res[0], res[1], res[3], res[4]

  # s2, mp2 = res[0], res[1]
  mf2, mp2 = res[0], res[1]
  s2 = 1 - mf2 - mp2
  print(f"*** {s2=} ***")
  # if config.debug>=4:
  #   print(f"{s2=} {mp2=}")
  psr = s2 ** par.hatchTime
  se = calc_se(arg, out)
  se1, se2, se3 = se

  if False:
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
    like_val = np.array([ mark_s,s2,mp2], dtype=np.longdouble)
  #~#if config.debugLL>=2: print(f"\t\t>> like_val: MARK={like_val[0]}, MCMC-surv={like_val[1]}, MCMC-pred={like_val[2]}")
  if to_r:
    like_val = [s2,psr,mp2,se1,se2]
  else:
    like_val = np.array([s2,psr,mp2,se1,se2], dtype=np.longdouble)
    # like_val = 
  if config.testing=="yes":
    print(f"{out=} {res=}")
    # np.savetxt("outval.txt")
    with open("outval.txt", "a") as f:
      np.savetxt(f,[res], delimiter=" ")
      # f.write("\n")
      np.savetxt(f,[out], delimiter=" ")
      f.write("\n")
    #   f.write(out)
    #   f.write(res)
      # f.write(res)
  return(like_val)
  
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

def r_logexp():
  """
  """
  # r['source']('logexp.R')
  # r.source('logexp.R')
  r['pi']

# def main(fnUnique, testing, parLists,stormdays=[], msg="", config=config, pStatic=staticPar):
# def main(fnUnique,atype,rng,nowStr, testing,config, parLists,pStatic,stormdays=[], msg=""):
def main(atype,config,rng, parLists,pStatic,nowStr,stormdays=[]):
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
  # rng = np.random.default_rng(seed=config.rngSeed)
  msg=config.msg
  testing=config.testing
  fnUnique=config.fnUnique
  pList = parLists
  # now_str = now_short
  now_str = nowStr
  odir  = mk_outdir(now_str, con=config)
  # print(f"\t|> output directory = {odir}")
  if len(msg) > 0:
    print("|> MSG: ", msg)
  print(f"\t\t<>CONFIG: {config}")
  if fnUnique:
    fname = mk_fnames(now_str, fdir=odir, suf=f"{atype}", uniq=True, con=config) 
  else:
    fname = mk_fnames(now_str,fdir=odir,suf=f"{atype}", con=config)
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
    paramsArray = mk_param_list_list(parL=pList, fdir=odir, suf=f"{config.rngSeed}{atype}")
    print(
        f"\n\t|>|>|>{len(paramsArray)} param sets x {config.nreps} reps ="
        f" {len(paramsArray)*config.nreps} total rows"
        )
    parID     = 0
    # summVars = ['stfrq','obfrq','inc','sfate','dsc','ha','fl','unk','exc','mc',
    summVars = ['dsc','ha','fl','unk','exc','mc',
                'dsrT','dsrA','dsrD','dsrC','diffA','diffD','diffC']
    summMat   = np.zeros(shape=(len(paramsArray), config.nreps, len(summVars)))
    # summMat   = np.zeros(shape=(len(paramsArray), config.nreps, 14))
    # print(f"dimensions of summary matrix: {summMat.shape}")
    for i in range(0, len(paramsArray)): # +> for each set of params
      par    = paramsArray[i] 
      par_merge  = {**par, **pStatic}
      par    = Params(**par_merge)
      print("\n\t<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
      print(f"\n>>>> param set #{parID} & {par=}\n")
      if len(stormdays)>0:
        stormDays=stormdays
        print("\t>> using storm days from arg")
        print(f"\t\t{stormDays=}")
      else:
        print("\t>> making storm days ")
        stormDays  = stormGen(par.stormFrq, par.stormDur,config,rng)
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
      nweek = np.round(par.brDays/7)-1 # for test w/ shorter season
      for r in range(config.nreps): 
        print(f"\t>---->----> replicate  {parID}.{repID} >---->----> ")
        try:
          # nestData1 = make_obs(par=par,storm=stormDays,survey=survey,nw=nweek,conf=config) 
          nestData1 = make_obs(par,rng,stormDays,survey,config,nw=nweek) 
        except IndexError as error:
          print(
            "\t\t>> !!! IndexError in nest data:", 
            error,
            ". Go to next replicate")
          traceback.print_exc()
          nEx = nEx + 1
          continue

        #~----------------------------------------------------------------------
        # +> save each rep as separate file-useful for loading into R
        if config.saveNData:
          print("\n\t\t!!!!! SAVING THIS REPLICATE'S NEST DATA TO FILE")
          ndName = Path(f"{odir}/nd_{config.rngSeed}{atype}/nd_p{parID:02}_r{repID:02}.npy")
          #   ndName = Path (f"{odir}/nests{config.rngSeed}_{atype}/nd_par{parID:03}.npy")
          ndName.parent.mkdir(parents=True, exist_ok=True)
          # np.save(ndName, nestData1)
          # rep_col = np.full(len(nestData1), repID)
          # rep_col = rep_col[:,np.newaxis]
          # nd2 = np.hstack([nestData1, rep_col])
          # np.save(ndName, nd2)
          np.save(ndName, nestData1)
          # ndMatrix[r,:,:] = nd2
        #
        #~----------------------------------------------------------------------
        ## +> print all nest data
        if config.debug>=3:
          print("\t\t\t|> ALL NEST DATA")
          nm = ["ID","init","end","fate"," i "," j "," k ","afate","nobs","fint"]
          # nm = ["ID","init","end","fate"," i "," j "," k ","afate","nobs","fint","nstm"]
          # print_nestdata(nestData1, names=nm,nprint=50) 
          # print("\nall nests:")
          print_nestdata(nestData1, names=nm, abbrv=False) 

        if config.debug>=1: 
          print("\n\t\t[*] [*] [*] [*] [*] calculating DSR [*] [*] [*] [*] [*] [*] [*] [*] ")

        #~----------------------------------------------------------------------
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
        # if config.obsSave:
        #   nestObs = nestData[:,np.r_[0,1,4:8,11]]
        #   nNest = nestData.shape[0]
        #   expoList = calc_daily_expo(nNest, survey[1], survey[2], nestData[:,4], nestData[:,6],db=config.debugNests)
        #   obsDat = make_daily_logex_df(nestObs,expos=expoList[1],covar1=expoList[2],db=config.debugNests)

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
        lVal = rep_loop(par=par,rng=rng, nData=nestData, storm=stormDays,
                   survey=survey,config=config)
        # llDSR = lVal[0]
        llDSR,llPSR,llDFR = lVal

        #~----------------------------------------------------------------------
        # if config.debugM>=2: ## +> matches the level for saving the info to print
        #   ##+> enable the saving in the prog_mark() definition
        #   prExp = True if config.debugM>=3 else False
        #   print_mark(print_exp=prExp)
        #
        #~----------------------------------------------------------------------
        if config.debugLL>=2: ## +> matches the level for saving the info to print
        #   ##+> print LL equations for each nest:
        #   ##+> enable the saving inside hte logLike function definition
        #
          printLL()
          # llArg = np.load('out/arg_PrintLL.npy') 
          # printLL(len(llArg), *llArg.T) # +> tranpose so it is unpacked colwise
        #
        #~----------------------------------------------------------------------
        # if config.testing == "yes":
        #   print(f"\n\t\t\t\t{mayfDSR_all=:.3f}, {mayfDSR_disc=:.3f}, {mayfDSR_an=:.3f}")
        #   print(f"\t\t\t\t{appDSR=:.3f}, {appDSR_disc=:.3f}, {appDSR_an=:.3f}")
        #

        # #   ##+> print summary:
        if config.debug>=2: ## +> matches the level for saving the info to print
          sum_list = [
              hatched,
              flooded,
              discover,
              unknown,
              misclass,
              exclude,
              llDSR,
              llPSR,
              # lVal[0],
              # lVal[1],
              mayfDSR_an,
              appDSR_an,
              appDSR,
              appDSR_disc,
              ]
          print_all(sum_list, nestData, par)
        #----------------------------------------------------------------------

        nVal = np.array([appDSR,        #4
                         appDSR_disc,
                         appDSR_an,
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

        # if parID == 0 and like_val[12] == 0: #+> only 1st line gets the header
        if parID == 0 and repID == 0: #+> only 1st line gets the header
          np.savetxt(f, [like_val], delimiter=",", header=colnames)
          # if debug: print(">> ** saving likelihood values with header **")
        else:
          np.savetxt(f, [like_val], delimiter=",")
          # if debug: print(">> ** saving likelihood values **")
        # need to save it in the function where f was opened?
        # likeVal[r] = like_val

      #~----------------------------------------------------------------------
        ### +> save vals to summary matrix for printing:
        if config.testing=="yes":
          # print("saving to matrix")
          # summVars = ['dsc','ha','fl','unk','exc','mc','dsrT','dsrA','dsrD','dsrC']
          # summVal =np.array([ discover, hatched, flooded, unknown, exclude, misclass, trueDSR, trueDSR_an, trueDSR_disc, lVal[1] ])
          summVal =np.array([
            # par.stormFrq,
            # par.obsFreq,
            # par.hatchTime,
            # par.stormFate,

            discover.sum(),
            hatched.sum(),
            flooded.sum(),
            unknown.sum(),
            exclude.sum(),
            misclass.sum(),
            appDSR,
            appDSR_an,
            appDSR_disc,
            llDSR,
            llDSR-appDSR,
            llDSR-appDSR_an,
            llDSR-appDSR_disc,
            ])
          # print(f" {summMat[i,r,:].shape=} | {summVal.shape=}")

          summMat[i,r,:] = np.array(summVal)

          ##+> also, plot initiation dates:

          # plt.hist(nestData[:,1])
          # kde = gaussian_kde(nestData[:,1])
          # distr = np.linspace(np.min(nestData[:,1]), np.max(nestData[:,1]),100)
          # plt.plot(distr,kde(distr),alpha=0.5)
          # plt.title("initiation dates")
      #----------------------------------------------------------------------

        ##NOTE don't comment! need to increment the repID
        repID = repID + 1
        
      #~----------------------------------------------------------------------
      # if debug>=2: arrPrint(ndMatrix)
      # if config.testing=="yes":

        #+> save initiation dates plot to file
        # figName = f"figs/{parID}inits_density.png"
        # initDates = pd.DataFrame([initDat])
        # kde = gaussian_kde(initDates)
        # distr = np.linspace(np.min(initDates), np.max(initDates),100)
        # plt.plot(distr,kde(distr),alpha=0.8,color='blue')
        # plt.savefig(figName)
        # plt.close()

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
    if config.testing=="yes":
      ## +> get means of summary matrix:
      meanMat = summMat.mean(axis=1)
      svars = ', '.join([str(x) for x in summVars]) # needs to be string
      pListN = mk_param_list_list(pList, listRet=True)
    #   # NOTE probably could make df out of a list of dicts anyway
      print(pListN)
    #   # pListNew = [pl[0,1,5,9] for pl in pListNew] #+> doesn't work!
    #   # pListNew = [pl[i] for pl in pListNew for i in [0,1,5,9]]
    #   # for pl in pListNew:
      # pListNew = [[pl[0], pl[1], pl[5], pl[9]] for pl in pListN]
      pListNew = [[pl[0], pl[1], pl[8], pl[12]] for pl in pListN]
    #
      print(pListNew)
    #   # pnames = list(dir(Params)) ##+> this one gives all components of class
    #   # pnames = list(vars(par).keys()) ##+> this one includes staticPar
    #   # pnames = list(paramsArray[0].keys()) ##+> get the names from the dict?
    #   # pnames = [pnames[i] for i in [0,1,5,9]]
    #   # print(pnames)
      pnames = ["nNest", "pSurv", "obsInt", "hTime"]
    #
      fname_mean = f"out/mean_mat_{config.rngSeed}{atype}.csv"
      np.savetxt(fname_mean, meanMat, fmt='%.5f', delimiter=",", header=svars)
      # if config.debug>=2: arrPrint(summMat) ## +> print entire matrix
      print("==>> mean values for each param set: ")
    #   # arrPrint(meanMat)
    #   # dfPrint([meanMat,pListNew], names=[summVars,pnames])
      dfPrint([meanMat,pListNew],abbr=False, concat="cwise", names=[summVars,pnames])
    #----------------------------------------------------------------------

# def make_nestdat(par, stormDays, survey, config, nWeeks, initFromFile):
#     nEx =0
#
#     try:
#       nestData1 = make_obs(par=par,
#                            storm=stormDays,
#                            survey=survey,
#                            conf=config,
#                            nw = nWeeks,
#                            inff = initFromFile
#                            ) 
#     except IndexError as error:
#       print(
#         "\t\t>> !!! IndexError in nest data:", 
#         error,
#         ". Go to next replicate")
#       nEx = nEx + 1
#       # continue
#       return 1
#
#     # +> calculate true DSR (apparent DSR w/ real numbers):
#     appDSR  = calc_dsr(nData=nestData1,
#                         nestType="all",
#                         calcType="apparent",
#                         conf=config,
#                         incTime=par.hatchTime,
#                         psurv=par.probSurv,
#                         debug=config.debugSummary)
#
#     flooded  = sum(nestData1[:,3]==2)
#     hatched  = sum(nestData1[:,3]==0)
#     discover = nestData1[:,8]>0
#
#     nestData = nestData1[(discover),:] # +> remove undiscovered nests
#     exclude  = ((nestData[:,7] == 7))
#     unknown  = (nestData[:,7]==7)
#     misclass = (nestData[:,7]!=nestData[:,3]) #+> out of discovered nests
#     avgFInt  = (nestData[:,9].sum()/len(discover))
#     avgK     = nestData[:,6].sum()/len(discover)
#
#     nestData  = nestData[~(exclude),:]  # +> remove excluded nests 
#     nestVals = np.array([
#       flooded,hatched,discover.sum(),exclude.sum(),unknown.sum(),
#       misclass.sum(), avgFInt, avgK, appDSR, mark_s, repID, parID])
