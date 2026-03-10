#!/usr/local/bin/python


# sudo vim -o file1 file2 [open 2 files] 
# BLAH
# :/^[^#]/ search for uncommented lines
# /^[^#]*\s*print  or /^\s*print 
# > kernprof -l simdata_vect.py > 20sepprofile.out
# > python -m line_profiler .\simdata_vect.py.lprof
# NOTE 5/16/25 - The percent bias responds more like I would expect when I use
#                the actual calculated DSR, not the assigned DSR (0.93 or 0.95)
#                BUT I still don't know why the calculated DSR is consistently low.

# import argparse

import numpy as np 
import scipy.stats as stats
import csv
import decimal
import itertools
import os
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
from settings import rng, config, staticPar, pLists

# config = load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)

from helpers import mk_param_list, mk_fnames
from makeNests import stormGen
from observer import make_obs, mk_surveys
from dsrCalc import calc_dsr, mark_wrapper
from MCmatrix import like_smd, triangle, logistic

# NOTE: 
# 1. turned off useSM (params)
# 2. output is 9 columns for now
# -----------------------------------------------------------------------------
#  SETTINGS 
# -----------------------------------------------------------------------------
# rng = np.random.default_rng(seed=102891)
# rng = np.random.default_rng(seed=config.rngSeed)

# debugTypes = None # output = None verbose = False
# print("debug options:", debugTypes)
debug = config.debug
# print(debug)

# now        = datetime.today().strftime('%m%d%Y_%H%M%S')
# config.fnUnique   = True
# storm_init = "/home/wodehouse/projects/sim_model/storm_init3.csv"
# like_f_dir = "/home/wodehouse/projects/sim_model/out"
# if config.useWSL:


def randArgs():
    """
    Choose random initial values for the optimizer.
    These will be log-transformed before going through the likelihood function
    
    Returns:
    array of s, mp, ss, mps (for like_smd) and srand (for mark_wrapper)
    """
    s     = rng.uniform(-10.0, 10.0)       
    mp    = rng.uniform(-10.0, 10.0)
    # ss    = rng.uniform(-10.0, 10.0)
    # mps   = rng.uniform(-10.0, 10.0)
    srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
    # z = np.array([s, mp, ss, mps, srand])
    z = np.array([s, mp])

    return(z)
# -----------------------------------------------------------------------------
#   CREATE NEST DATA AND RUN THE OPTIMIZER 
# -----------------------------------------------------------------------------
# need to loop through the param combinations
# within the loop, need to unpack the params and run the optimizer
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
        # out = optimize.minimize(fun, z, args=arg, method=met)
        # out = minimizer(fun, z, args=arg, method=met)
        out = choose_alg(minimizer, fun, z, arg, met)

        ex = 0.0
    except decimal.InvalidOperation as error2:
        # ex=100.0
        ex=-1.0
        print(">> Error: invalid operation in decimal:", error2, "Go to next replicate.")
        return(ex)
        # continue
    except OverflowError as error3:
        # ex=200.0
        ex=-2.0
        print(
            ">> Error: overflow error:", 
            error3, 
            "Go to next replicate."
            )
        return(ex)
        # continue
    #
    # print("Success?", out.success, out.message, "answer=", out.x)
    if fun==like_smd: 
        # print("Success?", out.success, out.message, "answer=", out.x)
        res = ansTransform(ans=out.x)
        # if res[1] < 0.6:
        #     print("run optimizer again with basinhopping")
        #     arg=
        #     try:
        #         out = optimize.minimize(fun, z, args=)
    else:
        # res=ansTransform(ans, unpack=False)
        # res=ansTransform(ans=out)
        res = logistic(out.x[0])
    return(res)

def choose_alg(minim, fun, z, arg, met):
    if minim=="norm":
        minimizer = optimize.minimize(fun, z, args=arg, method=met) 
    elif minim=="bh":
        min_kwargs={"args": arg}
        minimizer = optimize.basinhopping(fun, z, minimizer_kwargs=min_kwargs)
        
    return(minimizer)
# def make_obs(par, init, dfs, stormDays, surveyDays, config=config):
# @profile
    
def ansTransform(ans):
    """
    Transform the optimizer output so that it is between 0 and 1, and the 3 
    probabilities sum to 1. 

    'ans' is an object of type 'OptimizeResult', which has a number of components
    """
    # if unpack:
        # ans = ans.x    
    # s0   = ans.x[0]         # Series of transformations of optimizer output.
    s0   = ans[0]         # Series of transformations of optimizer output.
    mp0  = ans[1]         # These make sure the output is between 0 and 1, 
    # ss0  = ans[2]         # and that the three fate probabilities sum to 1.
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
    
    #ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.float128)
    # ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.longdouble)
    ansTransformed = np.array([s2, mp2, mf2], dtype=np.longdouble)
    # print(">> results as an array:\n", ansTransformed)
    # if debug: print(">> results (s, mp, mf, ss, mps, mfs, ex):\n", s2, mp2, mf2, ss2, mps2, mfs2, ex)
    return(ansTransformed)

# -----------------------------------------------------------------------------

# def rep_loop(par, repID, nData, vals, storm, survey, config):
# def rep_loop(par, nData, vals, storm, survey, config):
# @profile
def rep_loop(par, nData, storm, survey, config):
    """
    For each data replicate, call this function, which:
        - takes the reduced nest data as input
        - calls the optimizer on like_smd() and mark_wrapper()
        
    Returns:
        like_val
        [0]
    """
    # ---- empty array to store data for this replicate: ---------
    # like_val  = np.zeros(shape=(config.numOut), dtype=np.longdouble)
    # like_val  = np.zeros(shape=(3), dtype=np.longdouble)
    # perfectInfo = 0
    # whichL = par.whichLike
    # ex = np.longdouble("0.0")
    # stMat = state_vect(nNest=len(nData), fl=(nData[:,3]==2), ha=(nData[:,3]==0))
    dat = nData[:,4:10] # doesn't include column index 10
    # ans      = run_optim(fun=like_smd, z=randArgs(), 
    arg=(dat, par.obsFreq, par.useSMat, storm, survey, par.whichLike, config)
    res      = run_optim(minimizer="norm", fun=like_smd, z=randArgs(), arg=arg)
    # res      = run_optim(minimizer = optimize.minimize, fun=like_smd, z=randArgs(), 
                        #  arg=(dat, par.obsFreq, stMat, par.useSMat, storm, survey, 2))
                        #  arg=(dat, par.obsFreq, stMat, par.useSMat, storm, survey, 1))
                        #  arg=(dat, par.obsFreq, stMat, par.useSMat, storm, survey, par.whichLike))
                        #  )
                                # args=( dat, obsFr, stMat, useSM, 1),
    if res[0] < 0.6:
        if debug>=3: print(f"res={res[0]}; run optimizer again with basinhopping")
        # run_optim(fun=like_smd, z=randArgs(), arg=(dat,par.obsFreq,par.useSMat,storm,survey,par.whichLike),met=))
        res = run_optim(minimizer="bh", fun=like_smd, z=randArgs(), arg=arg)
    # res = ansTransform(ans)
    srand = rng.uniform(-10.00, 10.00)
    # markProb = mark_probs(s=srand, ndata=nData)
    # mark_s = run_optim(fun=mark_wrapper, z=srand, arg=(nData, markProb, par.brDays))
    mark_s = run_optim(minimizer="norm", fun=mark_wrapper, z=srand, arg=(nData, par.brDays, config))
    #NOTE ans2 is an "OptimizeResult" object; need to extract "x"
    # Transform the MARK optimizer output so that it is between 0 and 1:
    # mark_s = logistic(ans2.x[0]) # answer.x is a list itself - need to index
    # print("> logistic of MARK answer:", mark_s)
    # s2,mp2,mf2,ss2,mps2,mfs2 = res # unpack like() function output
    # NOTE scott was probably right - mps doesn't make sense. and DSR includes storms already
    # so check whether mort flood probability goes up with more intense storms?
    s2, mp2 = res[0], res[1]
    # mp2 = res[1]

    # s2,mp2,mf2,ss2,mps2,mfs2 = res2 # unpack like_old() function output
    # like_val = [ mark_s,s2,mp2,mf2,ss2,mps2,mfs2]
    like_val = np.array([ mark_s,s2,mp2], dtype=np.longdouble)
    if config.debugLL>=3: print(">> like_val:\n", like_val)
    # if config.debugLL: print(">> like_val:\n", like_val)
    # like_val = np.array(like_val, dtype=np.longdouble)
    return(like_val)
    
# def save_vals(parID, repID, like_val, ):
# def main(testing=False, fname=mk_fnames(), pStatic=staticPar):
# def main(testing=False, config=config, pStatic=staticPar):
# def set_debug(deb, debN, debS, config=config):
# def set_debug(deb=debugTypes):
# def main(fnUnique, debugOpt, testing, config=config, pStatic=staticPar):
def main(fnUnique, testing, parLists, config=config, pStatic=staticPar):
    """
    If 'fnUnique'==True, filename is "uniquified" and includes H:M:S
        Otherwise, just the date.
    """
    lf_suffix=""
    pList = parLists
    # these if-else statements only run once:
    # if debugOpt != None:
    # # if deb:
    # # if hasattr(args,"")
    #     # set_debug(debugTypes)
    #     set_debug(debugOpt)
    # fname = mk_fnames(like_f_dir=like_f_dir) if fnUnique else mk_fnames(unique=False)
    fname = mk_fnames(suf = lf_suffix) if fnUnique else mk_fnames(suf =lf_suffix, unique=False)
    fdir  = fname[0].parent
    # fdir  = os.path.split(fname[0])[0]
    # print(fdir)
    # config.likeFile = fname[0]
    likeFile = fname[0]
    # config.colNames = fname[1]
    colNames = fname[1]
    # print(config)
    with open(likeFile, "wb") as f: # doesn't need to be 'a' bc file is open
        paramsArray = mk_param_list(parList=pList, fdir=fdir)
        # if debug: print(f">>>> there will be {len(paramsArray)*config.nreps} total rows")
        print(f"\n|>|>|>there will be {len(paramsArray)} param sets & {len(paramsArray)*config.nreps} total rows")
        parID       = 0
        for i in range(0, len(paramsArray)): # for each set of params
            # if debug: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            par        = paramsArray[i] 
            par_merge  = {**par, **pStatic}
            par        = Params(**par_merge)
            print(">>>>>>>>> param set number:", parID, "and params in set:", par)
            stormDays  = stormGen(frq=par.stormFrq, dur=par.stormDur)
            survey     = mk_surveys(stormDays, par.obsFreq, par.brDays, conf=config)
            # surveyDays, surveyInts = survey
            repID = numMC = nEx = 0 # number of nests misclassified, number of exceptions
            # likeVal    = np.zeros(shape=(config.nreps,config.numOut))
            for r in range(config.nreps): 
                # if debug: print("\n>>>>>>>>>>>> replicate ID: >>>>>>>>>>>>>>>>>>>>>>>>>>", repID)
                try:
                    nestData1 = make_obs(par=par, storm=stormDays, survey=survey, conf=config) 
                # except:
                except IndexError as error:
                    print(
                        ">> IndexError in nest data:", 
                        error,
                        ". Go to next replicate")
                    nEx = nEx + 1
                    # nestData = np.zeros((par.numNests, 11))
                    # return(nestData)
                    continue

                trueDSR  = calc_dsr(nData=nestData1, nestType="all", calcType="true", conf=config)
                flooded  = sum(nestData1[:,3]==2)
                hatched  = sum(nestData1[:,3]==0)
                # print_prop(nestData[:,7], nestData[:,3], )
                discover = nestData1[:,6]!=0
                nestData = nestData1[(discover),:] # remove undiscovered nests
                exclude  = ((nestData[:,7] == 7) | (nestData[:,4]==nestData[:,5]))                         
                unknown  = (nestData[:,7]==7)
                misclass = (nestData[:,7]!=nestData[:,3])
                nestData    = nestData[~(exclude),:]    # remove excluded nests 
                trueDSR_an   = calc_dsr(nData=nestData, nestType="analysis", calcType="true", conf=config) 
                lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
                                   survey=survey,config=config)
                # pars = np.array([par.probSurv, par.stormDur, par.stormFrq, par.numNests, 
                #                  par.hatchTime,par.obsFreq]) 
                # nVal = np.array([trueDSR, trueDSR_an, sum(discover), sum(exclude), repID])  
                nVal = np.array([trueDSR, trueDSR_an, sum(discover), sum(exclude), sum(unknown), sum(misclass), flooded, hatched, nEx, repID, parID])  
                like_val = np.concatenate((lVal, nVal))
                # colnames=config.colNames
                colnames=colNames
                # if (trueDSR_an - lVal[1]) / trueDSR_an > 40:
                # print("bias:",(trueDSR-lVal[1])/trueDSR)
                # newL = np.zeros((par.numNests, 2))
                # newL = np.zeros(2)
                # fdir  = fname[0].parent
                # if (trueDSR - lVal[1]) / trueDSR > 0.40:

                    # print("still high bias")
                    
                    # newLVal = rep_loop(par=par, nData=nestData, storm=stormDays, survey=survey, config=config)
                    # print("new psurv val:", newLVal[1])
                    # newL[0] = newLVal[1]
                    # newL[1] = newLVal[2]
                    # newL = newLVal[1:2]
                    # if (trueDSR - newLVal[1]) / trueDSR > 0.40:
                    #     print("high bias again, try new starting vals")
                    #     newLVal2 = rep_loop(par=par, nData=nestData, storm=stormDays, survey=survey, config=config)
                #     np.save(f"{fdir}/nestdata_{parID:02}_{repID:02}_bias.npy", nestData1)
                # else:
                #     print("low bias")
                #     np.save(f"{fdir}/nestdata_{parID:02}_{repID:02}.npy", nestData1)
                # like_val = np.concatenate((lVal, nVal, newL))
                # colnames=config.colNames

                # if parID == 0 and like_val[17] == 0: # only the first line gets the header
                if parID == 0 and like_val[12] == 0: # only the first line gets the header
                    np.savetxt(f, [like_val], delimiter=",", header=colnames)
                    # if debug: print(">> ** saving likelihood values with header **")
                else:
                    np.savetxt(f, [like_val], delimiter=",")
                    # if debug: print(">> ** saving likelihood values **")
                # need to save it in the function where f was opened?
                # likeVal[r] = like_val
                repID = repID + 1
            # like_val = param_loop(par=par, parID=parID, storm=stormDays, 
            #                       survey=survey, config=config)
            # colnames=config.colNames
            # if parID == 0 and repID == 0: # only the first line gets the header
            # if parID == 0 and like_val[0,17] == 0: # only the first line gets the header
            # # if firstLine == True: # only the first line gets the header
            # # if parID == 0 : # only the first line gets the header
            #     np.savetxt(f, [like_val], delimiter=",", header=colnames)
            #     #np.savetxt(f, like_val, delimiter=",", header=colnames)
            #     if debug: print(">> ** saving likelihood values with header **")
            # else:
            #     np.savetxt(f, [like_val], delimiter=",")
            #     if debug: print(">> ** saving likelihood values **")
                  
                  #    nreps=settings.nreps)
                
            parID = parID + 1

    
# rng = config.rng
# debug_nest=config.debugNests
# debugM=config.debugLL
# debugLL=config.debugLL
# args = config.args
# args = sys.argv[1:]
# args = sys.argv()
# options = "htdwo:"
# optVal  =

# except getopt.error as err:
#     print(str(err))
# test1 = False if 

# using argparse gives more functionality, but optparse gives more control:
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", 
#                     help="run in testing mode w/ limited param values & limited num nests? (default:False)", 
#                     type=bool, action='store_true')
# parser.add_argument("-w", "--wsl", 
#                     help="using WSL? - will change filenames to match (default:False)", 
#                     type=bool, action='store_true')
# parser.add_argument("-d", "--debug", 
#                     help="turn on/off the basic debugger - for more print statements, see script (default:False)", 
#                     type=bool, action='store_true')
# parser.add_argument("-f", "--file", 
#                     help="change likelihood output file location",
#                     type=str)
# args=parser.parse_args()

# config.useWSL = args.wsl


# debug = config.debug
# NOTE need to decide if it's worth saving line-by-line
# main(fnUnique=False, testing=True,deb='none', debN="all", debS="none")
# main(fnUnique=False, useWSL=False, testing=True, deb='none', debN="none", debS="none")
# main(fnUnique=False, debugOpt=debugTypes)
# main(fnUnique=fnUnique, debugOpt=debugTypes, testing=config.testing)



# main(fnUnique=config.fnUnique, debugOpt=debugTypes, testing=config.testing)
main(fnUnique=config.fnUnique, parLists=pLists, testing=config.testing)
# main(fnUnique=False, testing=False, deb='none', debN="all")
# if len(args) > 1:
#     # debug = args[1] == "debugTrue"
#     # enumerate() give sboth index and value
#     for i in range(len(args)):
#         debugList[args[i]] = True
# for key, val in debugList.items():
#     globals()[key] = val
#     print(globals()[key], val)
    
# llArgs = randArgs()
# testLL  = like_smd(x=llArgs, obsData=dat, obsFreq=par.obsFreq, 
#                    stateMat=stMat, useSM=par.useSMat, whichRet=1)




              # "[-o --Options-debug] More specific print statements.\n",
              #   "\t\t\tOptions: 'like','nest','mark','flood','obs'.\n",
              #   "\t\t\tplace in single string with comma delim \n\n",
              # "[-w --Win-true] Use Windowdowss? filenames will be changed to match. (Default:False)\n",
