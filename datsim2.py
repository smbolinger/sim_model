
import sys
import numpy as np 
import scipy.stats as stats
import pandas as pd
import csv
import json
import pickle
from pathlib import Path
import os

from rsettings import config, rng, odir, atype, now_short
from dsrCalc import calc_dsr
from datsim import rep_loop
from print_func import print_nestdata, dfPrint, print_all, printLL
from log_exposure import calc_daily_expo, make_daily_logex_df
from getClass import Params,Config
from helpers import mk_fnames,print

#+> need to pass par, nestData1, stormDays, surveyDays...
# par = sys.argv[1]
# stormDays = sys.argv[2]
# survey = sys.argv[3]
# nestData1 = sys.argv[4]
# parf = 
# par = np.genfromtxt(fname="analysis/par.txt")
# stormDays = np.genfromtxt(fname="analysis/storm.npy")
# survey = np.genfromtxt(fname="analysis/survey.npy")
# nestData1 = np.genfromtxt(fname="analysis/nestData1.npy")
now_str=now_short
# outdir = sys.argv[1]
# outdir = os.environ.get('r_outdir') ##+> doesn't work with Path object
## actually, doesn't work at all
outdir = Path(f"{odir}/{config.rngSeed}{atype}") 
# if config.debug>=1: print(f"\t\t\tfrom R - {outdir=}")
# if config.debug>=4: print(f"\t\t\tdatsim2.py - {outdir=}", end=" ")
# nestData1 = pd.read_csv("analysis/nestData1.csv")
nestData1 = pd.read_csv(f"{outdir}/nestData1.csv")
nestData1 = nestData1.to_numpy()
with open(f"{outdir}/data.pkl","rb") as f:
  allDat = pickle.load(f)
# with open("analysis/data.json","r") as f:
#   allDat = json.load(f)
# outdir = allDat["outdir"]
parID = allDat["parID"]
repID = allDat["repID"]
parDic = allDat["par"]
par = Params(**parDic)
confDic = allDat["config"]
config = Config(**confDic)
stormDays = allDat["stormDays"]
survey = allDat["survey"]
# nestData1 = allDat["nestData1"]
nm = ["par", "stormDays", "survey", "config"]

# if par.debug >=2:
#NOTE par is a dict now:
# if par["debug"] >=2:
if config.debug>=4:
  # print("imported .json:")
  print(f"\t\t\t\told script - nestData1 <{len(nestData1)=}> :")
  dfPrint(nestData1)
  print("\t\t\t\timported .pkl:")
  # for d in range(len(allDat)):
  for d in nm:
    print(f"\t\t\t\t{d=}")
    print(f"\t\t\t\t{allDat[d]=}")
    print(f"\t\t\t\t{type(allDat[d])=}")
    # print()
# if par.debug >=2: print(allDat)

## +> can pass lists/arrays w/argparse, but not sys.argv
#~----------------------------------------------------------------------
## +> print all nest data
# if config.debug>=3:
#   print("\t\t\t|> ALL NEST DATA")
#   nm = ["ID","init","end","fate"," i "," j "," k ","afate","nobs","fint"]
#   # nm = ["ID","init","end","fate"," i "," j "," k ","afate","nobs","fint","nstm"]
#   # print_nestdata(nestData1, names=nm,nprint=50) 
#   # print("\nall nests:")
#   print_nestdata(nestData1, names=nm, abbrv=False) 

# if config.debug>=2: 
#   print("\n\t\t[*] [*] [*] [*] [*] calculating DSR [*] [*] [*] [*] [*] [*] [*] [*] ")

#~----------------------------------------------------------------------
##+> ends up weird bc undiscovered nests have negative exposure:
# mayfDSR_all  = calc_dsr(nData=nestData1,
#                     nestType="all",
#                     calcType="mayfield",
#                     conf=config,
#                     incTime=par.hatchTime,
#                     psurv=par.probSurv,
#                     debug=config.debugSummary)

##+> decreases slightly in accuracy w/ lower vals of DSR:
# appDSR  = calc_dsr(nData=nestData1,
#                     nestType="all",
#                     calcType="apparent",
#                     conf=config,
#                     incTime=par.hatchTime,
#                     psurv=par.probSurv,
#                     debug=config.debugSummary)
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

# mayfDSR_disc = calc_dsr(nData=nestData,
#                         nestType="disc",
#                         calcType="mayfield",
#                         conf=config,
#                         incTime=par.hatchTime,
#                         psurv=par.probSurv,
#                         debug=config.debugSummary)

##+> for discovered nests, these two estimates are very close
# appDSR_disc = calc_dsr(nData=nestData,
#                         nestType="disc",
#                         calcType="apparent",
#                         conf=config,
#                         incTime=par.hatchTime,
#                         psurv=par.probSurv,
#                         debug=config.debugSummary)

nestData  = nestData[~(exclude),:]  # +> remove excluded nests 
print("\n\t\tnest data length after excluding:",nestData.shape[0])
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
mayfDSR_an  = float(mayfDSR_an) 
# appDSR_an   = calc_dsr(nData=nestData,
#                         nestType="analysis",
#                         calcType="apparent",
#                         conf=config,
#                         incTime=par.hatchTime,
#                         psurv=par.probSurv,
#                         debug=config.debugSummary) 
# lVal_py = rep_loop(par, rng, nestData, stormDays, survey, config, to_r=True)
lVal_py = rep_loop(par, rng, nestData, stormDays, survey, config)
# llDSR = lVal[0]
# llDSR,llPSR,llDFR = lVal
lVal_py = lVal_py.astype('float')
lDSR,lPSR,lDFR = lVal_py
# print(f"{type(lDSR)=}")
# #   ##+> print summary:
# if config.debug>=3: ## +> matches the level for saving the info to print
#   sum_list = [
#       hatched,
#       flooded,
#       discover,
#       unknown,
#       misclass,
#       exclude,
#       lDSR,
#       lPSR,
#       # lVal[0],
#       # lVal[1],
#       mayfDSR_an,
#       appDSR_an,
#       appDSR,
#       appDSR_disc,
#       ]
#   print_all(sum_list, nestData, par)
#----------------------------------------------------------------------
# if config.debugLL>=2: ## +> matches the level for saving the info to print
# #   ##+> print LL equations for each nest:
# #   ##+> enable the saving inside hte logLike function definition
# #
#   # llArg = np.load('out/arg_PrintLL.npy') 
#   # printLL(len(llArg), *llArg.T) # +> tranpose so it is unpacked colwise
#   printLL() # +> tranpose so it is unpacked colwise

# if False:
#   # if config.debug>=4: print(f"{parID=} | {repID=}")
#   nVal = np.array([appDSR,        #4
#                    appDSR_disc,
#                    appDSR_an,
#                    mayfDSR_disc,
#                    mayfDSR_an,    #5
#                    sum(discover),
#                    sum(exclude),
#                    sum(unknown),
#                    sum(misclass),
#                    flooded,
#                    hatched,
#                    # nEx,
#                    repID,
#                    parID,
#                    ])  
#   like_val = np.concatenate((lVal, nVal))
#   fname = mk_fnames(now_str,fdir=odir,suf=f"{atype}", con=config)
#   likeFile = fname[0]
#   colNames = fname[1]
#   colnames=colNames # colnames=config.colNames
#
  # if parID == 0 and like_val[12] == 0: #+> only 1st line gets the header
  ## NOTE this is saving row-by-row
  # if parID == 0 and repID == 0: #+> only 1st line gets the header
  #   np.savetxt(f, [like_val], delimiter=",", header=colnames)
  #   # if debug: print(">> ** saving likelihood values with header **")
  # else:
  #   np.savetxt(f, [like_val], delimiter=",")
