import numpy as np

def indPrint(x:str, ntabs:int=2, nl=False): #+> print strings, indented
  # ntabs = ind/2 ## +> turns it into a float, which doesn't multiply w/str
  tabs = '\t' * ntabs
  print('\n',tabs,x) if nl else print(tabs,x) 

def objPrint(x, ntabs:int=2, nl=False): #+> print objects, indented
  #+> dunno if this one will work; probably prints 'x' instead of name of obj
  tabs = '\t' * ntabs
  print("\n",tabs,f"{x=}") if nl else print(tabs,f"{x=}") 
  # TODO: fix this

def arrPrint(x, ind=8, comp=True, wd=90): #+> print arrays, indented
  """
  print arrays, indented by <ind> spaces.
  converts if not np.ndarray.
  """
  # pprint.pprint(x, indent=ind, compact=comp, width=wd) ## doesn't indent properly
  # pprint.pprint(f"{i for i in x} ", width=wd)
  # print("\t" + str(x).replace("\n", "\n\t"))
  if not isinstance(x, np.ndarray):
    x = np.array(x)
  xlen = len(x)
  x = np.array2string(x, precision=3, separator=" ", prefix="    ")
  print("    ", x, xlen)

#--- PRINTING: -----------------------------------------------------------------
  # def pr_fates_dsr(nData, expo, trueDSR, nestType):
        # if debug:
        #   print(
        #     "> DISCOVERED NESTS - total | analyzed: hatched:", 
        #     discovered, "|", analyzed,
        #     # "excluded from analysis:", excluded,
        #     "failed:", failed, "|", failed2,
        #     # nestData.shape[0] - sum(nestData[:,3])
        #     # "exposure days:", expDays, "|", sum(nestData[:,15])
        #     "true DSR:", trueDSR_disc, "|", trueDSR_an
        #     )
# -----------------------------------------------------------------------------
# --- PRINT FUNCTIONS ---------------------------------------------------------
# -----------------------------------------------------------------------------
def print_mark():
  inp = np.load("out/inp.npy")
  expo = np.load("out/exposure.npy")

  print("\n\t\t |> input values for calc_exp:")
  for n in range(5):
    print(
          f"\t\t\tnest {n}: first found={inp[n,0]}"
          f"\tlast active={inp[n,1]} \tlast checked={inp[n,2]}"
        )
  print( f"\n\t\t|> output from exposure function:\n")
  for n in range(5):
    print(
          f"\t\t\tnest {n}: alive days={expo[n,0]}"
          f"\tfinal_int={expo[n,1]} \texposure={expo[n,2]}"
        )
  # print(">>>>> Program MARK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  # print("> number of nests:", len(ndata), end=" ")
  # print("| s=", s, "| nocc=", nocc)
  # # print("----------------------------")
  # print(">> all nest cell probabilities:\n", allp)
  # # print("> number of nests:", len(ndata), "discovered nests:", len(disc))
  # # print("inp (ID, i, j, k, fate:)\n",inp)
  # # print("l=", l, "| s=", s, "| nocc=", nocc)
  # # print(">> all degrees of freedom:\n", alldof)
  # # print("log of all nest cell probabilities:", lnp)
  # print(
  #     ">> sum log nest cell probs to get negative log likelihood of the data:", nll)


def print_observer(svysTilDiscovery, discovered):
  print("surveys til discovery; discovered T/F:", svysTilDiscovery, discovered)
  # if cn.debugObs: print("nestID, init, end, tfate, i, j, k, afate, nnobs, intFin:\n", 
  #             np.concatenate((nData,fate[:,None],out), axis=1))
  # if cn.debugObs: print("total observed days:\n", out[:,2]-out[:,0], 
  #             "& total days nest was active:\n", nData[:,2] - nData[:,1])
  
# -----------------------------------------------------------------------------
# def print_prop(assignedFate, trueFate, discovered):  
def print_prop(nData):  
  assignedFate, trueFate = nData[:,7], nData[:,3]
  discovered = nData[:,4] != 0
  aFates = [np.sum((assignedFate == x)[discovered==True]) for x in range(4)]
  # this proportion needs to be out of nests discovered AND assigned
  aFatesProp = [np.sum((assignedFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
  tFates = [np.sum((trueFate == x)[discovered==True]) for x in range(4)]
  tFatesProp = [np.sum((trueFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
  print(
      ">> assigned fate (hatched, depredated, flooded, unknown):", 
      # ">> assigned fate proportions (hatched, depredated, flooded, unknown):", 
      aFatesProp[0:3], 
      (np.sum(discovered==True) - np.sum(aFates)) / (np.sum(discovered==True)),
      # (np.sum(aFates==7) )/ (np.sum(discovered==True)),
      "\n\n>> proportions of known (assigned) fates (H, D, F):",
      aFates[0:3]/np.sum(aFates),
      "\n>> vs. actual proportions for discovered only (H, D, F):",
      tFatesProp[0:3],
      # np.sum(discovered==True)- np.sum(tFates)
      )
  # nestData[:,10] = assignedFate
  # expDays = survival.sum()
  # fail = ~trueHatch
  # expF = survival[fail==True].sum()
  # if debug: print(">> exposure days:", expDays) # why make this a separate column?
                        #   could just sum survival column?
  # exp = np.zeros(numNests)
  # exp.fill(expDays)
  # exposure varies based on the nest fate (failed nests are assumed to have 
  # survived half, or 60% for Johnson, of the final interval)
  # nestData[:,14] = exp
 
  # numDisc = discovered.sum()
  # numDiscH = trueHatch[discovered==True].sum()
  # survDisc = survival[discovered==True].sum()
  # if debug: print(">> number discovered:", numDisc, ", number discovered that hatched:", numDiscH, ", and exposure days (discovered nests):", survDisc)
  # trueDSR_disc = 1 - ((numDisc - numDiscH) / survDisc) # num failed / total exposure days
  # if debug: print(">> true DSR of discovered nests only:", trueDSR_disc) 
  # # NOTE issue may be that not enough flooded nests are discovered, not that too many hatched nests are
  
  # print(">> nest discovered?", discovered)
  # print(">> discovered & hatched:", discovered[trueHatch==True])
  # print(">> calculate exposure days for discovered nests by summing this list:", survival[discovered==True])
  #trueDSR_disc = 1 - ((discovered.sum() - hatched.sum()) / survival.sum()) 

  # nests = np.stack((discovered, trueFate, firstFound, lastActive, lastChecked, totalReal, numStorms))
  # nests = np.transpose(nests)
  # print(">> discovered?, fate, i, j, k, num surveys, num storm intervals:\n", nests)
  # print(
  #   ">> discovered?, fate, i, j, k, num surveys, num storm intervals:\n", 
  #   nests[:5,:], 
  #   "\n ... \n",
  #   nests[-5:,:]
  #   )

  #@#print(">> nest data:\n----id--ini-end-hch-fld-std-dsc-i--j--k-fate-nobs-sfin-nstm\n", nestData)
# -----------------------------------------------------------------------------
def print_nd(nestData, nDisc, pSurv, hatchTime):
  # if debug_nest: print("nestData, discovered only:\n", nestData)
  print("nestData, discovered only:\n", nestData)
  print(">> proportion of nests assigned hatch fate:", 
          np.sum(nestData[:,3]==0)/(nDisc),
          "vs period survival:", 
          pSurv**hatchTime)

  print(
    ">> nests w/ only 1 obs while active:",
    np.where(nestData[:,6] == nestData[:,7]), # where i==j
    "& nests w/ unknown fate:",
    np.where(nestData[:,9] == 7)
    ) 
        # if debug:
        #   print(
        #     "\n>> assigned DSR:",
        #     pSurv,
        #     "true DSR of all nests:", 
        #     trueDSR, 
        #     "discovered nests:",
        #     trueDSR_disc,
        #     "and nests used in analysis:", 
        #     trueDSR_an
        #     )
# -----------------------------------------------------------------------------
def print_mayf(expo):
  print("output from exposure function:", expo)
  
  for n in range(len(expo)):
    print(
      "days nest was alive:", expo[n,0],
      "& final int:", expo[n,1], 
      "& exposure:", expo[n,2]
      )
    
    
  # if debug:
  #   print(
  #     # f"> {nestType} nests - hatched:", hatched.sum(),
  #     f"> {nestType} nests - hatched:", hatched,
  #     "; failed:", nNests - hatched, 
  #     # "; exposure days:", expDays[:,2].sum(),
  #     "; exposure days:", expDays.sum(),
  #     "; & Mayfield-40 DSR:", 1-dmr
  #     )
# -----------------------------------------------------------------------------
def printLL(numNests, logLik, logLikFin, numInt, logL):
  """ print entire likelihood equation """

  for x in range(numNests): ## range excludes end point
    # print(">> likelihood equation: (",logLik[x],"*",numIntNorm[x],")+(",logLikStm[x],"*",numIntStm[x],")+(",logLikFin[x],"**(1 -",stormDuringFin[x],")+(", logLikFinStm[x],"**",stormDuringFin[x])
    print(
        f"\t\t>> likelihood equations [(numInt * logLik) + logLikFin]:"
        f"\n\t\t\tnest {x}: " 
    #  f"{numInt[x]:.0f} * {logLik[x]:.5f} + "
       f"\t\t\t({numInt[x]:.0f} * {logLik[x]:.5f}) + {logLikFin[x]:.5f} ="
    #  f"{logLikFinStm[x]:.5f} * (1-{stormDuringFin[x]:.0f}) + " 
    #  f"{logLikFinStm[x]:.2f} * {stormDuringFin[x]:.2f} = "
    #  f"{logLikelihood[x]:.2f}")
       f"\t\t\t{logL[x]:.2f}"
       )
  print( f"\n\t\t\t\t|> total log likelihood: {logL.sum():>30.3f}"
      )
# -----------------------------------------------------------------------------
# def logL(numNests, normalInt, finalInt, numInt, ha, config=config):
# def logL(numNests, intervals, numInt, ha, fl, config=config):
# @profile
# -----------------------------------------------------------------------------
#   PROGRAM MARK 
# -----------------------------------------------------------------------------
# The model used in Program MARK is based on Dinsmore (2002) -  
#    allows for variance in DSR & use of covariates

# These functions are based on info in 'Program MARK: A Gentle Introduction' 

# It also has a wrapper function that transforms the initial optimizer values
# using the logistic function.
# This way, the optimizer can work over the range of -infinity:infinity, but
# the values fed to the function are between 0 and 1 (probabilities)

# Lastly, it has a function to generate the probabilities before running the optimizer 
# on the MARK function, so I can take the for loop out of the function that is optimized.

# -----------------------------------------------------------------------------
# def prog_mark(s, ndata, probs, nocc, con=config):
# @profile

def print_all(sums, nestData, par, debug=0):
  """
    ARGS:
      nestData = nest data w/excluded removed
  """
  ha, fl,dsc, unk, mc, ex, dsr_c, dsr_a, dsr_t = sums #+> unpack vals
  # if debug>=2:
  print(
    f"\n\t\t |== (true)flooded: {fl.sum()} |"
    f" (true)hatched: {ha.sum()} |"
    f" (obs)flooded: {sum(nestData[:,7]==2)} |"
    f" (obs)hatched: {sum(nestData[:,7]==0)} |"
    f" unk: {unk.sum()} ==|"
    )
  # if debug>=1:
  print(
    f"\n\t\t |== discovered: {dsc.sum()} |"
    f" excluded: {ex.sum()} |"
    f" misclassified: {mc.sum()} |"
    f" true DSR - analyzed nests: {dsr_a} ==| "
    )
  # if debug>=2: 
  print(
    f"\n\t\t |== true DSR: {dsr_t} |"
    f" assigned DSR: {par.probSurv} |"
    f" calc DSR: {dsr_c} |"
    f" DSR bias: {(dsr_t-dsr_c)/dsr_t} ==| "
    )

# def print_bias():
        # if (trueDSR_an - lVal[1]) / trueDSR_an > 40:
        # if debug>=2: print("DSR bias:",(trueDSR-lVal[1])/trueDSR)
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
          #   print("high bias again, try new starting vals")
          #   newLVal2 = rep_loop(par=par, nData=nestData, storm=stormDays, survey=survey, config=config)
        #   np.save(f"{fdir}/nestdata_{parID:02}_{repID:02}_bias.npy", nestData1)
        # else:
        #   print("low bias")
        #   np.save(f"{fdir}/nestdata_{parID:02}_{repID:02}.npy", nestData1)

def print_nest_info(nestData, discover, exclude):
  print("nests not discovered:", nestData[:,0][~discover])
  print("nests to exclude from analysis:", nestData[:,0][exclude])
  print("nest data, analysis nests only:\n",
          "ID, init, survival, true fate, i, j, k, assigned fate, num normal obs, intFinal, num storms:\n",
           nestData[~discover and exclude])

#--- OLD: ----------------------------------------------------------------------

# def main(fnUnique, useWSL=False, testing=True, deb="none", debN="none", debS="none", config=config, pStatic=staticPar):
# def main(fnUnique, testing=config.testing, deb="none", debN="none", debS="none", config=config, pStatic=staticPar):
# def main(fnUnique, debugOpt, testing=config.testing, config=config, pStatic=staticPar): # calls config too early
# @profile
# def main(fnUnique, debugOpt, testing, pList, config=config, pStatic=staticPar):
#def like_old(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData, stormDays, surveyDays, obs_int):

# def like_old(argL, obsFreq, nestData, surveyDays, stormDays, numNests):
#   """
#
#   This function computes the overall likelyhood of the data given the model parameter estimates.
#
#   The model parameters are expected to be received in the following order:
#   - a_s   = probability of survival during non-storm days
#   - a_mp  = conditional probability of predation given failure during non-storm days
#   - a_mf  = conditional probability of flooding given failure during non-storm days
#   - a_ss  = probability of survival during storm days
#   - a_mfs = conditional probability of predation given failure during storm days
#   - a_mps = conxditional probability of predation given failure during storm days
#   - sM  = for program MARK?
#
#   """
#
#   nCol = 18
#   a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
#   obs_int = obsFreq
#   likeData = np.zeros(shape=(numNests, nCol), dtype=np.longdouble) 
#
#   stillAlive  = np.array([1, 0, 0])
#   mortFlood   = np.array([0, 1, 0])
#   mortPred  = np.array([0, 0, 1])
#
#   # > starting matrix, from etterson 2007:
#   startMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
#   # > use this matrix for storm weeks:
#   stormMatrix = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]]) 
#     # how is this matrix actually being incorporated during the analysis?
#
#   logLike = Decimal(0.0)      # initialize the overall likelihood counter
#   #logLike = float(logLike)
#   rowNum = 0
#   for row in nestData:
#   # columns for the likelihood comparison datarframe:
#   #     num obs, log lik of nest, log lik period 1, period 2, etc.  
#
#   # FOR EACH NEST -------------------------------------------------------------------------------------
#
#     nest  = row     # choose one nest (multiple output components from mk_obs())
#     # print('obs_int check: ', obs_int)
#
#     # disc  = nest[7].astype(int)  # first found
#     disc  = nest[0].astype(int)  # first found
#     # endObs  = nest[9].astype(int)  # last observed
#     endObs  = nest[2].astype(int)  # last observed
#     # hatched = nest[3]
#     # flooded = nest[4]
#     hatched = nest[3] == 0
#     flooded = nest[3] == 2
#
#     if flooded == True & hatched == False:
#       fate = 2
#     elif flooded == False & hatched == False:
#       fate = 3
#     else:
#       fate = 1
#     # fate  = nest[7].astype(int)  # assigned fate
#
#     if np.isnan(disc):
#       ###print("this nest was not discovered but made it through")
#       continue
#
#     num   = len(np.arange(disc, endObs, obs_int)) + 1 # number o observations
#     # print('#############################################################')
#     # print('nest =', nest[0], 'row number =', rowNum, 'number of obs =', num)
#
#     likeData[rowNum, 0:3]  = np.array([nest[0], fate, num] )
#
#     # print("num=", num)
#     obsDays = in1d_sorted(
#       # (np.linspace(disc, endObs, num=num)), surveyDays)
#       np.linspace(disc, endObs, num=num),
#       surveyDays)
#     # print("obs days for nest:", obsDays)
#     obsPairs = np.fromiter(
#       itertools.pairwise(obsDays), 
#       dtype=np.dtype((int,2))
#       ) # do elements of numpy arrays have to be floats?
#     # print("date pairs in observation period:", obsPairs) 
#
#     # > make a list of intervals between each pair of observations 
#     #   (necessary for likelihood function)
#     intList = obsPairs[:,1] - obsPairs[:,0]
#     # print("interval list:", intList)
#
#     # > start off with all intervals = alive
#     obs   = [stillAlive for _ in range(len(obsPairs)+1)] 
#
#     # > change the last obs if nest failed:
#     if fate == 2:
#       obs[-1] = mortFlood
#     elif fate == 3:
#       obs[-1] = mortPred
#
#     # print("fate, obs = ", fate, " , ", obs) # check that last entry in obs corresponds to fate
#
#     # if hatch, leave as is?
#
#     # place this likelihood counter inside the for loop so it resets 
#     # with each nest:
#     logLikelihood = Decimal(0.0)   
#     #logLikelihood = float(logLikelihood)
#
#     # likeData[0, rowNum] = nest[0]
#     obsNum = 0
#     for i in range(len(obs)-1):
#     # FOR EACH OBSERVATION OF THIS NEST ---------------------------------------------------------------------
#
#       # print("observation number:", obsNum)
#
#       intElt  = (intList[i-1]).astype(int)  # access the (i-1)th element of intList,
#                   # which is the interval from the (i-1)th
#                   # to the ith observation
#
#       #stateF  = obs[i]
#       stateF  = obs[i+1] 
#       stateI  = obs[i]
#       # print("stateF:",stateF)
#       TstateI = np.transpose(stateI)
#       # print("TstateI:", TstateI)
#
#       # if any(d in storm_days for d in range(i-1, i)):
#       if any(d in stormDays for d in range(i-1, i)):
#         # if any of the days in the current observation interval (range) is also in storm days, use storm matrix
#         # print("using storm matrix")
#         lDay = np.dot(stateF, np.linalg.matrix_power(stormMatrix, intElt))
#         # this is the dot product of the current state of the nest and the storm matrix ^ interval length
#      # look into using @ instead of nest dot calls 
#       else:
#         # print("using normal matrix")
#         lDay = np.dot(stateF, np.linalg.matrix_power(startMatrix, intElt))
#
#       lPer = np.dot(lDay, TstateI)
#       # print("likelihood for this interval:", lPer)
#
#       logL = Decimal(- np.log(lPer))
#       # print("negative log likelihood of this interval:", logL)
#
#       #logL = float(logL)
#
#       logLikelihood = logLikelihood + logL # add in the likelihood for this one observation
#       # print("log likelihood of nest observation history:", logLikelihood)
#       colNum = obsNum + 4 
#       likeData[rowNum, colNum] = logL
#       obsNum = obsNum + 1
#
#     likeData[rowNum,3] = logLikelihood
#     logLike = logLike + logLikelihood    # add in the likelihood for the observation history of this nest
#     rowNum  = rowNum + 1
#     # print("increment row number:", rowNum)
#     # print("overall log likelihood so far:", logLike)
#
#
#   # print(
#   #   "nest num, fate, num obs, lik(obs hist), lik(each obs int) ... \n",
#   #   likeData[:10,:] # print first ten rows
#   # )
#   return(logLike) 
