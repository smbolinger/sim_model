import numpy as np
import pandas as pd
import pickle
import functools
# from helpers import print
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None) ##+> print all columns
pd.set_option('display.width', 999)
np.set_printoptions(linewidth=150)
print = functools.partial(print, flush=True)

def dfPrint(x,abbr=True,nprint=10,ind=6,wd=90,concat="no",names:list=[]):
  """
    Print pandas dataframe.
    ----
    ARGS:
      x = df-like object (2D array, df, etc)
          or list of df-like objects
      abbr = print only first x rows (default=10)
      nprint = (optional) number of rows to print
      ind = number of spaces to indent
      wd  = width of output
      names = df names; can be list if cwise concat
          default "V1,V2..." if none 
      concat = concatenate the dataframes?
          options: cwise, rwise
    ----
    NOTES:
      Converts object to pandas DataFrame if needed
    ------
  """
  spaces = ind * ' '
  if not isinstance(x, list):
    names = [f"V{i}" for i in range(x.shape[1])] if len(names)<1 else names
    dfList = [pd.DataFrame(x, columns=names)]
  else:
    # if isinstance(names, list):
    if isinstance(names, list):
      dfList = [pd.DataFrame(v, columns=names[i]) for i,v in enumerate(x)]
    else:
      names = [f"V{i}" for i in range(x[0].shape[1])] if len(names)<1 else names
      dfList = [pd.DataFrame(i, columns=names) for i in x]
  # nrow = 5 if abbr else sum([len(df) for df in dfList]) ##+>num rows to print
  nrow = nprint if abbr else sum([len(df) for df in dfList]) ##+>num rows to print
  if concat=="no":
    for df in dfList:
      # dfp = pd.DataFrame(data=df, columns=names)
      df = df.iloc[:nrow]
      df_print = df.to_string()
      print(spaces + df_print.replace('\n', '\n'+spaces))
      print(" ") # add space between subsequent dfs
  elif concat=="cwise":
    print(pd.concat(dfList, axis=1).head(nrow))
  elif concat=="rwise":
    print(pd.concat(dfList, axis=0).head(nrow))

def arrPrint(x, ind=6,abbr=True,abval=12,comp=True, wd=90, sep=" "): #+> print arrays, indented
  """
    print 1d arrays, indented by <ind> spaces.
      - allows wrapping over multiple lines while still indenting
      - abbr=True prints only first 10 & last 10 items

    converts if not np.ndarray.
  """
  # TODO: add option to prepend a string to array output
  # TODO: add string formatting/padding
  # pprint.pprint(x, indent=ind, compact=comp, width=wd) ## doesn't indent properly
  # pprint.pprint(f"{i for i in x} ", width=wd)
  # print("\t" + str(x).replace("\n", "\n\t"))
  # wd=80 if abbr else 130
  np.set_printoptions(precision=3, linewidth=wd )
  tabs = ind*' '
  if not isinstance(x, np.ndarray):
    x = np.array(x)
  xlen = len(x)
  # x = np.array2string(x, precision=3, separator=" ", prefix="    ")
  if abbr:
    # print(tabs, x[:5], "....",x[-5:], xlen) # +> x is a string
    x1 = np.array2string(x[:abval], precision=3, separator=sep, prefix=tabs)
    x2 = np.array2string(x[-abval:], precision=3, separator=sep, prefix=tabs)
    print(tabs, x1, "....", x2, xlen)
  else:
    x = np.array2string(x, precision=3, separator=sep, prefix=tabs)
    print(tabs, x, xlen) #+> print 'tabs' again here to indent the first line

def indPrint(x:str, ntabs:int=2, nl=False): #+> print strings, indented
  # ntabs = ind/2 ## +> turns it into a float, which doesn't multiply w/str
  tabs = '\t' * ntabs
  print('\n',tabs,x) if nl else print(tabs,x) 

def objPrint(x, ntabs:int=2, nl=False): #+> print objects, indented
  #+> dunno if this one will work; probably prints 'x' instead of name of obj
  tabs = '\t' * ntabs
  print("\n",tabs,f"{x=}") if nl else print(tabs,f"{x=}") 
  # TODO: fix this

def print_prop(nData, obsColNum):  
    # nData = nData[nData[:,4]!=0]
  obsColNum = int(obsColNum)
  print(f"\t\t|> NEST FATE PROPORTIONS (discovered = where column {obsColNum} > 0):")
  nstrList=["ALL", "DISCOVERED", "ANALYZED"]
  nDataDisc = nData[nData[:,obsColNum]>0]
  nDataAn = nDataDisc[nDataDisc[:,7]!=7]
  # print(f"{nDataAn=}")
  nDataAn = nDataAn[nDataAn[:,2]>nDataAn[:,4]]
  # print(f"{nDataAn=}")
  ndataList = [nData, nDataDisc, nDataAn]
  for n in range(0,len(nstrList)):
    nstr = nstrList[n]
    nestData = ndataList[n]
    mk_print_prop(nestData, nstr, obsColNum)

  # nData = nData
def mk_print_prop(nData, nstr, obsColNum):
  assignedFate, trueFate = nData[:,7], nData[:,3]
  # print(f"{nData.shape[0]=}")
  aFates = [np.sum((assignedFate == x)) for x in [0,1,2,7]]
  # this proportion needs to be out of nests discovered AND assigned
  aFatesProp = [np.sum((assignedFate == x))/(nData.shape[0]) for x in [0,1,2,7]]
  aFatesProp = [f"{n:.3f}" for n in aFatesProp]

  tFates = [np.sum((trueFate == x)) for x in range(3)]
  tFatesProp = [np.sum((trueFate == x))/(nData.shape[0]) for x in range(3)]
  tFatesProp = [f"{n:.3f}" for n in tFatesProp]

  uFates = [np.sum((assignedFate==7) & (trueFate==x)) for x in range(3)]
  # print(f"{uFates=}")
  
  uFatesProp = [np.sum((assignedFate==7) & (trueFate==x))/(nData.shape[0]) for x in range(3)]
  uFatesProp = [f"{n:.3f}" for n in uFatesProp]
  # uFatesProp = uFates / nData.shape[0]
  print(
      f"\t\t\t==> {nstr} NESTS - count (proportion):"
      # f"\t\t>> assigned [H D Fl U]: {" ".join(aFates)} ({" ".join(aFatesProp)})", 
      # f"\t\t>> true [H D Fl]: {" ".join(tFates)} ({" ".join(tFatesProp)})", 
      # f"\t\t>> marked unknown [H D Fl]: {" ".join(tFates)} ({" ".join(tFatesProp)})", 
      f"\t\t>> assigned [H D Fl U]: {aFates} ({" ".join(aFatesProp)})", 
      f"\t\t>> true [H D Fl]: {tFates} ({" ".join(tFatesProp)})", 
      f"\t\t>> marked unknown [H D Fl]: {tFates} ({" ".join(tFatesProp)})", 
      )

def print_all(sums, nestData, par, debug=0):
  """
    ARGS:
      nestData = nest data w/excluded removed
    -------
    dsr_c = mcmc dsr; dsr_a = apparent dsr; dsr_d = app dsr (disc)
    dsr_t = app DSR - true; dsr_m = MARK dsr; 

    unpacked to ha,fl,dsc,unk,mc,ex,dsr_c,dsr_a,dsr_r,dsr_d
  """
  # ha,fl,dsc,unk,mc,ex,dsr_c,dsr_m,dsr_mf,dsr_a,dsr_t,dsr_d=sums #+>unpack vals
  ha,fl,dsc,unk,mc,ex,dsr_c,psr_c,dsr_mf,dsr_a,dsr_t,dsr_d=sums #+>unpack vals
  # if debug>=2:
  # obsHat = sum(nestData[:])
  print(
    f"\n\t|== (true)num"
    f" -flood: {fl.sum()} |"
    f" -hatch: {ha.sum()} |"
    f" (obs)num-flood: {sum(nestData[:,7]==2)} |"
    f" -hatch: {sum(nestData[:,7]==0)} |"
    f" unk: {unk.sum()}"
    f" disc: {dsc.sum()} |"
    f" excl: {ex.sum()} |"
    f" misclass: {mc.sum()} |"
    f" ==|"
    )
  # if debug>=1:
  print(
    f"\n\t|=="
    f" (true)prop-hatch: {ha/par.numNests:.3f} |"
    f" -flood: {ha/par.numNests:.3f} |"
    # f" (obs)prop-hatch: {ha/par.numNests:.3f} |"
    # f" -flood: {ha/par.numNests:.3f} |"
    f" expected PSR: {par.probSurv**par.hatchTime} "
    f" apparent DSR - all: {dsr_t:.3f} |"
    f" - analyzed: {dsr_a:.3f} |"
    f" - discovered: {dsr_d:.3f} ==| "
    )
  # if debug>=2: 
  print(
    f"\n\t|=="
    # f" apparent DSR: {dsr_t:.3f} |"
    f" assigned DSR: {par.probSurv} |"
    f" MCMC DSR: {dsr_c:.3f} |"
    f" Mayfield DSR: {dsr_mf:.3f} |"
    # f" MARK DSR: {dsr_m:.3f} |"
    # f" DSR diff"
    f" MCMC DSR bias: {(dsr_t-dsr_c)/dsr_t:.3f} ==| \n"
    )

def printLL():
  """ print entire likelihood equation """

  stillAlive = np.array([1,0,0]) 
  mortFlood  = np.array([0,1,0])
  mortPred   = np.array([0,0,1])
  TstateI = np.transpose(stillAlive)  # this is just one, not a vector? yes
  trMat = np.load('out/trMat.npy')
  with open("out/interval.pkl", "rb") as f:
    intArg = pickle.load(f)
  pwrN, pwrS, normInt, finPred, finStm = intArg
  print(f"\n\n\t\t>> Maximum Likelihood:")

  print(
      f"\n\t\t\ttransition matrix: {trMat=} "
      f"\n\t\t\tnormal interval: {stillAlive=} @ {pwrN=} @ {TstateI=} "
      f"\n\t\t\tfailure-predation: {mortPred=} @ {pwrS=} @ {TstateI=} "
      f"\n\t\t\tfailure-flood: {mortFlood=} @ {pwrS=} @ {TstateI=} "
      f"\n\t\t\t{normInt=} {finPred=} {finStm=}"
      )


  llArg = np.load('out/arg_PrintLL.npy') 
  numNests = len(llArg)
  # for x in range(numNests):

  logLik,logLikFin,numInt,logL,ids,fate = llArg.T
  print(f"\n\t\t\t>> likelihood equations [(numInt * logLik) + logLikFin]:")

  for x in range(numNests): ## range excludes end point
    # print(">> likelihood equation: (",logLik[x],"*",numIntNorm[x],")+(",logLikStm[x],"*",numIntStm[x],")+(",logLikFin[x],"**(1 -",stormDuringFin[x],")+(", logLikFinStm[x],"**",stormDuringFin[x])
    print(
        f"\t\t\t\t\t\tnest {ids[x]:.0f} [fate:{fate[x]:.0f}]: " 
    #  f"{numInt[x]:.0f} * {logLik[x]:.5f} + "
       f"({numInt[x]:.0f} * {logLik[x]:.5f}) + {logLikFin[x]:.5f} ="
    #  f"{logLikFinStm[x]:.5f} * (1-{stormDuringFin[x]:.0f}) + " 
    #  f"{logLikFinStm[x]:.2f} * {stormDuringFin[x]:.2f} = "
    #  f"{logLikelihood[x]:.2f}")
       f" {logL[x]:.2f}"
       )
  print( f"\n\t\t\t\t\t\t\t|> total log likelihood: {logL.sum():>30.3f}"
      )

def print_nestdata(nData, names:list=[], nprint=10, abbrv=False):
  """
    ---------
    PURPOSE: print values from nest data matrix
    ---------

    ARGS:
      1. data to print
      2. names [as list; default=[]]
      3. number of lines to print [default=10]
      
     [0]:nest ID...........[1]:initiation.......[2]:end date..........
     [3]:true fate ........[4]:first found......[5]:last active.......
     [6]:last checked......[7]:assigned fate....[8]:num obs active.... 
     [9]:len(final int)....[10]:num obs total....

    -------
    
  """
  # print("\t\t\tNESTS CREATED - DATA:")
  ## don't understand why this is giving an "already assigned" eror
  ## I didn't call dfPrint yet?
  # nm = ["ID","init","end","fate"," i "," j "," k ","afate","nobs","fint","stm","obs_tot"]
  nm = ["ID","init","end","fate","i","j","k","afate","nobs","fint","obs_tot"]
  dfPrint(nData,abbr=abbrv, names=nm,nprint=nprint,ind=8)
  # print(f"nData[0,1:9]=")

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

def fate_prop(assignedFate, trueFate):
  """
    ARGS: assigned fates & true fates for a given group of nests
      (all nests, discovered nests, analyzed nests, etc...)
  """
  fates=[0,1,2,7]
  numNests = len(assignedFate)
  print(f"{numNests=}")
  aFates = [np.sum((assignedFate == x)) for x in fates]
  # this proportion needs to be out of nests discovered AND assigned
  aFatesProp = [np.sum((assignedFate == x))/(numNests) for x in fates]
  tFates = [np.sum((trueFate == x)) for x in fates]
  tFatesProp = [np.sum((trueFate == x))/(numNests) for x in fates]

# def fate_prop(assignedFate, trueFate, discovered):
#   aFates = [np.sum((assignedFate == x)[discovered==True]) for x in range(4)]
#   # this proportion needs to be out of nests discovered AND assigned
#   aFatesProp = [np.sum((assignedFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]
#   tFates = [np.sum((trueFate == x)[discovered==True]) for x in range(4)]
#   tFatesProp = [np.sum((trueFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(4)]

# def print_prop(nData, whichNests, obsColNum):  
#   if whichNests == "disc":
#     # nData = nData[nData[:,4]!=0]
#     nData = nData[nData[:,obsColNum]>0]
#     nstr  = "DISCOVERED"
#   elif whichNests == "an":
#     nData = nData[nData[:,4]!=0]
#     nstr  = "ANALYZED"
#   else:
#     nData = nData
#     nstr="ALL"
#   assignedFate, trueFate = nData[:,7], nData[:,3]
#   aFates = [np.sum((assignedFate == x)) for x in [0,1,2,7]]
#   # this proportion needs to be out of nests discovered AND assigned
#   aFatesProp = [np.sum((assignedFate == x))/(nData.shape[0]) for x in [0,1,2,7]]
#
#   tFates = [np.sum((trueFate == x)) for x in range(3)]
#   tFatesProp = [np.sum((trueFate == x))/(nData.shape[0]) for x in range(3)]
#   uFates = [np.sum((assignedFate==7) & (trueFate==x)) for x in range(3)]
#   uFatesProp = uFates / nData.shape[0]
#   print(
#       f"\t\t==> {nstr} NESTS - count (proportion):"
#       f"\t\t>> assigned [H D Fl U]: {aFates} ({aFatesProp})", 
#       f"\t\t>> true [H D Fl]: {tFates} ({tFatesProp})", 
#       f"\t\t>> marked unknown [H D Fl]: {tFates} ({tFatesProp})", 
#       )

def print_prop_all(nData):  
  assignedFate, trueFate = nData[:,7], nData[:,3]
  aFates = [np.sum((assignedFate == x)) for x in [0,1,2,7]]
  # this proportion needs to be out of nests discovered AND assigned
  aFatesProp = [np.sum((assignedFate == x))/(nData.shape[0]) for x in [0,1,2,7]]

  tFates = [np.sum((trueFate == x)) for x in range(3)]
  tFatesProp = [np.sum((trueFate == x))/(nData.shape[0]) for x in range(3)]
  uFates = [np.sum((assignedFate==7 & trueFate==x)) for x in range(3)]
  print(
      "\t\t==> ALL NESTS - count (proportion):"
      "\t\t>> assigned [H D Fl U]: {aFates} ({aFatesProp})", 
      "\t\t>> true [H D Fl]: {tFates} ({tFatesProp})", 
      )

def print_prop_disc(nData):  
  nData = nData[nData[:,4]!=0]
  assignedFate, trueFate = nData[:,7], nData[:,3]
  # discovered = nData[:,4] != 0
  aFates = [np.sum((assignedFate == x)) for x in [0,1,2,7]]
  # this proportion needs to be out of nests discovered AND assigned
  # aFatesProp = [np.sum((assignedFate == x)[discovered==True])/(np.sum(discovered==True)) for x in [0,1,2,7]]
  aFatesProp = [np.sum((assignedFate == x))/(nData.shape[0]) for x in [0,1,2,7]]
  tFates = [np.sum((trueFate == x)) for x in range(3)]
  tFatesProp = [np.sum((trueFate == x))/(nData.shape[0]) for x in range(3)]
  uFates = [np.sum((assignedFate==7 & trueFate==x)) for x in range(3)]
  # tFates = [np.sum((trueFate == x)[discovered==True]) for x in range(3)]
  # tFatesProp = [np.sum((trueFate == x)[discovered==True])/(np.sum(discovered==True)) for x in range(3)]
  # uFates = [np.sum((assignedFate==7 & trueFate==x))[discovered==True] for x in range(3)]
  # uFates = [np.sum((assignedFate==))]
  print(
      "\t\t==> DISCOVERED NESTS - count (proportion):"
      "\t\t>> assigned [H D Fl U]: {aFates} ({aFatesProp})", 
      "\t\t>> true [H D Fl]: {tFates} ({tFatesProp})", 
      )
  if False:
    print(
        ">> assigned fate (hatched, depredated, flooded, unknown):", 
        # ">> assigned fate proportions (hatched, depredated, flooded, unknown):", 
        aFatesProp[0:3], 
        (np.sum(discovered==True) - np.sum(aFates)) / (np.sum(discovered==True)),
        # (np.sum(aFates==7) )/ (np.sum(discovered==True)),
        "\n\n>> proportions of known (assigned) fates (H, D, F):",
        aFates[0:3]/np.sum(aFates),
        "\n>> vs. true proportions for discovered only (H, D, F):",
        tFatesProp[0:3],
        # np.sum(discovered==True)- np.sum(tFates)
        )

def print_mark(nprintRow=5, print_exp=False):
  """
    ARGS: 
      loads "mark" from file
      columns: [0]=allp [1:3]=expo [4]=s(from mark) [5]=nest ID 
  """
  print("\t\t\t>>>>> Program MARK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  if print_exp:
    inp = np.load("out/inp.npy")
    expo = np.load("out/exposure.npy")
    print("\t\t\t\t|> input values for calc_exp:")
    for n in range(nprintRow):
      print(
            f"\t\t\t\t\tnest {n}: first found={inp[n,0]}"
            f"\tlast active={inp[n,1]} \tlast checked={inp[n,2]}"
          )
    print( f"\t\t\t\t|> output from exposure function:")
    for n in range(nprintRow):
      print(
            f"\t\t\t\t\tnest {n}: alive days={expo[n,0]}"
            f"\tfinal_int={expo[n,1]} \texposure={expo[n,2]}"
          )
  mark = np.load("out/MARK_print.npy")
  s    = mark[:,4]
  print(f"\t\t\t\t\t> ID: s * alive_days * (1 - (s**final_int))"
        f" = prob | number of nests: {mark.shape[0]}"
        f" | ret = {s[0]}"
        ) ## +> num rows
  if mark.shape[0] > nprintRow: nprintRow=mark.shape[0]
  for i in range(nprintRow):
    print(
        f"\t\t\t\t\t>nest {mark[i,5]:.0f} [fate:{mark[i,6]:.0f}]:"
        f" {s[i]:.5f}**{mark[i,1]} *"
        f" (1 - {s[i]:.5f}**{mark[i,2]:.5f}) = {mark[i,0]:.5f}"
        )
  print(f"\t\t\t\t\t\t|> total negative log likelihood (sum of probs) = {np.sum(mark[:,0])}")

def print_observer(nData, svysTilDiscovery, discovered):
  print("surveys til discovery; discovered T/F:", svysTilDiscovery, discovered)
  print("FATES:")
  print(f"\ttrue:{nData[a,3]} | assigned:{nData[a,7]}" for a in range(nData.shape[1]))

def print_mk_fates():
  """
    Print data after calling the mk_fates function
  """
  print("\t>> ", )

def print_mayf(expo):
  print("output from exposure function:", expo)
  
  for n in range(len(expo)):
    print(
      "days nest was alive:", expo[n,0],
      "& final int:", expo[n,1], 
      "& exposure:", expo[n,2]
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
  discovered = discover.sum()
  excluded   = exclude.sum()
  print("nests not discovered:", nestData[:,0][~discover])
  print("nests to exclude from analysis:", nestData[:,0][exclude])
  print("nest data, analysis nests only:\n",
          "ID, init, survival, true fate, i, j, k, assigned fate, num normal obs, intFinal, num storms:\n",
           nestData[~discover and exclude])

# def print_disc(discovered, trueHatch, survival):
def print_disc(discovered, hatched, exposure):
  """
    Print info about discovered nests (DSR, etc)
    ----
    ARGS:
      for each nest:
      > was nest discovered?
      > did nest hatch?
      > how many exposure days?
    ----

  """
  # NOTE issue may be that not enough flooded nests are discovered, not that too many hatched nests are
  numDisc = discovered.sum()
  # numDiscH = trueHatch[discovered==True].sum()
  numDiscH = hatched[discovered==True].sum()
  expDisc  = exposure[discovered==True].sum()
  # survDisc = survival[discovered==True].sum()
  print(">> number discovered:", numDisc,
        ", number discovered that hatched:", numDiscH,
        ", and exposure days (discovered nests):", expDisc)
  # trueDSR_disc = 1 - ((numDisc - numDiscH) / survDisc) # num failed / total exposure days

  #+> daily mortality rate = num failed / total exposure days
  trueDSR_disc = 1 - ((numDisc - numDiscH) / expDisc) 
  print(">> apparent DSR of discovered nests only:", trueDSR_disc) 
  
  # print(">> nest discovered?", discovered)
  # print(">> discovered & hatched:", discovered[trueHatch==True])
  # print(">> discovered & hatched(list):", discovered[hatched==True])
  # print(">> calculate exposure days for discovered nests by summing this list:", survival[discovered==True])
  # trueDSR_disc = 1 - ((discovered.sum() - hatched.sum()) / survival.sum()) 


    # def pr_fates_dsr(nData, expo, trueDSR, nestType):
                # if debug:
                #     print(
                #         "> DISCOVERED NESTS - total | analyzed: hatched:", 
                #         discovered, "|", analyzed,
                #         # "excluded from analysis:", excluded,
                #         "failed:", failed, "|", failed2,
                #         # nestData.shape[0] - sum(nestData[:,3])
                #         # "exposure days:", expDays, "|", sum(nestData[:,15])
                #         "true DSR:", trueDSR_disc, "|", trueDSR_an
                #         )
# -----------------------------------------------------------------------------
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
# def printLL(numNests, logLik, logLikFin, numInt, logL, ids, fate):
# -----------------------------------------------------------------------------
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
 
  #@#print(">> nest data:\n----id--ini-end-hch-fld-std-dsc-i--j--k-fate-nobs-sfin-nstm\n", nestData)
# -----------------------------------------------------------------------------
  # if cn.debugObs: print("nestID, init, end, tfate, i, j, k, afate, nnobs, intFin:\n", 
  #             np.concatenate((nData,fate[:,None],out), axis=1))
  # if cn.debugObs: print("total observed days:\n", out[:,2]-out[:,0], 
  #             "& total days nest was active:\n", nData[:,2] - nData[:,1])
  
  ##+> have to transpose to rows before unpacking:
  # init, end, trFate, i, j, k, asFate, nInt = nData.T[:,1:8]
  # print(f"{init=}")

# -----------------------------------------------------------------------------
# def print_prop(assignedFate, trueFate, discovered):  
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

