from datetime import datetime
import numpy as np
import os
from pathlib import Path
import sys
from typing import Dict, Generator
import yaml
# from datsim import config
from getClass import Config

def load_config(fpath, debug=False):
    with open(fpath, "r") as cfg:
        my_conf = yaml.safe_load(cfg)
    if debug: print(">=> config:\n", my_conf)
    my_conf = Config(**my_conf)
    if debug: print(">=> convert to class:", my_conf)
    return my_conf

config=load_config("/home/wodehouse/Projects/sim_model/config.yaml", debug=True)
debug = config.debug

# debug=False

def mk_param_list(parList: Dict[str, list], fdir: str) -> list:
    """
    Take the dictionary of lists of param values, then unpack the lists to a 
    list of lists. Then feed this list of lists to itertools.product using *.
    
    Can also uncomment some code to write entire set of param lists to csv.
    
    Returns
    -----
    a list of dicts representing all possible param combos, with keys!
    
    Notes
    -----
    product takes any number of iterables as input;
    input in the original is a bunch of lists;
    output in the original is a list of tuples

    """
    print(f"using the {parList} params lists")
    listVal = [parList[key] for key in parList]
    p_List = list(itertools.product(*listVal))
    # print(p_List)
    # plfile = Path(fdir / "param-lists.csv")
    # plfile = Path(fdir , "param-lists.csv")
    plfile = os.path.join(fdir, "param-lists.csv")
    with open(plfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(p_List)
    # make this list of lists into a list of dicts with the original keys 
    paramsList = [dict(zip(parList.keys(), p_List[x])) for x in range(len(p_List))]
    
    return(paramsList)
# def stormGen(frq, dur, wStart, pStorm):
def searchSorted2(a, b):
    """Get the index of where b would be located in a
    If bis in a, then return the index of that value instead of the next value
    """
    #out = np.zeros(a.shape)
    out = np.zeros((a.shape[0], len(b)))
    # out2 = np.zeros((a.shape[0], len(b)))
    for i in range(len(a)):
        #out[i] = np.searchsorted(a[i], b[i])
        #print("sorted search of\n", b, "within\n", a[i])
        # if debug: print(">> sorted search of", b, "within", a[i])
        out[i] = np.searchsorted(a[i], b)
        # if debug: print("sorted search of\n", b, "within\n", a[i], ":\n", out, out.shape)
        # if out[i] in b: out2[]
        # if debug: print("sorted search of\n", b, "within\n", a[i], "after accounting for exact match:\n", out, out.shape)
        # if debug: print("sorted search of\n", b, "within\n", a[i], ":\n", out[i])
        #print("index positions:", out, out.shape)
        # shouldn't the output have the shape of b?
    #print(">> index positions:\n", out, out.shape)
    return(out)
# -----------------------------------------------------------------------------
#@profile
def in1d_sorted(A,B): 
    """
    This function computes intersection of 2 arrays more quickly than intersect1d
        > ex: possible observations = intersection of observable & survey days
        
        Gets the index of each B if they were inserted in A, in order.
            > idx = np.searchsorted(B, A)

        Then makes the index of the last one zero?
            > idx[idx==len(B)] = 0

        Returns: 
            A value where A == B for each B index

                > A[B[idx] == A]
    """
    idx = np.searchsorted(B, A)
    idx[idx==len(B)] = 0
    return A[B[idx] == A]
# -----------------------------------------------------------------------------

def expDecay(n0, k, t):
    """
    Exponential decay function.

    n0 = initial value
    k  = rate of decay
    t  = time
    """
    # return(n0 * (1-lam) ** t)
    return n0 * np.exp(-k * t) 



def init_from_csv(file):
        # file="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"):
        # file=storm_init):
    """
    import initiation probabilities by week (based on real nest data).

    returns dict with key=weekStart, value=initProb
    """
    init= np.genfromtxt(
            #fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
            fname=file,
            dtype=float,
            delimiter=",",
            skip_header=1,
            usecols=2
            )
    # the initprob decimals in the csv don't sum to 1 anymore
    initProb = init / np.sum(init) # make them into probabilities again
    init_weeks = np.arange(14,29,1)
    weekStart = (init_weeks * 7) - 90 # why minus 90?
    weekStart = weekStart.astype(int)
    # return(dict(zip(weekStart, initProb)))
    ret = dict(zip(weekStart, initProb))
    if debug: print(">> week: init probability = ",ret)
    return(ret)
    # return(initProb)
# -----------------------------------------------------------------------------
def sprob_from_csv(file):
        # file="C:/Users/Sarah/Dropbox/Models/sim_model/storm_init3.csv"):
    """
    import storm probabilities by week (based on real storm data).

    returns dict with key=weekStart, value=stormProb
    """
    stormProb = np.genfromtxt(
            #fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
            fname=file,
            dtype=float,
            delimiter=",",
            skip_header=1,
            usecols=3 # 4th column 
            )
    storm_weeks2 = np.arange(14,29,1)
    weekStart = (storm_weeks2 * 7) - 90 # why minus 90?
    weekStart = weekStart.astype(int)
    ret = dict(zip(weekStart, stormProb))
    if debug: print(">> week start date: storm probability =\n",ret)
    return(ret)

#--- PRINTING: -----------------------------------------------------------------
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
# --- PRINT FUNCTIONS ---------------------------------------------------------
# -----------------------------------------------------------------------------
def uniquify(path):
    """
    from https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    
    Adds a number to the end of duplicate filenames
    """
    filename, extension = os.path.splitext(path)
    counter = 1
    # print("filename, extension:", filename, extension)

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path
# -----------------------------------------------------------------------------
# def mk_fnames(suf:str, f_dir, unique=True):
def mk_fnames(suf:str, con=config, unique=True):
    """
    1. Create a directory w/ a unique name using datetime.today() & uniquify().
    2. Create likelihood filepath (& parent dir, if necessary)
    3. Make a string out of the column names that can be used w/ np.savetxt()
    
    Returns:
    tuple of likelihood filepath & colnames string
    """
    like_f_dir = con.likeDir
    if unique:
        now    = datetime.today().strftime('%m%d%Y_%H%M%S')
        fdir   = Path(uniquify(Path.home()/ like_f_dir / (now + suf)))
        fname  = "ml_val_" + now + suf + ".csv"
        likeF  = Path(fdir / fname)
            
        # likeF = Path(uniquify(Path.home() / like_f_dir / now + suf / fname ))
                                    #    'C://Users/Sarah/Dropbox/Models/sim_model/py_output' / 
                                   # fname))
        likeF.parent.mkdir(parents=True, exist_ok=True)
    else:
        now = datetime.today().strftime("%Y%m%d")
        # fname  = f"ml_val_{now}.csv"
        fdir = Path(Path.home() / like_f_dir / (now + suf)) # need the parens or get an error about concatenating string and Path?
        fname  = "ml_val_" + now + suf +".csv"
        likeF = Path(fdir / fname )
        # print(likeF)
        # likeF = Path(Path.home() / like_f_dir / now + suf / fname )
        likeF.parent.mkdir(parents=True, exist_ok=True)
        
        # likeF = Path(Path.home()/'Dropbox/Models/sim_model/py_output/'/fname)
    # f_dir = "C:/Users/Sarah/Dropbox/Models/sim_model/py_output/"
    # fpath = Path(f_dir/ fname)
    # f_dir = con.likeDir
    fpath = like_f_dir + fname
    # print(fpath)
    # with open('likeFile-name.txt', 'w' ) as f:
    lfname = Path(fdir / 'likeFile-name.txt')
    with open(lfname, 'w' ) as f:
        # f.write(str(likeF))
        f.write(str(fpath))
    column_names = np.array([
        # 'mark_s', 'psurv_est', 'ppred_est', 'pfl_est', 'ss_est', 'mps_est', 'mfs_est',
        'mark_s', 'psurv_est', 'ppred_est',
        # 'ps_given', 'dur', 'freq', 'n_nest', 'h_time', 'obs_fr',
        'trueDSR', 'trueDSR_analysis', 'discovered', 'excluded', 'unknown', 'misclass','flooded','hatched',
        # 'nExc', 'repID', 'parID','psurv_est2', 'ppred_est2'
        'nExc', 'repID', 'parID'
        # 'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 
        # 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 
        # 'storm_freq', 'psurv_real', 'psurv_found', 'psurv_given',
        # 'stormsurv_given','pflood_given', 'hatch_time','num_nests',
        # # 'obs_int', 'num_discovered','num_excluded', 'exception'
        # 'obs_int', 'num_discovered','num_excluded'
        # 'rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 
        # 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 
        # 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 
        # 'obs_int', 'num_discovered','num_excluded', 'exception'
        ])
    colnames = ', '.join([str(x) for x in column_names]) # needs to be string

    # saveNames = dict(
    #     likeFile   = likeF,
    #     dirName    = datetime.today().strftime('%m%d%Y_%H%M%S'),
    #     todaysDate = datetime.today().strftime("%Y%m%d"),
    #     colnames = ', '.join([str(x) for x in column_names]) # needs to be string
    # )
    # print(">> save directory name:", dirName)
    print(">> likelihood file path:", likeF)
    # return(saveNames)
    return(likeF, colnames)
    # return(fdir, likeF, colnames)
# -----------------------------------------------------------------------------

def print_observer(svysTilDiscovery, discovered):
    print("surveys til discovery; discovered T/F:", svysTilDiscovery, discovered)
    # if cn.debugObs: print("nestID, init, end, tfate, i, j, k, afate, nnobs, intFin:\n", 
    #                       np.concatenate((nData,fate[:,None],out), axis=1))
    # if cn.debugObs: print("total observed days:\n", out[:,2]-out[:,0], 
    #                       "& total days nest was active:\n", nData[:,2] - nData[:,1])
    
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
    #     ">> discovered?, fate, i, j, k, num surveys, num storm intervals:\n", 
    #     nests[:5,:], 
    #     "\n ... \n",
    #     nests[-5:,:]
    #     )

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
                #     print(
                #         "\n>> assigned DSR:",
                #         pSurv,
                #         "true DSR of all nests:", 
                #         trueDSR, 
                #         "discovered nests:",
                #         trueDSR_disc,
                #         "and nests used in analysis:", 
                #         trueDSR_an
                #         )
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
    #     print(
    #         # f"> {nestType} nests - hatched:", hatched.sum(),
    #         f"> {nestType} nests - hatched:", hatched,
    #         "; failed:", nNests - hatched, 
    #         # "; exposure days:", expDays[:,2].sum(),
    #         "; exposure days:", expDays.sum(),
    #         "; & Mayfield-40 DSR:", 1-dmr
    #         )
# -----------------------------------------------------------------------------
def printLL(numNests, logLik, logLikFin, numInt, logL):
    """ print entire likelihood equation """

    for x in range(numNests):
        # print(">> likelihood equation: (",logLik[x],"*",numIntNorm[x],")+(",logLikStm[x],"*",numIntStm[x],")+(",logLikFin[x],"**(1 -",stormDuringFin[x],")+(", logLikFinStm[x],"**",stormDuringFin[x])
        print(
           f">> likelihood equation for nest {x}: " 
        #    f"{numInt[x]:.0f} * {logLik[x]:.5f} + "
           f"({numInt[x]:.0f} * {logLik[x]:.5f}) + {logLikFin[x]:.5f} ="
        #    f"{logLikFinStm[x]:.5f} * (1-{stormDuringFin[x]:.0f}) + " 
        #    f"{logLikFinStm[x]:.2f} * {stormDuringFin[x]:.2f} = "
        #    f"{logLikelihood[x]:.2f}")
           f"{logL[x]:.2f}"
           )
# -----------------------------------------------------------------------------
# def logL(numNests, normalInt, finalInt, numInt, ha, config=config):
# def logL(numNests, intervals, numInt, ha, fl, config=config):
# @profile
# -----------------------------------------------------------------------------
#   PROGRAM MARK 
# -----------------------------------------------------------------------------
# The model used in Program MARK is based on Dinsmore (2002) -  
#        allows for variance in DSR & use of covariates

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

def print_nest_info(nestData, discover, exclude):
    if debug: print("nests not discovered:", nestData[:,0][~discover])
    if debug: print("nests to exclude from analysis:", nestData[:,0][exclude])
    if debug: print("nest data, analysis nests only:\n",
                    "ID, init, survival, true fate, i, j, k, assigned fate, num normal obs, intFinal, num storms:\n",
                     nestData[~discover and exclude])
# def main(fnUnique, useWSL=False, testing=True, deb="none", debN="none", debS="none", config=config, pStatic=staticPar):
# def main(fnUnique, testing=config.testing, deb="none", debN="none", debS="none", config=config, pStatic=staticPar):
# def main(fnUnique, debugOpt, testing=config.testing, config=config, pStatic=staticPar): # calls config too early
# @profile
# def main(fnUnique, debugOpt, testing, pList, config=config, pStatic=staticPar):
#def like_old(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData, stormDays, surveyDays, obs_int):


#--- OLD: ----------------------------------------------------------------------
def like_old(argL, obsFreq, nestData, surveyDays, stormDays, numNests):
    """

    This function computes the overall likelyhood of the data given the model parameter estimates.
    
    The model parameters are expected to be received in the following order:
    - a_s   = probability of survival during non-storm days
    - a_mp  = conditional probability of predation given failure during non-storm days
    - a_mf  = conditional probability of flooding given failure during non-storm days
    - a_ss  = probability of survival during storm days
    - a_mfs = conditional probability of predation given failure during storm days
    - a_mps = conxditional probability of predation given failure during storm days
    - sM    = for program MARK?

    """
  
    nCol = 18
    a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
    obs_int = obsFreq
    likeData = np.zeros(shape=(numNests, nCol), dtype=np.longdouble) 

    stillAlive  = np.array([1, 0, 0])
    mortFlood   = np.array([0, 1, 0])
    mortPred    = np.array([0, 0, 1])

    # > starting matrix, from etterson 2007:
    startMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
    # > use this matrix for storm weeks:
    stormMatrix = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]]) 
        # how is this matrix actually being incorporated during the analysis?

    logLike = Decimal(0.0)          # initialize the overall likelihood counter
    #logLike = float(logLike)
    rowNum = 0
    for row in nestData:
    # columns for the likelihood comparison datarframe:
    #         num obs, log lik of nest, log lik period 1, period 2, etc.  
        
    # FOR EACH NEST -------------------------------------------------------------------------------------

        nest    = row         # choose one nest (multiple output components from mk_obs())
        # print('obs_int check: ', obs_int)

        # disc    = nest[7].astype(int)    # first found
        disc    = nest[0].astype(int)    # first found
        # endObs  = nest[9].astype(int)    # last observed
        endObs  = nest[2].astype(int)    # last observed
        # hatched = nest[3]
        # flooded = nest[4]
        hatched = nest[3] == 0
        flooded = nest[3] == 2

        if flooded == True & hatched == False:
            fate = 2
        elif flooded == False & hatched == False:
            fate = 3
        else:
            fate = 1
        # fate    = nest[7].astype(int)    # assigned fate

        if np.isnan(disc):
            ###print("this nest was not discovered but made it through")
            continue
        
        num     = len(np.arange(disc, endObs, obs_int)) + 1 # number o observations
        # print('#############################################################')
        # print('nest =', nest[0], 'row number =', rowNum, 'number of obs =', num)

        likeData[rowNum, 0:3]  = np.array([nest[0], fate, num] )

        # print("num=", num)
        obsDays = in1d_sorted(
            # (np.linspace(disc, endObs, num=num)), surveyDays)
            np.linspace(disc, endObs, num=num),
            surveyDays)
        # print("obs days for nest:", obsDays)
        obsPairs = np.fromiter(
            itertools.pairwise(obsDays), 
            dtype=np.dtype((int,2))
            ) # do elements of numpy arrays have to be floats?
        # print("date pairs in observation period:", obsPairs) 

        # > make a list of intervals between each pair of observations 
        #   (necessary for likelihood function)
        intList = obsPairs[:,1] - obsPairs[:,0]
        # print("interval list:", intList)
        
        # > start off with all intervals = alive
        obs     = [stillAlive for _ in range(len(obsPairs)+1)] 

        # > change the last obs if nest failed:
        if fate == 2:
            obs[-1] = mortFlood
        elif fate == 3:
            obs[-1] = mortPred

        # print("fate, obs = ", fate, " , ", obs) # check that last entry in obs corresponds to fate

        # if hatch, leave as is?

        # place this likelihood counter inside the for loop so it resets 
        # with each nest:
        logLikelihood = Decimal(0.0)   
        #logLikelihood = float(logLikelihood)

        # likeData[0, rowNum] = nest[0]
        obsNum = 0
        for i in range(len(obs)-1):
        # FOR EACH OBSERVATION OF THIS NEST ---------------------------------------------------------------------

            # print("observation number:", obsNum)
            
            intElt  = (intList[i-1]).astype(int)  # access the (i-1)th element of intList,
                                    # which is the interval from the (i-1)th
                                    # to the ith observation

            #stateF  = obs[i]
            stateF  = obs[i+1] 
            stateI  = obs[i]
            # print("stateF:",stateF)
            TstateI = np.transpose(stateI)
            # print("TstateI:", TstateI)

            # if any(d in storm_days for d in range(i-1, i)):
            if any(d in stormDays for d in range(i-1, i)):
                # if any of the days in the current observation interval (range) is also in storm days, use storm matrix
                # print("using storm matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(stormMatrix, intElt))
                # this is the dot product of the current state of the nest and the storm matrix ^ interval length
           # look into using @ instead of nest dot calls 
            else:
                # print("using normal matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(startMatrix, intElt))

            lPer = np.dot(lDay, TstateI)
            # print("likelihood for this interval:", lPer)

            logL = Decimal(- np.log(lPer))
            # print("negative log likelihood of this interval:", logL)

            #logL = float(logL)

            logLikelihood = logLikelihood + logL # add in the likelihood for this one observation
            # print("log likelihood of nest observation history:", logLikelihood)
            colNum = obsNum + 4 
            likeData[rowNum, colNum] = logL
            obsNum = obsNum + 1

        likeData[rowNum,3] = logLikelihood
        logLike = logLike + logLikelihood        # add in the likelihood for the observation history of this nest
        rowNum  = rowNum + 1
        # print("increment row number:", rowNum)
        # print("overall log likelihood so far:", logLike)
        

    # print(
    #     "nest num, fate, num obs, lik(obs hist), lik(each obs int) ... \n",
    #     likeData[:10,:] # print first ten rows
    # )
    return(logLike) 
