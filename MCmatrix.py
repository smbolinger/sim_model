import numpy as np
# from datsim import config ## in case anything was changed in place

def triangle(x0, y0):
    """
    This function remaps values from R^2 into the lower left triangle located 
    within the unit square.
    """
    if y0 > x0:
        ret = triangle(y0, x0)
        return ret[1], ret[0]

    r0 = np.sqrt( x0**2 + y0**2)
    m  = 1.0
    if y0 != x0:
        m = y0/x0

    theta = np.arctan(m)
    r3    = r0 * 1.0/(1.0 + m)
    x3    = r3 * np.cos(theta)
    y3    = r3 * np.sin(theta)
    return x3, y3
# -----------------------------------------------------------------------------
#def logistic(x)->np.float128:
def logistic(x)->np.longdouble:
    """This is just the logistic function"""
    # Trying out type hints (PEP 484) to keep output from overflowing
    #return 1.0/( 1.0 + math.exp(-x) )
    return 1.0/( 1.0 + np.exp(-x) )
# -----------------------------------------------------------------------------

def state_vect(nNest, fl, ha):# can maybe calculate these only once
# def state_vect(flooded, hatched):# can maybe calculate these only once
    """
    numNests is not the total number (param value) but the number not excluded
    1. Define state vectors (1x3 matrices) - all the possible nest states 
          [1 0 0] (alive)   [0 1 0] (fail-predation)   [0 0 1] (fail-flood) 
    ---------------------------------------------------------------------------------------------------
    2. Create arrays to hold state vectors for all nests:
       a. state of nest on date nest was first found (stateFF)
       b. state of nest on date nest was last checked (stateLC) - this is the fate as observed
    Could also calculate bassed on nest fate value
    """
    stillAlive = np.array([1,0,0]) 
    mortFlood  = np.array([0,1,0])
    mortPred   = np.array([0,0,1])
    # FOR THE INITIAL STATE (stateFF), just one vector (see notebook) - later in code
    # numNests = len
    stateEnd    = np.empty((nNest, 3))     # state at end of normal interval
    stateLC     = np.empty((nNest, 3))     # state at end of final interval
     # > use broadcasting - fill doesn't work with arrays as the fill value:
    stateEnd[:] = stillAlive # alive at end of normal interval
    stateLC[:]  = mortPred   # default is still depredation

    stateLC[fl==True] = mortFlood  # flooded status gets flooded state vector
    stateLC[ha==True] = stillAlive # hatched nests stay alive the entire time

    # nests always start alive, or they wouldn't be checked
    # TstateI = np.transpose(stillAlive)  # this is just one, not a vector?
    
    return([stateEnd, stateLC])
    ##print(">> transpose of initial state vector:", TstateI, TstateI.shape)
    #                                             this will depend on how many storms/when they are
    # NOTE need the transpose of each row, not entire matrix. 
    # print(">> state at the end of a normal interval:\n",stateEnd) 
    # print(">> state on final nest observation:\n", stateLC)
    # ---------------------------------------------------------------------------------------------------

    # in the following code, we calculate all matrix multiplications for all nests, and then turn them on/off
    # based on whether we need them. Would it be faster to only do the multiplications we need? That would require indexing
    # maybe calculate the one for normal and storm intervals (not final interval) once, and the final intervals for each nest
    # (based on what the actual fate is)
    # > all nests start interval as alive - don't need stateFF
    # > transition matrix, from etterson 2007:
    # the values being optimized are integral to the matrices, so can't pull that out
    # NOTE NOTE which of these are vectorized and which aren't??
# def nest_mat(a_s, a_mf, a_mp, obsFreq, stormTrue):
# def interval(pwr, stateEnd, stateLC ): 
# ---------------------------------------------------------------------------------------------------
# def nest_mat(argL, obsFreq, stormFin, useStormMat, config):
# @profile

def nest_mat(argL, obsFreq, stormFin, useStormMat, config):
    """
    Purpose
    -------
    1. Create transition matrix for normal intervals & storm intervals
    2. Raise transition matrix to the power of interval length
    - intervals with storms are longer (obs_int x 2)
    - there is no separate storm matrix anymore (see notes)
    
    Inputs
    ------
    - argL = vals for the minimizer
    - useStormMat = T/F should you use the storm transition matrix for storm intervals
    
    Returns
    -------
    - list containing the two matrices [pwr, pwrStm]
    
    Background
    ----------
    
    Compose the matrix equation for one observation interval. The formula used is from Etterson et al. (2007) 
    For this, you need: 
       > intElt - length in days of the observation interval being assessed 
       > initial state (stateI) - state of the nest at the beginning of this interval 
       > stateF - state of the nes at the end of this interval
       There is a transition matrix that is multiplied for each day in the interval 
       > in this case, the nest started the interval alive and ended it alive as well 
       > daily nest probabilities: s - survival; mp - mortality from predation; mf - mortality from flooding 
       > these are daily probabilities, so raise transition matrix to the power of number of days in interval  
    
                                        _         _  intElt           _   _ 
                 [ 1 0 0 ]             |  s  0  0  |                 |  1  | 
                                *      |  mp 1  0  |            *    |  0  | 
                                       |_ mf 0  1 _|                 |_ 0 _|  
                                 
         {  transpose(stateI) * trMatrix, raised to intElt power * stateF } 
    
    Then, you can multiply this equation times number of intervals (numIntTotal)
       Single in  interval --> all intervals --> likelihood

    """
    
    # a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
    a_s, a_mp, a_mf = argL
    # if debug: print("observation interval, storm in final interval?, use storm matrix?\n",obsFreq, stormFin, useStormMat)
    trMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
    pwr = np.linalg.matrix_power(trMatrix, obsFreq) # raise the matrix to the power of the number of days in obs int
    # storm matrix just has a longer observation interval
    pwrStm = np.linalg.matrix_power(trMatrix, obsFreq*2) 

    return([pwr, pwrStm])
# def logL(normalInt, normalFinal, stormFinal, numInt):
# ---------------------------------------------------------------------------------------------------
# def interval(pwr, stateMat, cn=config): 
# def interval(pwr,numNests, fl, ha ,cn=config): 
# @profile

def interval(pwr, numNests, fl, pr, cn): 
# def interval(pwr,numNests,sNest, fl, ha ,cn=config): 
    """
    Purpose
    -------
    Create matrix multiplication to get the likelihood for each type of interval:
    - active nests in all intervals start as alive
    - regular (alive-->alive), storm(alive-->alive during storm -- rare),
      final (alive-->failed), storm final (alive-->flooded)
    - final interval for hatched nests = 0
    - longer obs int for nests that survived storms should already be accounted for? 

    Arguments
    ---------
    - pwr = output from nest_mat()
    - stateMat = array of 1x3 end-of-interval and final matrices for all nests
    
    Returns
    -------
    list of 4 likelihood values, one for each type of interval:
    [normalInt, stormInt, finalInt, stormFinal]

    """
    stillAlive = np.array([1,0,0]) 
    mortFlood  = np.array([0,1,0])
    mortPred   = np.array([0,0,1])
    
    # nests always start alive, or they wouldn't be checked
    TstateI = np.transpose(stillAlive)  # this is just one, not a vector? yes
    pwrN, pwrS = pwr
    
    # --------------------------------------------------------------------------
    # DO MATRIX MULTIPLICATION ONCE AND BROADCAST TO ARRAY:
    # matrix multiplication resolves to a single value:
    oneNormInt = stillAlive@pwrN@TstateI
    oneFinalPr = mortPred@pwrN@TstateI
    oneFinalSt = mortFlood@pwrS@TstateI
    
    normalInt  = np.empty((numNests))
    finalInt   = np.ones((numNests))
    normalInt.fill(oneNormInt)
    finalInt[pr] = oneFinalPr
    finalInt[fl] = oneFinalSt
    # there is no final interval for hatched. log(1) will become zero beforee summing
    # --------------------------------------------------------------------------
    # DO MATRIX MULTIPLICATION ON THE ARRAYS:
    # stateEnd, stateLC = stateMat
    # # if cn.debugLL: print(">> end state of normal / final interval:\n", stateEnd, "/", stateLC)
    # # if cn.debugLL: print(">> end state of normal / final interval:\n",np.column_stack(((stateEnd), (stateLC))))
    # normalInt = stateEnd@pwrN@TstateI
    # stormInt = stateEnd@pwrS@TstateI
    # # oneStInt   = 
    # # The final interval is one of these two (ends in final state):
    # # normalFinal = stateLC@pwr@TstateI
    # finalInt = stateLC@pwrN@TstateI
    # # stormFinal  = stateLC@pwrS@TstateI
    # # finalInt[stormTrue] = stormFinal
    # --------------------------------------------------------------------------
    # if cn.debugLL: 
    #     print(">>>> ")
    #     print(">> end state of normal interval:\n", stateEnd)
    #     print(">> end state of final interval:\n",  stateLC)
    #     print(">> likelihood of one normal interval:\n",
    #     normalInt,
    #     normalInt.shape,
    #     normalInt.dtype)
    #     print(">> likelihood of final interval:\n",
    #     finalInt,
    #     finalInt.shape,
    #     finalInt.dtype)
    # NOTE now pwr has storms incorporated
    # print("final interval:", normalFinal, "and -log likelihood:", -np.log(normalFinal))

    # return([normalInt, stormInt, finalInt, stormFinal])
    return([normalInt, finalInt])
# -----------------------------------------------------------------------------

def logL(numNests, intervals, numInt, config):
    """
    Purpose
    -------
    Calculate the negative log likelihood for each nest:
    1. Create a counter using Decimal for precision
    2. Create an array of -log likelihood for each nest for:
        - regular interval (where storm==True, this is the storm interval)
        - final interval (where fl==True, this is the storm final interval)
        - for matrices, 1x3 * 3x3 * 3x1 leaves you with a single value for each nest
    3. Sum the -log likelihood values to get an overall value.
    
    - using the stormInt is too complicated for how little it affects things...

    Inputs
    ------
    normalInt, finalInt, & stormFinal are output from interval() - represent likelihood of one interval
    
    ha = True if nest hatched; fl = True if nest flooded
    
    Returns
    -------
    logLike
    
    """
# def logL(normalInt, finalInt, numInt):
    # numNests is not the total number (param value) but the number not excluded
    # 2. Initialize the overall likelihood counter; Decimal gives more precision
    # numNests = len(normalInt)
    # normalInt, stormInt, finalInt, stormFinal = intervals
    normalInt, finalInt = intervals
    # print("numInt excluding final interval:", numInt)
    # NOTE NOTE does multiplying the log likelihoods together give the same result as 
    # multiplying the matrices together?
    # numInt[~ha] = numInt[~ha]+1
    # if config.debugLL: print("numInt after adding final interval:", numInt)
    # logLike = logLikelihood = Decimal(0.0)    # maybe switching types is also slowing things down?     
    logLik  = np.empty(numNests, dtype=np.longdouble) # this should give it enough precision & avoid errors
    logLikFin = np.empty(numNests, dtype=np.longdouble)
    # logLik  = logLik * np.log(normalInt) * -1 # dtype changes to float64 unless you multiply it by itself
    logLik  = -np.log(normalInt) # dtype changes to float64 unless you multiply it by itself
    # logLik[stormInt==True] = logLik * np.log(stormInt) * -1
    # if config.debugLL: 
    # logLikFin    = np.ones(numNests, dtype=np.longdouble)
    # logLikFin.fill(-np.log(normalFinal))
    # logLikFin= -np.log(normalFinal)
    logLikFin= -np.log(finalInt)
    # logLikFin[hatched == True] = 0
    # logLikFin[ha==True] = 0 
    # now stormFinal should be part of normalFinal (finalInt)
    # logLikFin[fl==True] = -np.log(stormFinal[fl==True])
    # logLikFin[]    = logLikFin * (-np.log(normalFinal))
    # logLikFinStm = logLikFinStm * (-np.log(stormFinal))

    # stormDuringFin = nestData[:,12] # was there a storm during the final interval?
    logLikelihood  = (logLik*numInt) + (logLikFin) # elementwise multiplication, then add final interval NLL
    # logLikelihood  = ne.evaluate('(logLik*numInt) + (logLikFin)')

    logLike        = np.sum(logLikelihood)
    # logLike        = ne.evaluate('sum(logLikelihood)')
    # if config.debugLL:
    #     print("number of nests:", numNests, "\n hatched:", ha, "\n flooded:, fl") 
    #     print("numInt excluding final interval:", numInt)
    #     print(">> -log likelihood of 1 interval:", logLik)
    #     print(">> -log likelihood final interval, updated with storms/hatch:\n", logLikFin)
    #     print(">> -log likelihood of each nest history:", logLikelihood)
    #     print(">>>> overall -log likelihood:", logLike)
    #     printLL(numNests=numNests, logLik=logLik, logLikFin=logLikFin, 
    #                     numInt=numInt, logL=logLikelihood)
    return(logLike) # this is what is being optimized
# -----------------------------------------------------------------------------
# try to keep these in numpy:
# def like(perfectInfo, hatchTime, argL, numNests, obsFreq, obsDat, surveyDays):
# def like(argL, numN, obsFr, obsDat, stMat, sTrue, useSM):
# This one uses a matrix multiplication equation created from building blocks
#   > so you create these blocks:
#   > a normal interval, a final interval, and a final interval with storm
# def like(argL, numN, obsFr, obsDat, stMat, useSM, con=config):
# @profile

def like(argL, numN, obsFr, obsDat, useSM, con):
    """
    perfectInfo == 0 or 1 to tell you whether you know all nest fates or not
    ---------------------------------------------------------------------------------------------------
    1. Unpack:
       a. Initial values for optimizer:
    a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
       b. Observation history values from nest data:
    fl, ha, ff, la, lc, nInt, sTrue = obsDat
    fl, ha, ff, la, lc, nInt = obsDat
    
    """
    # numpy array should be unpacked along the first dimension, so transpose:
    # if con.debugLL: print("-------------------------------------------------------------------------------------")
    # if con.debugLL: print("------------------------- the MCMC ML function --------------------------------------")
    # if con.debugLL: print("-------------------------------------------------------------------------------------")
    ff, la, lc, fate, nInt, sFinal = obsDat.T
    fl = fate==2 # are these necessary for more than print statements?
    # ha = fate==0
    pr = fate==1
    # NOTE NOTE should this be the assigned fate or true fate?
    # NOTE these if .. print statements, esp inside the optimizer, take lots of time:
    # if con.debugLL:
    #     print("number of nests and observation interval:", numN, obsFr)
    #     print("nest observation data - ff, la, lc, fate, nInt, sFinal\n", obsDat)
    #     print("did nest hatch?\n", ha, "\ndid nest flood?\n", fl)
    #     print(
    #         ">>>> run likelihood function - ",
    #         "num flooded:", sum(fl), "& hatched:", sum(ha),
    #         "number of obs intervals:", nInt,
    #         # "\n>> ijk:\n", [ff, la, lc]
    #         "\n>> ijk:\n", np.column_stack((ff, la, lc))
    #         )
    # stEnd, stFin = stMat 
    # pwrOut = nest_mat(argL=argL, obsFreq=obsFr, stormFin=sFinal, useStormMat=useSM)
    pwrOut = nest_mat(argL=argL, obsFreq=obsFr, stormFin=sFinal, useStormMat=useSM, config=con)
    # pwr, pwrStm = pwrOut
    # inter  = interval(pwr=pwr, stateEnd=stEnd, stateLC=stFin)   
    # inter = interval(pwr=pwrOut, stateMat=stMat)   
    # inter = interval(pwr=pwrOut,ha=ha, fl=fl, numNests=numN)   
    inter = interval(pwr=pwrOut, fl=fl, pr=pr, numNests=numN, cn=con)   
    # norm, fin, sfin = inter
    # llVal = logL(normalInt=norm, normalFinal=fin, stormFinal=sfin, numInt=nInt)
    # llVal = logL(numNests=numN, intervals=inter, numInt=nInt, ha=ha, fl=fl)
    llVal = logL(numNests=numN, intervals=inter, numInt=nInt, config=con)
    # make sure numN is the number of analyzed nests, not the param value (total number)
    
    return(llVal)
# this function does the matrix multiplication for a SINGLE interval of length intElt days 
 # during observaton, nest state is assessed on each visit to form an observation history 
 # the function calculates the negative log likelihood of one interval from the observation history 
 # these can then be multiplied together to get the overall likelihood of the observation history 
# -----------------------------------------------------------------------------
#   THE LIKELIHOOD WRAPPER FUNCTION
# -----------------------------------------------------------------------------
# @profile

def like_smd( 
        # x, perfectInfo, hatchTime, nestData, obsFreq, 
        # stormDays, surveyDays, whichRet):
        # x, obsData, obsFreq, stateMat, useSM, stormDays, surveyDays, whichRet=1, config=config):
        # x, obsData, obsFreq, stateMat, useSM, stormDays, surveyDays, whichRet, config=config):
        x, obsData, obsFreq, useSM, stormDays, surveyDays, whichRet, config):
        # x, obsData, obsFreq, stateMat, useSM, whichRet=1, **kwargs):
    # use kwargs for stormDays & surveysDays, which are only needed w/ like_old
    """
    Arguments:
        x: output from randArgs(); 5 values, only 4 used by LL function
        
        obsData: just the columns from observer() output: 
        i, j, k, assigned fate, num obs, finalIntStorm
        
        whichRet: which like() function should be used? 1=new (default); 2=old

    The values are log-transformed before running them thru the likelihood 
    function, so the values given to the optimizer are the untransformed values,
    meaning the optimizer output will also be untransformed.
        >> Therefore, need to transform the output as well.
    """


    # unpack the initial values:
    s0   = x[0]
    mp0  = x[1]
    # ss0  = x[2]
    # mps0 = x[3]
    # sM   = x[4]
    # if config.debugLL: print("initial values:", s0, mp0, ss0, mps0, sM)

    # transform the initial values so all are between 0 and 1:
    s1   = logistic(s0)
    mp1  = logistic(mp0)
    # ss1  = logistic(ss0)
    # mps1 = logistic(mps0)
    #@#print("logistic-transformed initial values:", s1, mp1, ss1, mps1)

    # further transform so they remain in lower left triangle:
    tri1 = triangle(s1, mp1)
    # tri2 = triangle(ss1, mps1)
    s2   = tri1[0]
    mp2  = tri1[1]
    # ss2  = tri2[0]
    # mps2 = tri2[1]
    # if config.debugLL: print("log- & triangle-transformed initial values:", s2, mp2, ss2, mps2)

    # compute the conditional probability of mortality due to flooding:
    mf2  = 1.0 - s2 - mp2
    # mfs2 = 1.0 - ss2 - mps2

    numNests = obsData.shape[0]
    #@#print(">> number of nests:", numNests)

    # call the likelihood function:
    # argL = np.array([s2,mp2,mf2,ss2,mps2,mfs2, sM])
    argL = np.array([s2,mp2,mf2])
    #ret = like(argL, ndata, obs, storm, survey)
    #ret = like(argL, nestData, obsFreq, stormDays, surveyDays)
    # def like(argL, numN, obsFr, obsDat, stMat, useSM):
    # obsDat is a subset of nestData
    
    # if whichRet == 1:
    ret = like(argL, numN=numNests, obsFr=obsFreq, obsDat=obsData, 
    # ret = like(argL, numN=numNests, obsFr=obsFreq, obsDat=obsData, stMat=stateMat,
               useSM=useSM, con=config)
        # if config.debugLL: print('like_smd(): Msg : ret = ', ret)
    
    # else:
    # elif whichRet == 2:
    #     ret = like_old(argL, obsFreq, obsData, surveyDays[0], stormDays, numNests)
    #     print('like_smd(): Msg : using old function; ret = ', ret)
    
    # else:
    #     print(' argument whichRet is invalid ')
    
    # rets = np.array([ret, ret2])

    return(ret)

# ---------------------------------------------------------------------------------------
# def ansTransform(ans, unpack):
# if you add unpack param, can use for transforming input as well as output
# def ansTransform(ans, unpack=True):

# def like_old(argL, obsFreq, nestData, surveyDays, stormDays, numNests):
#     """
#
#     This function computes the overall likelyhood of the data given the model parameter estimates.
#
#     The model parameters are expected to be received in the following order:
#     - a_s   = probability of survival during non-storm days
#     - a_mp  = conditional probability of predation given failure during non-storm days
#     - a_mf  = conditional probability of flooding given failure during non-storm days
#     - a_ss  = probability of survival during storm days
#     - a_mfs = conditional probability of predation given failure during storm days
#     - a_mps = conxditional probability of predation given failure during storm days
#     - sM    = for program MARK?
#
#     """
#
#     nCol = 18
#     a_s, a_mp, a_mf, a_ss, a_mps, a_mfs, sM = argL
#     obs_int = obsFreq
#     likeData = np.zeros(shape=(numNests, nCol), dtype=np.longdouble) 
#
#     stillAlive  = np.array([1, 0, 0])
#     mortFlood   = np.array([0, 1, 0])
#     mortPred    = np.array([0, 0, 1])
#
#     # > starting matrix, from etterson 2007:
#     startMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) 
#     # > use this matrix for storm weeks:
#     stormMatrix = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]]) 
#         # how is this matrix actually being incorporated during the analysis?
#
#     logLike = Decimal(0.0)          # initialize the overall likelihood counter
#     #logLike = float(logLike)
#     rowNum = 0
#     for row in nestData:
#     # columns for the likelihood comparison datarframe:
#     #         num obs, log lik of nest, log lik period 1, period 2, etc.  
#
#     # FOR EACH NEST -------------------------------------------------------------------------------------
#
#         nest    = row         # choose one nest (multiple output components from mk_obs())
#         # print('obs_int check: ', obs_int)
#
#         # disc    = nest[7].astype(int)    # first found
#         disc    = nest[0].astype(int)    # first found
#         # endObs  = nest[9].astype(int)    # last observed
#         endObs  = nest[2].astype(int)    # last observed
#         # hatched = nest[3]
#         # flooded = nest[4]
#         hatched = nest[3] == 0
#         flooded = nest[3] == 2
#
#         if flooded == True & hatched == False:
#             fate = 2
#         elif flooded == False & hatched == False:
#             fate = 3
#         else:
#             fate = 1
#         # fate    = nest[7].astype(int)    # assigned fate
#
#         if np.isnan(disc):
#             ###print("this nest was not discovered but made it through")
#             continue
#
#         num     = len(np.arange(disc, endObs, obs_int)) + 1 # number o observations
#         # print('#############################################################')
#         # print('nest =', nest[0], 'row number =', rowNum, 'number of obs =', num)
#
#         likeData[rowNum, 0:3]  = np.array([nest[0], fate, num] )
#
#         # print("num=", num)
#         obsDays = in1d_sorted(
#             # (np.linspace(disc, endObs, num=num)), surveyDays)
#             np.linspace(disc, endObs, num=num),
#             surveyDays)
#         # print("obs days for nest:", obsDays)
#         obsPairs = np.fromiter(
#             itertools.pairwise(obsDays), 
#             dtype=np.dtype((int,2))
#             ) # do elements of numpy arrays have to be floats?
#         # print("date pairs in observation period:", obsPairs) 
#
#         # > make a list of intervals between each pair of observations 
#         #   (necessary for likelihood function)
#         intList = obsPairs[:,1] - obsPairs[:,0]
#         # print("interval list:", intList)
#
#         # > start off with all intervals = alive
#         obs     = [stillAlive for _ in range(len(obsPairs)+1)] 
#
#         # > change the last obs if nest failed:
#         if fate == 2:
#             obs[-1] = mortFlood
#         elif fate == 3:
#             obs[-1] = mortPred
#
#         # print("fate, obs = ", fate, " , ", obs) # check that last entry in obs corresponds to fate
#
#         # if hatch, leave as is?
#
#         # place this likelihood counter inside the for loop so it resets 
#         # with each nest:
#         logLikelihood = Decimal(0.0)   
#         #logLikelihood = float(logLikelihood)
#
#         # likeData[0, rowNum] = nest[0]
#         obsNum = 0
#         for i in range(len(obs)-1):
#         # FOR EACH OBSERVATION OF THIS NEST ---------------------------------------------------------------------
#
#             # print("observation number:", obsNum)
#
#             intElt  = (intList[i-1]).astype(int)  # access the (i-1)th element of intList,
#                                     # which is the interval from the (i-1)th
#                                     # to the ith observation
#
#             #stateF  = obs[i]
#             stateF  = obs[i+1] 
#             stateI  = obs[i]
#             # print("stateF:",stateF)
#             TstateI = np.transpose(stateI)
#             # print("TstateI:", TstateI)
#
#             # if any(d in storm_days for d in range(i-1, i)):
#             if any(d in stormDays for d in range(i-1, i)):
#                 # if any of the days in the current observation interval (range) is also in storm days, use storm matrix
#                 # print("using storm matrix")
#                 lDay = np.dot(stateF, np.linalg.matrix_power(stormMatrix, intElt))
#                 # this is the dot product of the current state of the nest and the storm matrix ^ interval length
#            # look into using @ instead of nest dot calls 
#             else:
#                 # print("using normal matrix")
#                 lDay = np.dot(stateF, np.linalg.matrix_power(startMatrix, intElt))
#
#             lPer = np.dot(lDay, TstateI)
#             # print("likelihood for this interval:", lPer)
#
#             logL = Decimal(- np.log(lPer))
#             # print("negative log likelihood of this interval:", logL)
#
#             #logL = float(logL)
#
#             logLikelihood = logLikelihood + logL # add in the likelihood for this one observation
#             # print("log likelihood of nest observation history:", logLikelihood)
#             colNum = obsNum + 4 
#             likeData[rowNum, colNum] = logL
#             obsNum = obsNum + 1
#
#         likeData[rowNum,3] = logLikelihood
#         logLike = logLike + logLikelihood        # add in the likelihood for the observation history of this nest
#         rowNum  = rowNum + 1
#         # print("increment row number:", rowNum)
#         # print("overall log likelihood so far:", logLike)
#
#
#     # print(
#     #     "nest num, fate, num obs, lik(obs hist), lik(each obs int) ... \n",
#     #     likeData[:10,:] # print first ten rows
#     # )
#     return(logLike) 
# # -----------------------------------------------------------------------------
# # def state_vect(numNests, flooded, hatched):# can maybe calculate these only once
#
