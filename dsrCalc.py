import numpy as np
# from datsim import config ## in case anything was changed in place

# def calc_exp(inp, expPercent=0.5, cn=config): 
def calc_exp(inp, expPercent=0.5, cn): 
    """
    Calculate the exposure period for a nest (number of days observed)
        
    Arguments
    ---------
    inp = [i,j,k] for all nests in set\n
    default expPercent is from Mayfield; Johnson recommended 0.4
    
    Returns
    -------
    ndarray. nrows=len(inp); cols=alive_days, final_int, exposure
    
    More info
    ----------
        > the ijk values should tell you failed vs hatched
        > I think I couldn't get it to work as vectorized, so I used a loop
    For the basic case where psurv is constant across all nests and times:
      1. count the total number of alive days when nest was observed
      2. count the number of days in the final interval (for failed nests)
      3. calculate the exposure
            > #days under observation before final int + (#days in final int * expPercent)
                > expPercent = percent of final interval nest is assumed alive
                > Mayfield used 50%, Johnson corrected it to 40%
                > final interval assumed to be zero days for hatched nests, which
                 were found after hatching (exposure of incubation period is over)
            > not calculating nestling exposure bc precocial/semi-precocial chicks
              leave the nest so early 
    """
    # expo = np.zeros((numNests, 3))
    expo = np.zeros((len(inp), 3))
    # inp = ijk (cols) for all nests (rows)
    for n in range(len(inp)-1): # want n to be the row NUMBER
        # alive_days = inp[n,1] - inp[n,0] # interval from first found to last active 
        expo[n,0] = inp[n,1] - inp[n,0] # interval from first found to last active 
                                         # all nests are KNOWN to be alive
        # expo[n,0] = expo[n,0] - 1 # since this is essentially 1-day intervals, 
                                    # need 1 fewer than total number
        expo[n,1] = inp[n,2] - inp[n,1] # interval from last active to last checked
        # if expo[n,1]!=0: expo[n,1] = expo[n,1]- 1 # for hatched nests, stays 0
        # exposure = sum(alive days) + days in final int * expPercent
        expo[n,2]   = expo[n,0] + (expo[n,1]*expPercent)
        # NOTE need nests to be alive for at least one interval
    # if cn.debugM: print("output from exposure function:", expo)
    return(expo)
# -----------------------------------------------------------------------------
# def mayfield(ndata, expo):
# def mayfield(num_fail, expo, all=False):

def mayfield(num_fail, expo):
    """ 
    The Mayfield estimator of DSR 
    
    Mayfield's original estimator was defined as: 
            > DSR = 1 - (# failed nests / # exposure days)
    so if DSR = 1 - daily mortality, then:
            > daily mortality = # failed nests / # exposure days
    
    Arguments:
        num_fail = count of failed nests (total-hatched)
        expo     = just the exposure days output of calc_exp() (out[:,2])

    Returns: the daily mortality 
    """
#    I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
    mayf = num_fail / (expo.sum())
    # if debug: print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

    return(mayf)
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
    -----------------------------------------------------------------------------

    """
    jEst = (1/srn) * sum()
# -----------------------------------------------------------------------------
# @profile

def calc_dsr(nData, nestType, calcType):
    """ 
    Calculate exposure and DSR for a given set of nests. 

    pass i,j,k from nest data to calc_exp() and then run mayfield()
    
    Returns DSR value. 
    """
    nNests  = len(nData)
    # hatched = len(nData[:,3] == 0)
    # failed  = nNests-hatched
    
    # expDays = exposure(nestData[:,6:9], numNests=numN, expPercent=0.4)
    # expDays = calc_exp(nData[:,6:9], expPercent=0.4)
    # if nestType=="all":
    if calcType=="true":
        expDays = sum((nData[:,2]-nData[:,1]))
        hatched = sum(nData[:,3] == 0)
    else:
        expDays = calc_exp(nData[:,4:7], expPercent=0.4)
        expDays = expDays[:,2]
        hatched = sum(nData[:,7] == 0)
    dmr     = mayfield(num_fail=nNests-hatched, expo=expDays)
    return(1-dmr)


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

    """
    # prob, dof = probs
    # allp, alldof = mark_probs(s=s, ndata=ndata)
    # ALL IN ONE FUNCTION:
    s      = s.item() # makes singleton array into scalar
    allp   = np.array(range(1,len(ndata)), dtype=np.longdouble) # all nest probabilities 
    expo = calc_exp(inp=ndata[:,4:7], expPercent=0.4)
    for n in range(len(ndata)-1): # want n to be the row NUMBER
        alive_days = expo[n,0] - 1
        final_int  = expo[n,1] - 1
    
        if final_int > 0: # final int for hatched nests == 0
            p   = (s**alive_days)*(1-(s**final_int)) 
        else:
            p   = s**alive_days
        allp[n]   = p # NOTE this line is throwing the Deprecation Warning
    nll = sum(-np.log(allp))
    # NOTE these if statements take up lots of time, esp inside the optimizer
    # if con.debugM:
    #     print(">>>>> Program MARK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     print("> number of nests:", len(ndata), "discovered nests:", len(disc))
    #     print("inp (ID, i, j, k, fate:)\n",inp)
    #     print("l=", l, "| s=", s, "| nocc=", nocc)
    #     print("----------------------------")
    #     print(">> all nest cell probabilities:\n", allp)
    #     print(">> all degrees of freedom:\n", alldof)
    #     print("log of all nest cell probabilities:", lnp)
    #     print(">> sum to get negative log likelihood of the data:", NLL)
    return(nll)
# -----------------------------------------------------------------------------

def mark_wrapper(srn, ndata, nocc):
    """
    This function calls the program MARK function when given a random starting 
    value (srn) and some nest data (ndata)
        > values given to optimizer are transformed then passed to MARK function
            > allows larger range of values for optimizer to work over w/o overflow
            > but values given to the function are still between 0 and 1, as required
        > Create vector to store the log-transformed values, then fill

    """
    # s = np.ones(numNests, dtype=np.longdouble) # why is s an array?
    s = logistic(srn)
    #@#print("logistic of random starting value for program MARK:", s, s.dtype)
    # the logistic function tends to overflow if it's a normal float; make it np.float128
    ret = prog_mark(s, ndata, nocc)
    #@#print("ret=", ret)
    return ret
# -----------------------------------------------------------------------------
#   THE LIKELIHOOD FUNCTION
# -----------------------------------------------------------------------------
#def like_old(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData, stormDays, surveyDays, obs_int):
