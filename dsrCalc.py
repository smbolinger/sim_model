import numpy as np
from MCmatrix import logistic
from observer import svy_position
import pandas as pd
import matplotlib as pl
from print_func import arrPrint, dfPrint
from settings import config
import itertools
# from marginaleffects import predictions, datagrid
# from marginaleffects import *
import statsmodels.formula.api as smf
import statsmodels.api as sm
import rpy2
import rpy2.robjects as robj
from rpy2.robjects.packages import importr, data



# np.set_printoptions(precision=5, legacy='1.25', linewidth=120)
np.set_printoptions(precision=5, legacy='1.25', linewidth=999)
debug = config.debug

def calc_exp(inp, cn, expPercent=0.5, debug=0): 
  """
    Calculate the exposure period for a nest (number of days observed)
      For each nest: 
        - known alive days plus estimate of alive days in final interval
        > alive days (before final int) + final int * expPercent
    ---------
    ARGUMENTS
      :param inp: = [ i,j,k] for all nests in set\n
      default expPercent is from Mayfield; Johnson recommended 0.4
    -------
    RETURNS
      ndarray. nrows=len(inp); cols=alive_days, final_int, exposure
    ----------
    MORE INFO
      - the ijk values should tell you failed vs hatched

    For the basic case where psurv is constant across all nests and times:
      1. count the total number of alive days when nest was observed
      2. count the number of days in the final interval (for failed nests)
      3. calculate the exposure
        - days obs before final int + (final int * expPercent)
      *expPercent* = percent of final interval nest is assumed alive
        - Mayfield used 50%, Johnson corrected it to 40%
        - final interval assumed to be 0 days for hatched nests, which
           were found after hatch (exposure of incubation period is over)
        - no nestling exposure bc precocial/semi-precocial chicks
          leave the nest so early 

    *NOTES* added debug function so it only prints outside optimizer
      > I think I couldn't get it to work as vectorized, so I used a loop
    ----------
  """
  # if cn.debugM>=2:
  #   np.save("out/inp.npy", inp)
  expo = np.zeros((len(inp), 3))
  # print(f"\t\t{expo.shape[0]=}")
  # if debug>=2:
  #   print("\t\t\t\tINP:  |> i:", end=" ")
  #   arrPrint(inp[:,0],abbr=False)
  #   print("\t\t\t\t\t\t\t|> j:", end=" ")
  #   arrPrint(inp[:,1],abbr=False)
  #   print("\t\t\t\t\t\t\t|> k:", end=" ")
  #   arrPrint(inp[:,2],abbr=False)
  #   # print("\t\t\t\t|>inp:")
  #   # arrPrint(inp[0:5,:], ind=8)
  #
  # for n in range(len(inp)-1): # want n to be the row NUMBER
  for n in range(len(inp)): # want n to be the row NUMBER
    #+> interval from first found - last active 
    #   +> all nests are KNOWN to be alive
    expo[n,0] = inp[n,1] - inp[n,0]
    # expo[n,0] = expo[n,0] - 1 # since this is essentially 1-day intervals, 
                  # need 1 fewer than total number? no?
    #+>interval from last active - last checked
    expo[n,1] = inp[n,2] - inp[n,1]
    # expo[n,1] = expo[n,1] - 1
    # +>if expo[n,1]!=0: expo[n,1] = expo[n,1]- 1 ; for hatched nests, stays 0

    # +>exposure = sum(alive days) + days in final int * expPercent
    expo[n,2]   = expo[n,0] + (expo[n,1]*expPercent)
    # NOTE need nests to be alive for at least one interval
  # if debug>=2: 
  #   print("\t\t\t\tEXPO:  |> aliv:", end=" ")
  #   arrPrint(expo[:,0],abbr=False)
  #   print("\t\t\t\t\t\t\t|> final int:", end=" ")
  #   arrPrint(expo[:,1],abbr=False)
  #   print("\t\t\t\t\t\t\t|> exposure:", end=" ")
  #   arrPrint(expo[:,2],abbr=False)
  #   print("\t\t\t\tEXPO: |> alive days:", expo[:,0].T)
  #   print("\t\t\t\t\t\t\t|> final int:", expo[:,1].T)
  #   print("\t\t\t\t\t\t\t|> exposure:", expo[:,2].T)
  #   print("\t\t\t\t|>expo:")
  #   arrPrint(expo[0:5,:], ind=8)
  # if cn.debugM>=2:
  #   np.save("out/exposure.npy", expo)
  return(expo)
#-----------------------------------------------------------------------------

def mayfield(num_fail, expo):
  """ 
    The Mayfield estimator of DSR 
    
    Mayfield's original estimator was defined as: 
        > DSR = 1 - (# failed nests / # exposure days)
    so if DSR = 1 - daily mortality, then:
        > daily mortality = # failed nests / # exposure days
    
    Arguments:
      num_fail = count of failed nests (total-hatched)
      expo   = sum of exposure days output of calc_exp() (out[:,2])

    Returns: the daily mortality 
    Note: I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
    expo needs to be a SUM
  """
  # print(expo, type(expo))
  # mayf = num_fail / (expo.sum())
  mayf = num_fail / (expo) # expo is already the sum?
  # print(f"\t\t> mayfield DSR = ({num_fail=}) / ({expo=}) = {mayf}")
  # print(f"\t\t> mayfield DSR = ({num_fail=}) / ({expo=}) =", end=" ")
  # if cn.debugM: print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

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
    ---------------------------------------------------------------------------
  """
  print("calculate Johnson estimator")
  # jEst = (1/srn) * sum()

# -----------------------------------------------------------------------------

# @profile

def calc_dsr(nData, nestType, calcType, conf, incTime=0, psurv=0, debug=0):
  """ 
    Calculate exposure and DSR for a given set of nests. 
    ------

    - calculate daily mortality rate using Mayfield:
      if calc type == 'true':
        'exposure' = end - init
         may cause bias in 'true' val

      else:
        calculate with actual exposure days
        pass i,j,k from ndata to calc_exp()

    - pass exp and num_fail to mayfield()
    
    -----
    Returns DSR value (1-DMR). 
    -----
    NOTES
    Use debug argument so it only prints when not optimizing
    calc_exp doesn't work for all nests since many weren't discovered
  """

  nNests  = len(nData)
  if calcType=="mayfield":
    expDays = calc_exp(nData[:,4:7], expPercent=0.4, cn=conf, debug=debug)
  # hatched = len(nData[:,3] == 0)
  # failed  = nNests-hatched
  
  # expDays = exposure(nestData[:,6:9], numNests=numN, expPercent=0.4)
  # expDays = calc_exp(nData[:,6:9], expPercent=0.4)
  # if nestType=="all":
  ## +> calc type 'true' means ???
  # if debug>=2: print(f"{nestType=} | {calcType=} |> ", end="   ")
  # if debug>=2:
    # print(f"\n\t\t\t|>{nestType=}|{calcType=}|{incTime=}|{psurv=}|>", end=" ")
  if calcType=="apparent":
    # if debug>=3: 
    #   print("\t> exposure days calc type = 'true'")
    #   print("\t> calculating exposure days from all nests")

    allDays = sum((nData[:,2]-nData[:,1]))
    avgExp  = allDays/nNests
    hatched = sum(nData[:,3] == 0)
    apparent = hatched/nNests
    appDSR   = 1-((nNests-hatched)/allDays)

    # if debug>=2:
      # print(
          # f"\t\tapp. DSR (1-(true_num_fail/total_days)):{appDSR:.3f} "
          # f"\t\t1 - (({nNests}-{hatched}) / {allDays}) = {appDSR:.3f} "
          # # f"| expected DSR: {psurv}"
          # )
    # if debug>=3: 
      # print(
            # f"\n\t\t\tapparent nest success s (hatched/total): {apparent:.3f} "
            # f"| expected PSR: {psurv ** incTime:.3f}"
            # f"| nrows of nest data: {nData.shape[0]}"
            # )
      # for n in range(5):
        # print(f"\t\t\t\texposure days: {nData[n,2]} - {nData[n,1]}")
    return(appDSR)
  else:
    # expDays = calc_exp(nData[:,4:7], expPercent=0.4, cn=conf, debug=0)
    expDays = expDays[:,2].sum()
    hatched = sum(nData[:,7] == 0)
    dmr   = mayfield(num_fail=nNests-hatched, expo=expDays)

    ## +> now mayfield function prints instead..
    # if debug>=2:
    #   print(
          # f"\t{expDays=:.3f}|{nNests-hatched=}|"
          # f"{dmr=:.3f}|{1-dmr=:.3f} "
          # # f"| expected DSR={psurv}"
          # )
    return(1-dmr)


# if debug: 
  ### duhhh, turn on if-else statements by defining a deparate dbug version,
  # BUT still has same issue as separate debug file (need to keep both updated)

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
    ----
    NOTE
      
      **doesn't work** for obs_int=1

      The model used in Program MARK is based on Dinsmore (2002) -  
         allows for variance in DSR & use of covariates

      These functions are based on info in 'Program MARK: A Gentle Introduction' 

  """

  # prob, dof = probs
  # allp, alldof = mark_probs(s=s, ndata=ndata)
  # ALL IN ONE FUNCTION:
  s    = s.item() # EX makes singleton array into scalar
  # print(f"\t\t{s=}")
  # allp   = np.array(range(1,len(ndata)), dtype=np.longdouble) # all nest probabilities 
  allp   = np.array(range(0,len(ndata)), dtype=np.longdouble) # all nest probabilities 
  expo = calc_exp(inp=ndata[:,4:7], expPercent=0.4, cn=con)
  # print(f"{expo.shape[0]=} | {allp.shape[0]=}")
  for n in range(len(ndata)-1): # want n to be the row NUMBER
    # alive_days = expo[n,0] - 1
    # final_int  = expo[n,1] - 1
    alive_days = expo[n,0] 
    final_int  = expo[n,1]
  
    ##+> don't know why these equaations don't work when final_int = 1
    if final_int > 0: # final int for hatched nests == 0
      p   = (s**alive_days)*(1-(s**final_int)) 
    else:
      p   = s**alive_days
    allp[n]   = p # NOTE this line is throwing the Deprecation Warning
  nll = sum(-np.log(allp)) # +> sum of log = log of product & maybe faster
  # nll = -np.log(np.prod(allp))
  # NOTE these if statements take up lots of time, esp inside the optimizer

  #~----------------------------------------------------------------------------
  # if con.debugM>=2:
  #   s_arr = np.full(ndata.shape[0], s) ##+> length & fill value
  #   id_arr = ndata[:,0]
  #   fate_arr = ndata[:,7]
  #   mark_out = np.column_stack((allp, expo, s_arr, id_arr, fate_arr))
  #   np.save("out/MARK_print.npy", mark_out)
  return(nll)

# -----------------------------------------------------------------------------

def mark_wrapper(srn, ndata, nocc, conf):
  """
    This function calls the program MARK function when given a random starting 
    value (srn), number of occasions (nocc), and some nest data (ndata)

      > values given to optimizer are transformed then passed to MARK function
        > allows larger range of values for optimizer to work over w/o overflow
        > but values given to the function are still between 0 and 1, as required

      > Create vector to store the log-transformed values, then fill

    ---------
    RETURNS:
      output from prog_mark()

    -------
    NOTES:
      the logistic function tends to overflow if it's a normal float; make it np.float128

  """
  s = logistic(srn)
  #@#print("logistic of random starting value for program MARK:", s, s.dtype)
  ret = prog_mark(s, ndata, nocc, con=conf)
  #@#print("ret=", ret)
  return ret



def time_model():
  """
  """

def make_logexp_df(nData):
  afate = nData[:,7]
  # date  = nData[:,2]
  ## +> center date to reduce multicollinearity
  # date  = nData[:,6] # +> k value
  date  = nData[:,4] # +> i value
  mndate = np.mean(date)
  date = date - mndate

  week = np.floor(date/7)
  # mnweek = np.mean(week)

  age  = nData[:,6] - nData[:,1] 
  # print(f"{week=}")
  # print("centered dates:\n", date)
  expo = calc_exp(nData[:,4:7], cn=config, debug=2)
  expDF = pd.DataFrame({'exposure': expo[:,2],
                        'afate': afate,
                        'date': date,
                        'week': week
                        })
  # print(expDF.corr())
  # print(expDF['date'][expDF['survive']==0])

  expDF['survive'] =  [0 if x in [1,2] else 1 for x in expDF['afate']]
  expDF['log_expo'] = np.log(expDF['exposure'])

  # TODO: fix something weird with dfPrint output? just shows NA

    # print("survive column:")
    # arrPrint(expDF['survive'], abbr=False)
    # print("log exposure:")
    # arrPrint(expDF['log_expo'], abbr=False)
    # print("dates where survive==0 & survive==1:")
    # print(expDF.loc[expDF['survive']==0, ['date']])
    # print(expDF.loc[expDF['survive']==1, ['date']])
  return(expDF)

# def log_exp_r(nData, output='predict'):
# def log_exp(expo, afate, date):
def log_exp(nData,par,nSurvey, fname="", modSave=False, predSave=False):
  """
    The logistic exposure model for an intercept-only model &
    a model with effect of end date & a quadratic end date model
    -----
    ARGS: 
      exposure days, assigned fate,
        & date of fate assignment for each nest
    -----
    RETURNS:
      
    -----
    NOTES:
      Uses clog-log link function, and then adds an offset
      to account for exposure

      Date is centered
  """
  expDF = make_logexp_df(nData)

  nNest = nData.shape[0]
  #---BUILD TRUE NEST HISTORY:
  initList = nData[:,1].astype(int) #+> true init date
  endList = nData[:,2].astype(int)  #+> true end date
  #+> convert fate to 0 = failed, 1 = hatched
  fateList = np.array([0 if i in [1,2] else 1 for i in nData[:,3]])
  trueHist = build_dsr_mat_old(nNest=nData.shape[0],
                           init=initList,
                           end=endList,
                           fate=fateList,
                           nDays=par.brDays)

  #---BUILD NEST OBSERVATION HISTORY:
  initList = nData[:,4].astype(int) #+> discovery date
  endList = nData[:,5].astype(int)  #+> date when last active
  #+> convert fate to 0 = failed, 1 = hatched
  fateList = np.array([0 if i in [1,2] else 1 for i in nData[:,7]])
  expo = calc_daily_expo(nNest, )
  obsHist = build_dsr_mat(nNest=nData.shape[0],
                           init=initList,
                           end=endList,
                           fate=fateList,
                           # nDays=len(svydays))
                           nDays=nSurvey)

  fam = sm.families.Binomial(link=sm.families.links.CLogLog())
  # modForm = ["survive ~ 1","survive ~ date", "survive ~ date * nstorm"]
  # modForm = ["I(1-survive) ~ 1",
  #            "I(1-survive) ~ date",
  #            "I(1-survive) ~ date + I(date**2)",
  modForm = ["I(survive) ~ 1",
             "I(survive) ~ date",
             # "I(survive) ~ week",
             "I(survive) ~ date + I(date**2)",
             # "I(survive) ~ date - I(date**2)",
             ]
  out = predict_vals(modForm, fam, expDF) #
  coef, pred = out
  if modSave:
    np.save(fname,coef)
  return out

# def build_dsr_mat(nData, par):
def build_dsr_mat_old(nNest,init, end,fate, nDays ):
  """
  """
  # +> create matrix with nNests rows and nDays columns
  # mat = np.empty((par.numNests, par.hatchTime))
  # mat = np.empty((par.numNests, par.brDays))
  #+> number analyzed nests
  # nNest = nData.shape[0]
  # mat = np.empty((nNest, par.brDays))
  mat = np.empty((nNest, nDays))
  mat.fill(np.nan)
  # mat[:, ]
  init2d = init[:,np.newaxis]
  end2d = end[:,np.newaxis]
  fate2d = fate[:,np.newaxis]
  # print(f"{nNest=} ; {len(init)=} ; {len(end)=} ; {len(fate)=}")
  print(
      f"{nNest=} ; {init2d.shape=} ; {end2d.shape=}"
      f" ; {fate2d.shape=} ; {mat.shape=}"
      )
  print(f"{fate=}")
  #+> make true nest history:
  #+> for each row in mat, at col value (axis 1) matching init, put "1"
  np.put_along_axis(mat, init2d, 1, axis=1)
  np.put_along_axis(mat, end2d, fate2d, axis=1)
  # mat[init:end-1] = 1
  for n in range(nNest):
    initInd = int(init[n])
    endInd  = int(end[n])
    print(f"{initInd=} ; {endInd=}")
    # mat[:,initInd:endInd-1] = 1 # NOTE this modifies all rows in array
    mat[n,initInd:endInd] = 1
  print(f"{mat=}")



def calc_daily_expo(numNests, surveyInts, surveyDays, init, end):
  """
    RETURNS:
      a list of lists of exposure days for each interval nest was observed
  """
  svyInd = svy_position(init, end, surveyDays)
  initPos, endPos = svyInd
  expo = []
  for n in numNests:
    expo.append(svyInd[initPos:endPos])

  print(expo)


# def build_dsr_mat(nNest, nObs, init, end, expo, fate, nDays, covars=True):
def build_dsr_mat(nestObs, nDays, expos, covars=True):
  """
    Build a nest survival matrix with rows for each obs of each nest
    ----
    ARGS:
      nestObs = id, i, j, k, fate, nObs
      expos = list of list of exposure intervals
  """
  if covars:
    cols = ["id", "surv", "expo", "covar1", "covar2"]
  else:
    cols = ["id", "surv", "expo"]

  nestID, ff, la, lc, fate, nObs = nestObs
  nNest = nestObs.shape[0]

  nrows = np.sum(nObs)
  endDay = np.cumsum(nObs) -1 #+> zero-indexed
  # mat = np.empty((nNest*np.sum(nObs), len(cols) ))
  mat = np.ones((np.sum(nObs), len(cols) ))
  # mat[:,0] = [np.repeat(i, nObs[i]) for i in range(nNest)]
  mat[:,0] = np.repeat(range(nNest), nObs) #+> repeat ID nObs times
  mat[:,1][endDay] = fate
  mat[:,2] =  

def predict_daily(modForm, fam, expDF, ndays=180):
  """
  """
  out = {} # NOTE change this to list?
  pred = []
  dateVal   = np.arange(1,ndays+1,1) # dateVal   = np.arange(1,181,1)
  nNests = expDF.shape[0]
  like = []
  for f in modForm:
    for n in nNests:

      like.append()


def predict_vals(modForm, fam, expDF,ndays=180):
  """
    Returns 
      A list of 2 lists:
        1. model coefs/cl/pvals 
        2. predicted vals
  """
  out = {} # NOTE change this to list?
  pred = []
  dateVal   = np.arange(1,ndays+1,1) # dateVal   = np.arange(1,181,1)
  for f in modForm:
    #+> this seems to work ok, but try a custom link function?
    mod = smf.glm(f, data=expDF, family=fam, offset=expDF['log_expo'])
    fit = mod.fit(method='nm', maxiter=50000,maxfun=5000)
    # out[f] = fit # +>this will output the model fit directly
    # print("\tMODEL RESULTS:")
    # print(fit.summary()) 
    # +> choose how to make the predictions:
    # if f=='I(1-survive) ~ 1':
    #   # preee = fit.predict(expDF).to_numpy() # +> produces a pd.Series; convert to np
    #   # print(f"{preee=}")
    #   # pre = fit.get_prediction(expDF).to_numpy() # +> produces a pd.Series; convert to np
    #   pre = fit.get_prediction(expDF)
    #   mean = pre.predicted_mean
    #   print(f"{mean=}")
    #   conf = pre.conf_int()
    #   print(f"{conf=}")
    #   # print(f"{pre=}")
    #   pred.append(pre)
    # else:
    # +> new data for making predictions:
    newDat    = pd.DataFrame({'date': dateVal}) # print("new data: ") print(newDat)
    # NOTE predict generates points; get_predict also has CIs
    # preee = fit.predict(newDat).to_numpy()
    # print(f"{preee=}")
    # pre = fit.get_prediction(newDat).to_numpy()
    pre = fit.get_prediction(newDat)
    mean = pre.predicted_mean
    # print(f"{mean=}")
    conf = pre.conf_int()
    # print(f"{conf=}")
    lcl = conf[:,0] #+> indexing w/1 val returns 1d arr # print("lcl:", lcl, lcl.shape)
    ucl = conf[:,1]
    pr = np.column_stack((mean,lcl,ucl))
    # print(f"{pr=}")
    # nd = datagrid(newDat)
    # pred = predictions(fit, )
    # print(f"{pred=}")
    pred.append(pr)
    # print(f"{pred=}")

    coef = fit.params.values # print(coef, coef.shape,type(coef))
    confint = fit.conf_int(alpha=0.05).values # print(confint, confint.shape,type(confint))
    pval = fit.pvalues.values # print(pval, pval.shape)
    lcl = confint[:,0] #+> indexing w/1 val returns 1d arr # print("lcl:", lcl, lcl.shape)
    ucl = confint[:,1]
    # stack = np.column_stack((coef,lcl,ucl,pval)) # print(stack, stack.shape)
    # out[f] = np.concatenate((coef,lcl,ucl,pval)) 
    out[f] = np.column_stack((coef,lcl,ucl,pval))

  # print("predictions:\n",pred, type(pred))
  # #NOTE explicitly call the columns here:
  # pred2 = np.column_stack((dateVal,pred[0],pred[1],pred[2]))
  # NOTE: add the date column later, after averaging
  # pred2 = np.column_stack((pred[0],pred[1],pred[2]))
  # pred2 = np.vstack(pred)
  pred2 = np.column_stack(pred)
  
  # pred2 = np.column_stack((pred[0],pred[1]))
  # print("|> predictions:")
  # print(pred2[:5,:])
  # print(". . . . . . . . . . . ")
  # print(pred2[-5:,:])
  # coefs = np.concatenate(list(out.values()))
  coefs = np.vstack(list(out.values()))
  # print(">> coefs:\n", coefs, type(coefs), coefs.shape)
  # return [out, pred2] #
  return [coefs, pred2] #
  # return out #+> returns a 1D numpy array
      
      # out[f] = [fit.params.values, fit.conf_int(alpha=0.05).values]
      # out[f] = np.concatenate(coef,confint[0,:])
      # out[f] = np.concatenate((coef,lcl,pval)) 
      # survVal = np.linspace(0.9,0.99,9)
      # sfrqVal = np.array([1,2,3,4,5])
      # sdurVal = np.array([1,2,3])
      # pflVal  = np.linspace(0.6,0.95,7)
      # combo   = list(itertools.product(survVal, sfrqVal, sdurVal, pflVal))
      # dateVal   = np.linspace(0,180,181)
      # if f=='survive ~ 1':
    # elif output == 'predict':
    # out[f] = fit.predict(newDat)

  # print(out)
  # print(type(pred[1]))
  # print("dates:\n", dateVal, type(dateVal))
  
  # pred2 = np.vstack(pred)
  # pred2 = np.column_stack((dateVal,pred))
  # NOTE instead of saving to file, return & then save average to file
  # np.save(outf, pred2)
  # if output == 'coef':
    # out = np.concatenate(list(out.values()))

        # logexArr = np.concatenate(list(logex.values()))

  # ret = [out[1].params]
    # TODO save param vals?

  
  # expMod = [sm.GLM.from_formula(f, data=expDF, family=fam) for f in modForm]
  # TODO decide on output

def logit(x):
  odds = x / (1-x)
  logodds = np.log(odds)
  return logodds  

def predict_vals_r(modForm, fam, expDF,ndays=180):
  """
    Fit the models using R functions and plot the
    predictions from the R model objects so you
    get appropriate confidence intervals, etc
  """
  
def plot_predict(pred):
  """
  """

def logexp_eq(exp):
  """
  """
  


#-----------------------------------------------------------------------------
#   THE LIKELIHOOD FUNCTION
# -----------------------------------------------------------------------------
#def like_old(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData, stormDays, surveyDays, obs_int):
# -----------------------------------------------------------------------------
#   PROGRAM MARK 
# -----------------------------------------------------------------------------

# It also has a wrapper function that transforms the initial optimizer values
# using the logistic function.
# This way, the optimizer can work over the range of -infinity:infinity, but
# the values fed to the function are between 0 and 1 (probabilities)

# Lastly, it has a function to generate the probabilities before running the optimizer 
# on the MARK function, so I can take the for loop out of the function that is optimized.

# -----------------------------------------------------------------------------
# def prog_mark(s, ndata, probs, nocc, con=config):
# @profile


