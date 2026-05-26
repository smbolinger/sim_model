
import numpy as np
from getClass import ExpLink
from observer import svy_position
import pandas as pd
import matplotlib.pyplot as plt
from dsrCalc import calc_exp
from print_func import arrPrint, dfPrint
# from settings import config
# from rsettings import config
import itertools
from helpers import centerDat,print
import statsmodels.formula.api as smf
import statsmodels.api as sm
import rpy2
import rpy2.robjects as robj
from rpy2.robjects.packages import importr, data
# from marginaleffects import predictions, datagrid
# from marginaleffects import *

np.set_printoptions(precision=5, legacy='1.25', linewidth=999)
# debug = config.debug
# NOTE previously part of dsrCalc (if you need git history)

# def calc_daily_expo(numNests, surveyInts, surveyDays, firstDay, lastDay, config,db=0):
def calc_daily_expo(numNests,surveyInts,surveyDays,firstDay,lastDay,config):
  """
    Calculate the daily exposure of each nest AND all survey days it's active
    ----
    ARGS:
      
    RETURNS:
      a list containing:
      1. a list of all exposure days across all nests
      2. a list of all survey days across all nests
  """
  # svyInd = svy_position(init, end, surveyDays)
  # print(f"{surveyInts=}")
  db=config.debugLogEx
  svyInd = svy_position(firstDay, lastDay, surveyDays,cn=config)
  initPos, endPos = svyInd
  # print(f"\t\t{initPos=} ; \n\t\t{endPos=}")
  # print(f"\t\t{surveyInts=}")

  # expo = [surveyInts[initPos[n]:endPos[n]] for n in range(numNests)]
  expo = []
  sdays = []
  # if(db>=3): print(f"\n\tSurvey ints that are zero: {surveyInts[surveyInts==0]}")
  for n in range(numNests):
    # NOTE becomes a list of arrays:
    # expo.append(surveyInts[initPos[n]:endPos[n]]) #+> probably slow
    #+> extend flattens the added arrays
    # expo.extend(surveyInts[(initPos[n]+1):endPos[n]+1].tolist()) #+> probably slow
    expo.extend(surveyInts[(initPos[n]):endPos[n]].tolist()) #+> probably slow
    # sdays.extend(surveyDays[(initPos[n]+1):endPos[n]+1].tolist()) #+> probably slow
    sdays.extend(surveyDays[(initPos[n]):endPos[n]].tolist()) #+> probably slow
    
    # expo.append(surveyInts[(initPos[n]+1):endPos[n]+1].tolist()) #+> probably slow

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if db>=3:
  #   # print(f"{numNests=} ")
  #   print(f"\t\t{len(initPos)=} {initPos=} "
  #         f"\n\t\t{len(endPos)=} {endPos=}"
  #         )
  #   print(f"\t\t{len(surveyDays[initPos])=} {surveyDays[initPos]=} "
  #         f"\n\t\t{len(surveyDays[endPos])=} {surveyDays[endPos]=}"
  #         )
  #   print(f"\t{len(expo)=} {expo=} ")
  #   print(f"\t{len(sdays)=} {sdays=} ")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if db>=3:
  #   print(f"\t{surveyInts[initPos+1:endPos+1]=} ")
    # print(f"\t{len(sdays2)=} {sdays2=} ")

  ## These don't work:
  # sdays2 = np.concatenate([surveyDays[initPos:endPos]])
  # if db>=3: print(f"\t{len(sdays2)=} {sdays2=} ")
  # expo2 = np.concatenate([surveyInts[(initPos+1):(endPos+1)]])
  # if db>=3: print(f"\t{len(expo2)=} {expo2=} ")

  # return expo
  return [np.array(expo), sdays]

def make_daily_logex_df(obsData,
                        nObs,
                        # expos,
                        expoList,
                        survey,
                        config,
                        # covar1,
                        # saveDF=False,
                        alldiff=True,
                        zeroInd=True,
                        multiFate=False,
                        pandas=False,

                        exp1=False,
                        # ctr=False,
                        
                        # db=0,
                        ):
  """
    idate = date of initial obs
    leave alldiff = True and get both columns
    exp1 is for calculating true nest survival (exposure=1)
    obsData = ID, init, i, j, k, afate
    nObs = num obs (varies depening on which DSR is being calculated)

    calls make_df 
    ----
    RETURNS:
      pandas dataframe with 7 columns
        [ID, surv, expo, avDate, Date, avAge, Age]
  """
  #NOTE 04-Apr: getting errors about length of column replacements,
  #NOTE   but only for stormFate==True
  db = config.debugLogEx
  # if db>=2: print("\t>-> making daily obs df",end=" ")
  if db>=2: print("\t>-> making daily obs df")
  # cols = ["id", "survive", "exposure", "idate", "date"]
  # cols = ["Nest.ID", "Surv", "Exposure", "ffDate", "Date", "Age", "propInit"]
  cols = ["Nest.ID", "Surv", "Exposure", "avDate", "Date","avAge", "Age"]

  ## CONVERT TO NUMPY ARRAY IF NOT ALREADY:
  if isinstance(obsData, pd.DataFrame):
    # if db>=3: print(f"\n\t\t\t{type(obsData)=}", end=" ")
    # if db>=3: print(f"\t\t{obsData=}")
    obsData = obsData.to_numpy()
  elif not isinstance(obsData, np.ndarray):
    # if db>=3: print(f"\n\t\t\t{type(obsData)=}", end=" ")
    # if db>=3: print(f"\t\t{obsData=}")
    obsData = np.array(obsData)
  # svyDay, svyInt = survey[1:2]
  # svyDays, svyInts,stormSvy = survey
  # if db>=4: print(f"\t\t{svyDay=}\n\t\t{svyInt=}")
    
  # if db>=4:
    # print(f"\t\t\t{type(obsData)=}\n\t\t{obsData=}")
    # print(f"\t\t >>> \t\t{obsData.shape=} {type(obsData)=}")
    # dfPrint(obsData, names=cols)
    # print(f"\t\t{cols=}")
  # nestID, init,ff, la, lc, afate, nObs = nestData.T
  # nestID, init,end,fate,ff, la, lc, afate = nestData.T
  ID, init,end,tfate,ff, la, lc, afate = obsData.T
  
  # if db>=4: print(f"\t\t|>{ID=}")
  nNest = obsData.shape[0]
  # nObs = nObs.astype(int) if exp1 else 
  # if db>=4: print(f"\t\t{np.sum(nObs)=} ; {nObs=}")
  if exp1:
    first,last,fate = init,end,tfate
  else:
    first,last,fate = ff,la,afate
  # if db>=4: print(f"\tpass to make_df:")
  # if db>=4: print(f"\t\t|>{first=}\n\t\t|>{last=}\n\t\t|>{fate=}")
  # nObs[afate!=0] +=1
  # print(f"nrows = {np.sum(nObs)=} ; {nObs=} ")
  # if alldiff:
  #   mat[:,3] = covar1
  # else:
  #   mat[:,3] = np.repeat(covar1, nObs)
  # obsDay    = svyDay[(svyDay>=first)&(svyDay<=last)]
  # expos    = np.diff(obsDay)
  # expos = [expo1, expo2]
  # if db>=4: print(f"\t\t{obsDay=}\n\t\t{expos=}")
  if multiFate==False: # +> make all failures "0"
    # afate = np.array([0 if i in [1,2,7] else 1 for i in afate])
    fate = np.array([0 if i in [1,2,7] else 1 for i in fate])

  # expoList   = calc_daily_expo(numNests=nNest, surveyDays=svyDays,
  #                                  surveyInts=svyInts, firstDay=first,
  #                                  lastDay=last, config=config,)
  # mat = make_df(ID,init,first,last,fate,nObs,nNest,expos,covar1,cols,db=db)
  # mat = make_df(ID,init,first,last,fate,nObs,nNest,expos,obsDay,cols,db=db)
  mat = make_df(ID,init,first,last,fate,nObs,nNest,expoList,cols,db=db)
  # mat = make_df(ID,init,first,last,fate,nObs,nNest,survey,cols,db=db)
  # if db>=2: print(f"\t\t{mat.shape=}")
  dfNew = pd.DataFrame(mat, columns=cols)
  # if db>=4: print(f"\t\tBEFORE: {dfNew.shape=}, AFTER:", end=" ")
  dfNew['log_expo'] = np.log(dfNew['Exposure'])
  cols = ["Nest.ID", "Surv", "Exposure", "avDate", "Date","avAge", "Age", "log_expo"]
  # if db>=3:
    # print(f"\t\t{dfNew.shape=}")
    # dfPrint(dfNew, names=cols)
    # print(f"\t\t\t{dfNew=}")
  # if db>=2: dfPrint(dfNew)

  # NOTE save df in outer function
  # if saveDF:
  #   fn = f"_{parID:04}_{repID:03}.npy"
  #       prOut = Path(odir/f"pred{config.rngSeed}"/prFile)
  #       prOut.parent.mkdir(parents=True, exist_ok=True)

  return dfNew

# def make_df(ID,init, first, last, fate, nObs, nNest, expos, covar, cols, db=0):
# def make_df(ID,init, first, last, fate, nObs, nNest, expos, obsDay, cols, db=0):
def make_df(ID,init, first, last, fate, nObs, nNest, expo, cols, db=0):
# def make_df(ID,init, first, last, fate, nObs, nNest,survey, cols, db=0):
  """
  PURPOSE
    calculate exposure, age, & date covariates for nest data

    for true DSR:
      first=init, last=end, fate=true fate,nObs=total days
    for obs DSR:
      first=i, last=k, fate=assigned fate, nObs=total observations
  """
  # nrows = np.sum(nObs)
  # svyDay, svyInt,stormSvy = survey
  # if db>=4: print(f"\t\t{svyDay=}\n\t\t{svyInt=}")
  nrows    = int(np.sum(nObs))
  initDay  = np.repeat(init, nObs)
  ageStart = first-init 
  ageEnd   = last-init
  avAge    = (ageEnd+ageStart)/2 
  # obsDay    = np.repeat(0,nObs)
  # obsDay    = svyDay[(svyDay>=first)&(svyDay<=last)]
  # expos    = np.diff(obsDay)
  expos,obsDay=expo
  expos,obsDay=expo
  if db>=3: print(f"\t{expos=}{len(expos)}\n\t{obsDay=}{len(obsDay)}")
  #   print(f"\t\t\t{nrows=}{type(nrows)=}")
  #   print(f"\t\t\t{avAge=}")
  #   print(f"\t\t\t{initDay=}")
  # expo1, expo2=expos

  # if zeroInd:
  endDay = np.cumsum(nObs) -1 #+> zero-indexed
  # else:
    # endDay = np.cumsum(nObs) 
  endDay = endDay.astype(int)
  # if db>=3: print(f"\t\t\t{len(endDay)=} {endDay=}")

  mat = np.ones((nrows, len(cols) ))
  # if db>=3: print(f"\t\t{mat.shape=} | {len(expos)=}")
  # mat[:,0] = np.repeat(range(nNest), nObs) #+> repeat ID nObs times
  mat[:,0] = np.repeat(ID, nObs) #+> repeat ID nObs times
  mat[:,1][endDay] = fate ## nest status is 1 unless failed on last check
  mat[:,2] = expos
  # mat[:,2] = expo2
  # mat[:,3] = np.repeat(ff, nObs)
  mat[:,3] = np.repeat((first+last)/2, nObs) # avg observation day
  mat[:,4] = obsDay ## observation day
  mat[:,5] = np.repeat(avAge,nObs)
  mat[:,6] = obsDay - initDay
  # if db>=3: print(f"\t\t\t{mat.shape=} \n\t{mat=}")
  # if db>=4: dfPrint(mat,nprint=20,names=cols)

  return mat


#---------------------------------------------------------------------------------

def make_logexp_df(nData,config):
  afate = nData[:,7]
  ## +> center date to reduce multicollinearity
  # date  = nData[:,6] # +> k value
  date  = nData[:,4] # +> i value
  mndate = np.mean(date)
  date = date - mndate

  week = np.floor(date/7)
  mnweek = np.mean(week)
  week = week - mnweek

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

def log_exp(nData,
            par,
            svy,
            conf,
            alldiff=True,
            typ="daily",
            fname="",
            plname="",
            obsSave=False,
            ctr=False,
            link="clog-log",
            debug=0,
            # modSave=False,
            # predSave=False,
            ):
  """
    The logistic exposure model for an intercept-only model &
    a model with effect of end date & a quadratic end date model
    -----
    ARGS: 
      exposure days, assigned fate,
         & date of fate assignment for each nest
      ctr - center the covariate? default False
    -----
    RETURNS:
      
    -----
    NOTES:
      Uses clog-log link function, and then adds an offset
      to account for exposure

      can also use custom logistic-exposure link 

      
  """
  # +> PERIOD SURVIVAL

  nNest = nData.shape[0]
  svyDays, svyInt, stormSvy = svy
  # nSurvey = len(svyDays)
  ffList = nData[:,4].astype(int)
  lcList = nData[:,6].astype(int)
  #---BUILD TRUE NEST HISTORY:
  # initList = nData[:,1].astype(int) #+> true init date
  # endList = nData[:,2].astype(int)  #+> true end date
  # #+> convert fate to 0 = failed, 1 = hatched
  # fateList = np.array([0 if i in [1,2] else 1 for i in nData[:,3]])
  # trueHist = build_dsr_mat_old(nNest=nData.shape[0],
  #                          init=initList,
  #                          end=endList,
  #                          fate=fateList,
  #                          nDays=par.brDays)

  #+> DAILY SURVIVAL
  #---BUILD NEST OBSERVATION HISTORY:
  #+> convert fate to 0 = failed, 1 = hatched
  if typ == "daily":
    # expo = calc_daily_expo(nNest, svyDays, svyInts, initList, endList  )
    expList = calc_daily_expo(numNests=nNest,
                              surveyDays=svyDays,
                              surveyInts=svyInt,
                              firstDay=ffList,
                              lastDay=lcList,
                              config=conf,
                              )
    expo, date = expList # print(nData[:,np.r_[0,4:8]])
    # obsHist = build_dsr_mat( nestObs=nData[:,np.r_[0,4:9]],
    covar = date if alldiff else nData[:,4]
    covar = centerDat(covar) if ctr else covar
    # obsHist = make_daily_logex_df( nestObs=nData[:,np.r_[0,4:9]],
    # obsHist = make_daily_logex_df( nestObs=nData[:,np.r_[0,4:8,10]],
    obsHist = make_daily_logex_df( obsData=nData[:,np.r_[0:2,4:8]],
                                  nObs = nData[:,10],
                                  expos=expo,
                                  covar1=covar,
                                  # saveDF=conf.obsSave,

                                  ) # print(f"{obsHist=}")
    expDF = obsHist
  else:
    expDF = make_logexp_df(nData,config=conf)
    expo = expDF['exposure'] # for custom link below
  # obsHist = build_dsr_mat(nNest=nData.shape[0],
  #                          init=initList,
  #                          end=endList,
  #                          fate=fateList,
  #                          # nDays=len(svydays))
  #                          nDays=nSurvey)
  #
  if link=="clog-log":
    fam = sm.families.Binomial(link=sm.families.links.CLogLog())
  else:
    # fam = sm.families.Binomial(link=ExpLink(exposure=expo), check_link=False)
    fam = sm.families.Binomial(link=ExpLink(exposure=expo))
  # modForm = ["survive ~ 1","survive ~ date", "survive ~ date * nstorm"]
  modForm = ["survive ~ 1","survive ~ date", "survive ~ date + I(date**2)"]
  # modForm = ["I(1-survive) ~ 1",
  #            "I(1-survive) ~ date",
  #            "I(1-survive) ~ date + I(date**2)",
  # modForm = ["I(survive) ~ 1",
  #            "I(survive) ~ date",
  #            # "I(survive) ~ week",
  #            "I(survive) ~ date + I(date**2)",
  #            # "I(survive) ~ date - I(date**2)",
  #            ]
  out = predict_vals(modForm, fam, expDF,ctr=ctr, debug=debug) #
  coef, pred = out
  if obsSave:
    # np.save(fname, expDF.to_numpy()) # this doesn't save the columns properly
    expDF.to_csv(fname, index=False) # save w/o row names
  # if modSave:
  #   np.save(fname,coef)
  return out

# def calc_true(obsDat):
#   """
#   """
#   out = log_exp()

def plot_predict(modFit, newDat):
  """
    pass the model fit object & the new data points
    then create the prediction line from it
  """
  preee = modFit.predict(newDat).to_numpy()
  # fig = plt.figure()
  fig, ax = plt.subplots(figsize=(5,3))
  # ax.scatter(data=expDF)
  # ax.scatter()
  
  # ax.plot(x1, y, "o", label="Data")
  # ax.plot(x1, y_true, "b-", label="True")
  # ax.plot(np.hstack((x1, x1n)), np.hstack((ypred, ynewpred)), "r", label="OLS prediction")
  # ax.legend(loc="best")


def predict_vals(modForm, fam, expDF,plotfile="",ndays=180,ctr=False,debug=0):
  """
    Returns 
      A list of 2 lists:
        1. model coef/LCL/UCL/pval 
        2. predicted vals
  """
  out = {} # NOTE change this to list?
  pred = []
  dateVal   = np.arange(1,ndays+1,1) # dateVal   = np.arange(1,181,1)
  if ctr: dateVal = centerDat(dateVal)
  # mnDate    = np.mean(dateVal)
  # dateVal   = dateVal - mnDate
  if debug >=2:
    print("INPUT:\n")
    print(expDF.head(30))
  for f in modForm:
    #+> this seems to work ok, but try a custom link function?
    mod = smf.glm(f, data=expDF, family=fam, offset=expDF['log_expo'])

    # +> even with the full obs histories, seems to be estimating PSR, not DSR
    # +> maybe daily doesn't work with clog-log link?
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
    # newDat = sm.add_constant(dateVal)
    # NOTE predict generates points; get_predict also has CIs
    # preee = fit.predict(newDat).to_numpy()
    # print(f"{preee=}")
    # pre = fit.get_prediction(newDat).to_numpy()
    pre = fit.get_prediction(newDat, offset=expDF['log_expo'])
    mean = pre.predicted_mean
    # print(f"{mean=}")
    conf = pre.conf_int()
    # print(f"{conf=}")
    lcl_pr = conf[:,0] #+> indexing w/1 val returns 1d arr # print("lcl:", lcl, lcl.shape)
    ucl_pr = conf[:,1]
    pr = np.column_stack((mean,lcl_pr,ucl_pr))
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
  if debug >= 2:
    print("|> predictions (coef, lcl, ucl):")
    print(pred2[:15,:])
    print(". . . . . . . . . . . ")
    print(pred2[-15:,:])
  # coefs = np.concatenate(list(out.values()))
  coefs = np.vstack(list(out.values()))
  # if plotfile!="":
  #   prepl = plt.figure()
    # ax1   = prepl.add_subplot()
    # pl = expDF.plot(x='date', y='survive', marker="o", label="Data")


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
  
def logexp_eq(exp):
  """
  """
  


def time_model():
  """
  """

def build_dsr_mat(nestData,surveyDays,nDays,true=False,save=False,fname="",db=0 ):
  """
  """
  #NOTE moved from build_dsr_mat_old
  # mat = np.empty((par.numNests, par.hatchTime))
  # mat = np.empty((par.numNests, par.brDays))
  if db>=1:
    # ind =np.arange(len(surveyDays))
    print(
        f"building obs history matrix with {nDays=} ;"
        f" true DSR? {true} (if not, then obs DSR)"
        # f"{[ind:day for ind,day in enumerate(surveyDays)]}"
        f"\nsurvey days:\n"
        f"{np.array2string(np.arange(len(surveyDays))):03}\n"
        # f"{ind:03}\n"
        f"{np.array2string(surveyDays):03}\n"
        )

    # [print( f"{ind}:{day}", end="|") for ind,day in enumerate(surveyDays)]

  #+> number analyzed nests
  nNest = nestData.shape[0]
  # NOTE init & end can either = index w/in all br days or w/in survey days
  if true:
    init = nestData[:,1].astype(int)
    end = nestData[:,2].astype(int)
    fate = nestData[:,3].astype(int)
  else:
    init = nestData[:,4].astype(int)
    end = nestData[:,6].astype(int)
    fate= nestData[:,7].astype(int)
  # mat = np.empty((nNest, par.brDays))
  # +> create matrix with nNests rows and nDays columns
  # +> nDays can be all days in nesting period or all obs days
  mat = np.empty((nNest, nDays))
  mat.fill(np.nan)
  initInd = np.searchsorted(surveyDays, init)+1
  endInd = np.searchsorted(surveyDays, end)+1
  # mat[:, ]
  init2d = initInd[:,np.newaxis]
  end2d = endInd[:,np.newaxis]
  fate2d = fate[:,np.newaxis]
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if db>=3:
  #   print(f"{initInd=}")
  #   print(f"{endInd=}")
  #   print(f"{mat.shape=}")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # print(f"{init2d=}")
  # print(f"{end2d=}")
  # print(f"{nNest=} ; {len(init)=} ; {len(end)=} ; {len(fate)=}")
  # print(
      # f"{nNest=} ; {init2d.shape=} ; {end2d.shape=}"
      # f" ; {fate2d.shape=} ; {mat.shape=}"
      # )
  # print(f"{fate=}")
  #+> make true nest history:
  #+> for each row in mat, at col value (axis 1) matching init, put "1"
  # NOTE this works if you have total breeding days, but not nObs days
  # NOTE could probably make it work and have obs in rows corresponding to nests
  # NOTE but might be easier to do one row per obs per nest? like shaffer
  np.put_along_axis(mat, init2d, 1, axis=1) # set init val in row
  np.put_along_axis(mat, end2d, fate2d, axis=1) # set end val
  # mat[init:end-1] = 1
  for n in range(nNest):                        # fill in between init & end
    initX = int(initInd[n])
    endX  = int(endInd[n])
    # print(f"{initInd=} ; {endInd=}")
    # mat[:,initInd:endInd-1] = 1 # NOTE this modifies all rows in array
    mat[n,initX:endX] = 1
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # with np.printoptions(threshold=10000): # don't truncate
    # if db>=2: print(f"observations {mat=}")
  # if save:
  #   print(f"{mat.shape=}")
  #   mat = np.nan_to_num(mat, nan=-9999.0)
  #   np.savetxt(fname, mat, delimiter=",")
    # np.save(fname, mat)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return mat

def build_dsr_mat_blah(nestObs, expos, covar1=0):
  """
    Build a nest survival matrix with rows for each obs of each nest
    ----
    ARGS:
      nestObs = id, i, j, k, fate, nObs
      expos = flattened list of exposure intervals for all obs for all nests
  """
  # if len(covar2) > 0:
  # cols = ["id", "surv", "expo", "covar1", "covar2"]
  cols = ["id", "surv", "expo", "covar1"]
  # else:
    # cols = ["id", "surv", "expo"]

  nestID, ff, la, lc, fate, nObs = nestObs.T
  nObs = nObs.astype(int)
  # nObs[fate!=0] +=1
  # print(f"nrows: {np.sum(nObs)=} {nObs=}")

  nNest = nestObs.shape[0]

  nrows = np.sum(nObs)
  endDay = np.cumsum(nObs) -1 #+> zero-indexed
  # mat = np.empty((nNest*np.sum(nObs), len(cols) ))
  mat = np.ones((nrows, len(cols) ))
  # mat[:,0] = [np.repeat(i, nObs[i]) for i in range(nNest)]
  mat[:,0] = np.repeat(range(nNest), nObs) #+> repeat ID nObs times
  mat[:,1][endDay] = fate
  mat[:,2] = expos
  # if covar1 != 0:
  mat[:,3] = np.repeat(covar1, nObs)
  # mat[:,4] = np.repeat(covar2, nObs)
  # print(f"{mat=}")

  return mat



def predict_daily(modForm, fam, expDF, ndays=180):
  """
  """
  out = {} # NOTE change this to list?
  pred = []
  dateVal   = np.arange(1,ndays+1,1) # dateVal   = np.arange(1,181,1)
  nNests = expDF.shape[0]
  like = []
  # for f in modForm:
  #   for n in nNests:
  #
  #     like.append()




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


