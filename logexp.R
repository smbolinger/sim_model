
library(MASS)
library(dplyr) # load dplyr last so as not to mask select?
library(brglm2)
library(reticulate)
source("lexp_fun.R")

# fileList <- c("observer.py", )
# source_python()
# arg <- commandArgs(trailingOnly=TRUE)
arg <- unlist(strsplit(commandArgs(trailingOnly=TRUE), split=" "))
if(length(arg)==0){
  message("** NO atype ARGUMENT PROVIDED; USING DEFAULT")
  atype=""
} else {
  for(a in arg){
    cat("arg =",a)
    if(grepl("at\\w+", a)) atype <- stringr::str_extract(a, "(?<=at)\\w+") # extract words after "at"
    # if(a=="test") params$test <- TRUE
  }
}
# atype <- "full"

## doesn't work to pass vars to python:
# Sys.setenv(atypeR="norm", script_name="logexp.R")
Sys.setenv(atypeR=atype, script_name="logexp.R")
py_run_file("rsettings.py")
## maybe just running this ^ starts the repl in python?
## after running it, loading rsettings below still loads the correct values
## but, it doesn't seem to load anything into the R environment, so still need to do that below?

obs <- import("observer")
dsr <- import("dsrCalc")
nest <- import("makeNests")
funs <- import("helpers")
# sett <- import("settings", )
sett <- import("rsettings", convert=FALSE)
# sett <- import("rsettings") # refactored into functions - would need to change other scripts, too
mod <- import("datsim")
# mod <- import("sim_func")
logex <- import("log_exposure")
# but the OTHER scripts are still importing from settings.py, not rsettings.py
# so regardless of what happens here, they will load the default config & param lists
# atype_r = "norm"
# config = sett$choose_config(atype_r)
# pLists = sett$choose_parlist(atype_r, config)
# odir  = funs$mk_outdir(now_short, con=config)
# paramsArray = funs$mk_param_list_list(parList=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False)
# pArrList = funs$mk_param_list_list(parList=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False, listRet=True)
# rng = np.random.default_rng(seed=config.rngSeed)
# config <- config::get() # shouldn't import here - could have changed config in settings
config <- py_to_r(sett$config) # I DO want to convert config, but not the param dicts
# config <- py_to_r(sett$choose_config(atype)) # I DO want to convert config, but not the param dicts
# # now <- sett$now_short
nreps <- config$nreps
debug <- config$debug
cat("CONFIG:", paste(config,collapse=";"), "\n")

# print(class(nreps))

# odir <- funs$mk_outdir(now)
odir <- sett$odir
homedir <- "/home/wodehouse/Projects/sim_model"
dirName <- sprintf("%s%s", config$rngSeed, atype)
outdir <- file.path(odir,dirName)
if(!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)
# coef_fname <- sprintf("%s/out/%s/coefs_r_%s%s.rds", homedir, sett$now_short, config$rngSeed, atype)
# cat("\noutput file: ", coef_fname)
suff <- sprintf("%s%s", config$rngSeed, atype)
# rng = np.random.default_rng(seed=config.rngSeed)
# paramsArray <- reticulate::dict(sett$paramsArray)
# staticPar <- reticulate::dict(sett$staticPar) # this one IS a list
paramsArray <- sett$paramsArray
staticPar <- sett$staticPar # this one IS a list
pArrList <- sett$pArrList ## should be able to use in R?
# +> automatic type conversion is NOT working for dicts. and neither is explicit conversion.
print(length(paramsArray))
print(length(pArrList))
# print(class(staticPar$brDays))
# print(staticPar$brDays)
parr <- print(py_to_r(paramsArray))
preDays <- parr$brDays

# paramsArray <- funs$mk_param_list_list(parList=sett$pLists, fdir=odir,suf=suff)
# print(paramsArray)
parID = 0
nmod <- 3
# print(config$nreps)
# cat("nreps = ", config$nreps)
# coef_list <- c("_est", "_lcl", "_ucl")
# preDays     <- as.numeric(staticPar$brDays)
# preDays     <- py_to_r(staticPar$brDays)
cat("\ndays for prediction:")
print(preDays)
dims <- c(preDays,3,length(pArrList),nmod)
print(dims)
dnames <- list(seq(preDays),seq(3), seq(length(pArrList)), seq(nmod))
print(dnames)
pred <- array(NA,
               dim=c(preDays, 3, length(pArrList),nmod),
               dimnames=list(seq(preDays),seq(3), seq(length(pArrList)), seq(nmod)) )
if(debug) print(dimnames(pred))
coef_list <- c("est", "lcl", "ucl")
# coef_names <- paste0(c("dot", "date", "datesq"), coef_list)
mod_names <- c("_dot", "_date_int", "_date_b1", "_datesq_int","_datesq_b1","_datesq_b2")
# coef_names <- do.call(paste0, expand.grid(mod_names, coef_list))
coef_names <- do.call(paste0, expand.grid(coef_list, mod_names))
# print(coef_names)
coefs <- array(NA,
               dim=c(length(coef_names), nreps, length(pArrList)),
               dimnames=list(coef_names,seq(nreps), seq(length(pArrList))))
if(debug) print(dim(coefs))
if(debug) print(dimnames(coefs))

# for(i in 1:nrow(pArrList)){
# for(i in 0:length(pArrList)-1){
## should stay 1-indexed in R code and then subtract one when giving to python function/object
# for(i in 1:length(pArrList)){
for(i in seq(length(pArrList))){
  if(debug) cat("\ni=",i)
  # print(i)

  # print(sett$staticPar)
  ## I think paramsArray is 0-indexed since it's a python object, but surveys (below) is 1-indexed
  # if(debug) print(class(paramsArray[i-1]))
  # if(debug) print(class(staticPar))
  # par <- funs$mk_param_list(paramsArray[i], sett$staticPar)
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  print(par)
  if(debug) print(par$stormFrq)
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur)
  survey    <- withCallingHandlers(
                                   {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=config)},
                                   error=function(e){ 
                                     reticulate::py_last_error() 
                                     # print(sys.calls()) # doesn't help if error in python
                                   }  )

  # print(config$nreps)
  # for(r in 0:config$nreps){
  repID=0
  #------------------------------------------------------------------------------------------------------------------
  predictions <- array(NA, 
                       dim=c(3, nreps,preDays,nmod),
                       # dimnames=list(c("term","est", "lcl","ucl"), seq(nreps), seq(preDays))
                       dimnames=list(c("est", "lcl","ucl"), seq(nreps), seq(preDays), seq(nmod))
  )
  if(debug) print(dimnames(predictions))
  #------------------------------------------------------------------------------------------------------------------
  for(r in seq(nreps)){
    cat("\nr=",r)
    skiptoNext <- FALSE

    # nestData1 <- tryCatch({
    ## the nest data are fine, but suddenly pandas is having an issue???
    nestData1 <- withCallingHandlers({
      # obs$make_obs(par,stormDays,survey,config,sett$nWeeks,sett$initFromFile,pandas=TRUE)
      obs$make_obs(par,stormDays,survey,config,sett$nWeeks,sett$initFromFile,pandas=FALSE)
      # obs$make_obs(par,stormDays,survey,config,nWeeks,initFromFile,pandas=TRUE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate.")
      print(sys.calls())
      reticulate::py_last_error()
    })

    # print(skiptoNext)
    if(skiptoNext) {
      next
    # } else {
    #   cat("no issue")
    }

    #---------------------------
    # names(nestData1)
    # cat("\nall nest data:\n")
    # print(class(nestData1))
    # print(nestData1)
    #
    # this function may need to be edited to work with pd.DataFrame, but everything else should be easier:
    # flooded,hatched,discover.sum(),exclude.sum(),unknown.sum(),
    # misclass.sum(), avgFInt, avgK, appDSR, mark_s, repID, parID])
    nVal <- mod$calc_nests(nestData1, par, repID, parID)
    cat("\nN VAL (flood,\thatch,\tdiscover,\texclude,\tunknown,\tmisclass,\navg final int,\tavg K,\tapp DSR,\tMARK DSR,\trepID,\tparID:\n)")
    print(nVal)

    # excl <- nVal
    # nestData <- subset(nestData1, )
    # nestData <- nestData1[nestData1[[11]]!=0] # remove undiscovered nests
    # nestData <- nestData1[nestData1$totobs!=0] # remove undiscovered nests
    colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
  
    # nestData <- nestData1 |> as.data.frame(col.names=colnames) |> filter(totobs!=0) # remove undiscovered nests
    nestData <- nestData1 |> as.data.frame() |> setNames(colnames) |> filter(totobs!=0) # remove undiscovered nests
    print(nestData)
    # cat("\ndiscovered nests:\n")
    # print(nestData)
    # nestData <- nestData[nestData[[8]]!=7] # remove unknown fates
    # nestData <- nestData[nestData$afate!=7] # remove unknown fates
    nestData <- nestData |> filter(afate!=7) # remove undiscovered nests
    if(debug) cat("\nanalyzed nests:\n")
    if(debug) print(nestData)
    nNest <- nrow(nestData)
    # cat("\nnumber of nests:", nNest)
    nestObs <- nestData |> dplyr::select(ID, i, j, k, afate, totobs)
    # print(head(nestObs))

    expoList   <- logex$calc_daily_expo(numNests=nNest,
                                     surveyDays=survey[[1]],
                                     surveyInts=survey[[2]],
                                     firstDay=nestData$i,
                                     lastDay=nestData$k,
                                     db=config$debugNests)

    dat2S <- logex$make_daily_logex_df(nestObs,
                                        expos=expoList[[1]],
                                        covar1=expoList[[2]], # all survey dates for all nests
                                        db=config$debugNests)
    # cat("\n|> made obs data\n")
    # print(obsDat)

    # cat("\nnot in the function")
    # nmod=3
    # modOut <- list()
    # # int_glm  <- glm(Surv~1,
    # cat(".1")
    # modOut[[1]]  <- glm(Surv~1,
    #                family=binomial(link=logexp(dat2S$Exposure)),
    #                data=dat2S,start=c(1))
    # if (debug) print(summary(modOut[[1]]))
    # if (debug) print(coef(modOut[[1]]))
    # # if (debug) print(cbind(coef(modOut[[1]]), confint.default(modOut[1])))
    #
    #
    # cat(".2")
    # modOut[[2]]<- glm(Surv~Date,
    #                family=binomial(link=logexp(dat2S$Exposure)),
    #                data=dat2S,start=c(1,0))
    # if (debug) print(summary(modOut[[2]]))
    # if (debug) print(coef(modOut[[2]]))
    # # if (debug) print(cbind(coef(modOut[[2]]), confint.default(modOut[2])))
    #
    # cat(".3")
    # modOut[[3]]<- glm(Surv~Date + I(Date^2),
    #                family=binomial(link=logexp(dat2S$Exposure)),
    #                data=dat2S,start=c(1,0,0))
    # if (debug) print(summary(modOut[[3]]))
    # if (debug) print(coef(modOut[[3]]))
    # # if (debug) print(cbind(coef(modOut[[3]]), confint.default(modOut[3])))
    #
    # # coefs[,parID,repID] = sapply(seq(nmod), function(x){
    #
    # # coefs[,r,i] = sapply(modOut, function(x){
    # if (debug) cat("\n|> CREATE COEFS ARRAY \n")
    # ## I could swear this was outputting a vector before; not sure what changed..

    cat("\nin the function")
    mList <- c("Surv~1", "Surv~Date", "Surv~Date+I(Date^2)")
    # mList <- list(Surv~1, Surv~Date, Surv~Date+I(Date^2))
    excpt <- FALSE
    tryCatch({ modOut <- fit_glm(mList,dat=dat2S,debug=T) },
      error = function(e) { 
        message("!! error in glm:", e, "go to next") 
        excpt <<- TRUE
        coefsArray <- rep(-999, length(coef_names))
        coefs[,r,i] <- coefsArray
      },
      warning = function(w) { 
        message("!! warning in glm:", w, "go to next") 
        excpt <<- TRUE
        coefsArray <- rep(-999, length(coef_names))
        coefs[,r,i] <- coefsArray
      })

    if(excpt) {next}

    coefsArray = sapply(modOut, function(x){
                          # print(x)
                                  ## confint for intercept model is not a 2x2 array...annoying
                                  # sapply(seq(length(coef)), function(y){ c(coef(x)[y], confint(x)[y,])})
                           ## for each model, extract coefs & confints
                                  # coeff <- coef(x)
                                  # ## when you assign something like this R just returns it instead??
                                  # conf <- confint.default(x)
                                  #
                                  # print(coeff)
                                  # print(conf)
                          # print(coef(x))
                                  sapply(seq_along(coef(x)), function(y){
                                           ## R STILL trying to return conf instead of coef_arr?
                                          # conf <- confint.default(x)
                                           # print(y)
                                           if (is.matrix(confint.default(x))){ 
                                           # if (is.matrix(conf)){ 
                                             # c(coef(x)[y], confint(x)[y,])
                                             ## use confint.default to avoid long profiling step
                                            # coef_arr <- c(coef(x)[y], conf[y,])
                                            # coef_arr <- c(coef(x)[y], conf[y,])
                                            # coef_arr <- c(coef(x)[y], confint.default(x)[y,])
                                              if (debug) print(c(coef(x)[y], confint.default(x)[y,]))
                                              return(c(coef(x)[y], confint.default(x)[y,]))
                                           } else {
                                           # coef_arr <- c(coef(x)[y], conf[y])
                                           # coef_arr <- c(coef(x)[y], confint.default(x)[y])
                                             if (debug) print(c(coef(x)[y], confint.default(x)[y]))
                                             return(c(coef(x)[y], confint.default(x)[y]))
                                           }
                                           # cat("\ncoef_arr:\n")
                                           # print(coef_arr)
                                           # return(coef_arr)


                                           }
                                   )
                   })
    if (debug) cat("\ncoefs output:\n")
    if (debug) print(coefsArray)
    coefs[,r,i] = unlist(coefsArray)
    # if (debug) print(coefs[,repID,parID])
    if (debug) print(coefs[,r,i])
    # date_glm <- glm(Surv~poly(Date,2),

    #------------------------------------------------------------------------------------------------------------------
    # to get CIs for predictions, choose between ggpredict, ciTools, glm.predict, rockchalk 
    newDat <- data.frame(Date=seq(180))
    ..exposure <- mean(dat2S$Exposure)
    # pred <- predict(modOut[[1]], newdata=newDat, type="response")
    # print(predictions[,r,])
    # print(ggeffects::ggpredict(modOut[[1]])[c("predicted", "conf.low", "conf.high")] )
    # predictions[,r,] <- ggeffects::ggpredict(modOut[[1]])[c("predicted", "conf.low", "conf.high")] 
    # pre <- marginaleffects::predictions(modOut[[1]], newdata=newDat)
    # pr <- pre[,c("Estimate", "2.5 %", "97.5 %")]
    # pr <- marginaleffects::predictions(modOut[[1]], newdata=newDat)[,c("Estimate", "2.5 %", "97.5 %")]
    # print(pre)
    # predictions[,r,] <- marginaleffects::predictions(modOut[[1]], newdata=newDat)[c("term", "estimate", "conf.low", "conf.high")] 
    # predictions[,r,] <- t(pre[c(2,5,6)])
    # print(predictions[,r,])
    for(m in 1:nmod){
      pre <- marginaleffects::predictions(modOut[[m]], newdata=newDat)
      predictions[,r,,m] <- t(pre[c(2,5,6)])
    }
    rm(..exposure)
    # plot(x=seq(180), y=pre$Estimate)
    #------------------------------------------------------------------------------------------------------------------

    repID = repID + 1
  }
  # coefname <- sprintf("%s/out/%s/%s%s/coefs_r_%s.rds", homedir, sett$now_short, config$rngSeed, atype, parID)

  #------------------------------------------------------------------------------------------------------------------
  if(config$predSave=="mean"){
    for (m in 1:nmod){
      pred[,,i,m] <- apply(predictions, c(1,3), mean)
    }
    print(pred[,,i,])
  }
  #------------------------------------------------------------------------------------------------------------------

  if(config$coefSave=="mean"){
    coefname <- sprintf("%s/coefs_r_%03d.rds", outdir, as.numeric(parID))
    print(coefname)
    saveRDS(coefs, coefname)
  }
  parID = parID + 1
}
if (debug) print(coefs)
# coef_fname <- sprintf("%s/out/%s/coefs_r_%s%s.rds", homedir, sett$now_short, config$rngSeed, atype)
# saveRDS(coefs, coef_fname)

