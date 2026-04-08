
library(reticulate)
library(MASS)
library(dplyr) # load dplyr last so as not to mask select?
source("lexp_fun.R")

# fileList <- c("observer.py", )
# source_python()
# atype <- "test"
Sys.setenv(atypeR="norm", script_name="logexp.R")
py_run_file("rsettings.py")

obs <- import("observer")
dsr <- import("dsrCalc")
nest <- import("makeNests")
funs <- import("helpers")
# sett <- import("settings", )
# sett <- import("rsettings", convert=FALSE)
# sett <- import("rsettings") # refactored into functions - would need to change other scripts, too
mod <- import("datsim")
logex <- import("log_exposure")
# config <- config::get() # could have changed config in settings
# config <- py_to_r(sett$config) # I DO want to convert config, but not the param dicts
# config <- py_to_r(sett$choose_config(atype)) # I DO want to convert config, but not the param dicts
# now <- sett$now_short
nreps <- config$nreps
debug <- config$debug
cat("CONFIG:", paste(config,collapse=";"))

# print(class(nreps))

# odir <- funs$mk_outdir(now)
# odir <- sett$odir
suff <- sprintf("%s%s", config$rngSeed, atype)
# rng = np.random.default_rng(seed=config.rngSeed)
# paramsArray <- reticulate::dict(sett$paramsArray)
# staticPar <- reticulate::dict(sett$staticPar) # this one IS a list
# paramsArray <- sett$paramsArray
# staticPar <- sett$staticPar # this one IS a list
# pArrList <- sett$pArrList ## should be able to use in R?
# +> automatic type conversion is NOT working for dicts. and neither is explicit conversion.
# print(paramsArray)
# print(pArrList)

# paramsArray <- funs$mk_param_list_list(parList=sett$pLists, fdir=odir,suf=suff)
# print(paramsArray)
parID = 0
# print(config$nreps)
# cat("nreps = ", config$nreps)
# coef_list <- c("_est", "_lcl", "_ucl")
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
  if(debug) print(class(paramsArray[i-1]))
  if(debug) print(class(staticPar))
  # par <- funs$mk_param_list(paramsArray[i], sett$staticPar)
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                    reticulate::py_last_error()
                  }
                  )
  print(par)
  if(debug) print(par$stormFrq)
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur)
  survey    <- obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=config)
  # print(config$nreps)
  # for(r in 0:config$nreps){
  repID=0
  for(r in seq(nreps)){
    if (debug) cat("\nr=",r)

    skiptoNext <- FALSE

    nestData1 <- tryCatch({
      # obs$make_obs(par,stormDays,survey,config,sett$nWeeks,sett$initFromFile,pandas=TRUE)
      obs$make_obs(par,stormDays,survey,config,nWeeks,initFromFile,pandas=TRUE)
    },
    error=function(e){
      message("error in nest data: ", e, "go to next replicate.")
      skiptoNext <- TRUE
    })

    if(skiptoNext) {next}
    # names(nestData1)
    # cat("\nall nest data:\n")
    # print(class(nestData1))
    # print(nestData1)
    #
    # this function may need to be edited to work with pd.DataFrame, but everything else should be easier:
    # nVal <- mod$calc_nests(nestData1, par, repID, parID)

    # excl <- nVal
    # nestData <- subset(nestData1, )
    # nestData <- nestData1[nestData1[[11]]!=0] # remove undiscovered nests
    # nestData <- nestData1[nestData1$totobs!=0] # remove undiscovered nests
    nestData <- nestData1 |> filter(totobs!=0) # remove undiscovered nests
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

    # date_glm <- glm(Surv~poly(Date,2),
    nmod=3
    modOut <- list()
    # int_glm  <- glm(Surv~1,
    modOut[[1]]  <- glm(Surv~1,
                   family=binomial(link=logexp(dat2S$Exposure)),
                   data=dat2S,start=c(1))
    if (debug) print(summary(modOut[[1]]))
    if (debug) print(coef(modOut[[1]]))
    # if (debug) print(cbind(coef(modOut[[1]]), confint.default(modOut[1])))


    modOut[[2]]<- glm(Surv~Date,
                   family=binomial(link=logexp(dat2S$Exposure)),
                   data=dat2S,start=c(1,0))
    if (debug) print(summary(modOut[[2]]))
    if (debug) print(coef(modOut[[2]]))
    # if (debug) print(cbind(coef(modOut[[2]]), confint.default(modOut[2])))

    modOut[[3]]<- glm(Surv~Date + I(Date^2),
                   family=binomial(link=logexp(dat2S$Exposure)),
                   data=dat2S,start=c(1,0,0))
    if (debug) print(summary(modOut[[3]]))
    if (debug) print(coef(modOut[[3]]))
    # if (debug) print(cbind(coef(modOut[[3]]), confint.default(modOut[3])))
    
    # coefs[,parID,repID] = sapply(seq(nmod), function(x){

    # coefs[,r,i] = sapply(modOut, function(x){
    if (debug) cat("\n|> CREATE COEFS ARRAY \n")
    ## I could swear this was outputting a vector before; not sure what changed..
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


    repID = repID + 1
  }
  parID = parID + 1
}
if (debug) print(coefs)

