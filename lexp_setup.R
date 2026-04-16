

options(width=1000)
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
} # atype <- "full"

## doesn't work to pass vars to python:
# Sys.setenv(atypeR="norm", script_name="logexp.R")
Sys.setenv(atypeR=atype, script_name="logexp.R")
py_run_file("rsettings.py")
## maybe just running this ^ starts the repl in python?
## after running it, loading rsettings below still loads the correct values
## but, it doesn't seem to load anything into the R environment, so still need to do that below?

##--------------- IMPORT PYTHON FUNCTIONS: ----------------------------------------------
if(TRUE){
  obs <- import("observer")
  dsr <- import("dsrCalc")
  nest <- import("makeNests")
  funs <- import("helpers")
# sett <- import("settings", )
  sett <- import("rsettings", convert=FALSE)
  mod <- import("datsim")
  logex <- import("log_exposure")
  config <- py_to_r(sett$config) # I DO want to convert config, but not the param dicts
  nreps <- config$nreps
  debug <- config$debug
}
cat("CONFIG:", paste(config,collapse=";"), "\n")
if(config$testing=="yes") library(ggplot2)

##--------------- SET SOME VALUES: ----------------------------------------------
# odir <- funs$mk_outdir(now)
if(TRUE){
  suff <- sprintf("%s%s", config$rngSeed, atype)
  odir <- sett$odir
  # homedir <- "/home/wodehouse/Projects/sim_model"
  dirName <- sprintf("%s%s", config$rngSeed, atype)
  outdir <- file.path(odir,dirName)
  if(!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)
  
  paramsArray <- sett$paramsArray
  staticPar <- sett$staticPar # this one IS a list
  pArrList <- sett$pArrList ## should be able to use in R?
# if(debug>=2) print(pArrList)
# +> automatic type conversion is NOT working for dicts. and neither is explicit conversion.
# print(length(paramsArray))
# print(length(pArrList))
  parr <- py_to_r(paramsArray)[[1]]
  if(debug>=2) print(parr)
# write.csv(parr,sprintf("param_sets%s.csv",config$rngSeed))
  
  preDays <- parr$brDays
  prDays <- seq(preDays)
  cat(sprintf("\n\tnumber of days for prediction: %s \n",preDays))

  mList <- c("Surv~1", "Surv~Date", "Surv~Date+I(Date^2)", "Surv~Age", "Surv~Age+Date")
  mNames <- c("m1", "m2", "m3", "m4")

  nmod <- length(mList)
  nparsets <- length(pArrList)
}
# initDateList <- as.data.frame(py_to_r(nest$initDat))

if(config$testing=="yes"){
  cat("\ntrue init date list:\n")
  initDateList <- py_to_r(nest$initDat)
  print(class(initDateList))
  print(initDateList)
}

##--------------- CREATE ARRAYS TO STORE DATA: ----------------------------------------------
if (config$predict){
  dimss <- c(3,preDays,nmod,length(pArrList)) # print(dimss)
  dnames <- list(c("est", "lcl", "ucl"), seq(preDays),mNames,seq(length(pArrList))) # print(dnames)
  pred <- array(NA, dim=dimss, dimnames=dnames )
  if(debug>=4) print(dimnames(pred))
}
coef_list <- c("est", "lcl", "ucl")
mod_names <- c("_dot", "_date_int", "_date_b1", "_datesq_int","_datesq_b1","_datesq_b2","age_int", "age_b1","agedate_int", "agedate_b1", "agedate_b2")
coef_names <- do.call(paste0, expand.grid(coef_list, mod_names))
coefs <- array(NA,
               dim=c(length(coef_names), nreps, length(pArrList)),
               dimnames=list(coef_names,seq(nreps), seq(length(pArrList))))
if(debug>=4) print(dimnames(coefs))

# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mfDSR","appDSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mfDSR","appDSR")
# dsr_name <- c("leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
dsr_name <- c("leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDFR")
mark_name <- c("markDSR","markPSR")
allval_name <- c(nval_name, dsr_name,mcmc_name,mark_name)
# allval_name <- c(nval_name, dsr_name,mark_name)
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","rmDSR","rmPSR")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1")
# nValMat <- array(NA, dim=c(length(nval_name), nreps,length(pArrList)), dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))
nValMat <- array(NA, dim=c(length(allval_name), nreps,length(pArrList)), dimnames=list(allval_name,seq(nreps), seq(length(pArrList))))

pred2 <- array(NA, dim=c(2, preDays, nparsets), dimnames=list(c("m2","m3"),seq(preDays), seq(nparsets)) ) # print(pred2)

# vnames <- c("trueDSR","discDSR","anDSR","lexpDSR", "diff1","diff2","diff3")
vnames <- c("trueDSR","lexpDSR","mcmcDSR","markDSR","mayfDSR", "diff_lexp","diff_mcmc","diff_mark","diff_mayf")
valMat <- array(NA, dim=c(length(vnames), nreps,nparsets), dimnames=list(vnames,seq(nreps), seq(nparsets)))
# if(debug>=3) print(valMat)

