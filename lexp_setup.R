

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

##--------------- SET SOME VALUES: ----------------------------------------------
# odir <- funs$mk_outdir(now)
odir <- sett$odir
homedir <- "/home/wodehouse/Projects/sim_model"
dirName <- sprintf("%s%s", config$rngSeed, atype)
outdir <- file.path(odir,dirName)
if(!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)
suff <- sprintf("%s%s", config$rngSeed, atype)
paramsArray <- sett$paramsArray
staticPar <- sett$staticPar # this one IS a list
pArrList <- sett$pArrList ## should be able to use in R?
# +> automatic type conversion is NOT working for dicts. and neither is explicit conversion.
# print(length(paramsArray))
# print(length(pArrList))
parr <- py_to_r(paramsArray)[[1]] # print(parr)
preDays <- parr$brDays
prDays <- seq(preDays)
nparsets <- length(pArrList)
cat(sprintf("\n\tnumber of days for prediction: %s \n",preDays))
mList <- c("Surv~1", "Surv~Date", "Surv~Date+I(Date^2)", "Surv~Age", "Surv~Age+Date")
mNames <- c("m1", "m2", "m3", "m4")
nmod <- length(mList)

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

nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","rmDSR","rmPSR")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1")
nValMat <- array(NA, dim=c(length(nval_name), nreps,length(pArrList)), dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))
pred2 <- array(NA, dim=c(2, preDays, nparsets), dimnames=list(c("m2","m3"),seq(preDays), seq(nparsets)) ) # print(pred2)

