

source("/home/wodehouse/.local/bin/r_func.R")
options(width=1000, digits=5, scipen=999)
Sys.setenv(printset=TRUE) 
arg <- unlist(strsplit(commandArgs(trailingOnly=TRUE), split=" "))
py_vars <- import_builtins()$vars ## function to turn class instance into dictionary
mcType=2 ## default value

if(length(arg)==0){
  message("** NO atype ARGUMENT PROVIDED; USING DEFAULT")
  atype=""
} else {
  for(a in arg){
    cat("length(arg) = ", length(arg),"arg = ",a)
    if(grepl("at\\w+", a)) atype <- stringr::str_extract(a, "(?<=at)\\w+") # extract words after "at"
    if(grepl("mc\\w+", a)) mcType <- stringr::str_extract(a, "(?<=mc)\\w+") # extract words after "at"
    cat("\t\t>> atype =",atype)
    cat("\t\t>> mcType =",mcType)
    # if(a=="test") params$test <- TRUE
  }
} # atype <- "full"

## doesn't work to pass vars to python:
# Sys.setenv(atypeR="norm", script_name="logexp.R")
# Sys.setenv(atypeR=atype, script_name="logexp.R")
Sys.setenv(atypeR=atype) 
Sys.setenv(mcTypeR=mcType) 
# py_run_file("rsettings.py")
# py_run_file("init.py")
## maybe just running this ^ starts the repl in python?
## after running it, loading rsettings below still loads the correct values
## but, it doesn't seem to load anything into the R environment, so still need to do that below?

##--------------- IMPORT PYTHON FUNCTIONS: ----------------------------------------------
if(TRUE){
  cat("\t>> importing python functions \t")
  json <- import("json")
  dc <- import("dataclasses")
  pickle <- import("pickle")
  # np <- import("numpy")
  cat("\t>> importing custom functions \t")
  
  obs <- import("observer")
  printFun <- import("print_func")
  dsr <- import("dsrCalc")
  nest <- import("makeNests")
  funs <- import("helpers")
  mod <- import("datsim")
  logex <- import("log_exposure")
  cat("\t>> importing settings \t\n")
  sett <- import("rsettings", convert=FALSE) # sett <- import("settings", )
  # cat("\n\t>> extracting config \t")
  pyconfig <- sett$config
  config <- py_to_r(sett$config) # I DO want to convert config, but not the param dicts
  # print(class(config))
  cat(sprintf("\n\t|> importing config; type=%s\n", class(config)[1]))
  # print(unlist(as.list(config)))
  # withr::with_options( list(width=120), print(unlist( py_vars(config) )) )
  # cat(unlist(py_vars(config)))
  # print(py_vars(config))
}
# cat("CONFIG:", paste(config,collapse=";"), "\n")
# if(debug>=4) cat("\n\t|>python-format config: ", class(pyconfig))
if(config$testing=="yes") library(ggplot2)

##--------------- CONFIG PRESETS: ----------------------------------------------
# cat("\nPRESET: ",preset)
# if(preset==1){
#   config$rngSeed = 1111953
#   config$hTime = 2
# } else if(preset==2){
#   config$rngSeed=7139158
#   config$hTime = 1
# } else if(preset==3){
#   config$rngSeed=10281991
#   config$hTime = 0
# } else if(preset==4){
# } else if(preset==5){
# }
cat("\nCONFIG:\n")
withr::with_options( list(width=120), print(unlist( py_vars(config) )) )

##--------------- SET SOME VALUES: ----------------------------------------------
# odir <- funs$mk_outdir(now)

if(TRUE){
  nreps <- config$nreps
  debug <- config$debug

  suff <- sprintf("%s%s", config$rngSeed, atype)
  rng <- sett$rng
  # if(debug>=4) cat(sprintf("\t|> debug: %s |>numpy rng: %s %s \n", debug, rng, class(rng)))
  homeDir <- "/home/wodehouse/Projects/sim_model"
  odir <- sett$odir
  dirName <- sprintf("%s%s", config$rngSeed, atype) # homedir <- "/home/wodehouse/Projects/sim_model"
  outdir <- file.path(odir,dirName)
  if(!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)

  stormDat <- sett$stormDat
  initDat <- sett$initDat
  
  paramsArray <- sett$paramsArray
  pLists <- sett$pLists
  # print(unlist(py_to_r(pLists)))
  staticPar <- sett$staticPar # this one IS a list
  pArrList <- sett$pArrList ## should be able to use in R? # if(debug>=2) print(pArrList)
# +> automatic type conversion is NOT working for dicts. and neither is explicit conversion.
  parr <- py_to_r(paramsArray)[[1]] ## first set; breeding days is the same in all
  # if(debug>=2) print(unlist(parr))
  # parr <- py_to_r(paramsArray)
  # if(debug>=2) print(unlist(parr))
  # print(unlist(py_to_r(pLists)))
  nparsets <- length(pArrList)
  
  preDays <- parr$brDays
  prDays <- seq(preDays)
  # cat(sprintf("\n\tnumber of days for prediction: %s \n",preDays))
}

##--------------- MODEL LISTS: ----------------------------------------------
mList <- c("Surv~1", "Surv~Date", "Surv~Date+I(Date^2)", "Surv~Age", "Surv~Age+Date", "Surv~avDate")
mList_supp <- c("Surv~avDate","Surv~avAge","Surv~avAge+avDate")
# mNames <- c("m1", "m2", "m3", "m4","m5")
mNames <- c("m1", "m2", "m3", "m4")
nmod <- length(mList)
# initDateList <- as.data.frame(py_to_r(nest$initDat))

# if(config$testing=="yes"){
#   cat("\ntrue init date list:\n")
#   initDateList <- py_to_r(nest$initDat)
#   print(class(initDateList))
#   print(initDateList)
# }

##--------------- CREATE ARRAYS TO STORE DATA: ----------------------------------------------
if (config$predict){
  dimss <- c(3,preDays,nmod,length(pArrList)) # print(dimss)
  dnames <- list(c("est", "lcl", "ucl"), seq(preDays),mNames,seq(length(pArrList))) # print(dnames)
  pred <- array(NA, dim=dimss, dimnames=dnames )
  if(debug>=6) print(dimnames(pred))
}
## NOTE: this only includes the first 5 models
coef_list <- c("est","lcl","ucl")
mod_names <- c("_dot","_date_int","_date_b1","_datesq_int","_datesq_b1","_datesq_b2",
               "age_int","age_b1","agedate_int","agedate_b1","agedate_b2")
coef_names <- do.call(paste0, expand.grid(coef_list, mod_names))
coefs <- array(NA,
               dim=c(length(coef_names), nreps, length(pArrList)),
               dimnames=list(coef_names,seq(nreps), seq(length(pArrList))))
if(debug>=6) print(dimnames(coefs))

# lexp_name <- c("leDSR1","lePSR1","leDSRdate","lePSRdate","lePSR2","lePSR3","lePSR4","lePSR5")
if(config$logex){
  lexp_name <- c("leDSR1","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
# lexp_supp <- c("leDSRava","lePSRava","leDSRavd","lePSRavd","leDSRavad","lePSRavad")
  lexp_supp <- c("leDSRavd","lePSRavd","leDSRava","lePSRava")
# truedsr_name <- c("tDSR","tPSR","tDSRdate","tPSRdate","tDSRage","tPSRage")
} else {
  lexp_name <- c()
  lexp_supp <- c()
}
truedsr_name <- c("tDSR","tPSR")
if(config$mcmcOld){
  mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDFR","mcmcDSR_old","mcmcPSR_old","mcmcDFR_old")
  mayfdsr_name <- c("mayfDSR","mayfDSR_old")
} else {
  mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDFR")
  mayfdsr_name <- c("mayfDSR")
}
mark_name <- c()
if(config$mark) mark_name <- c("markDSR","markPSR","markDSRdate","markPSRdate","markDSRdsAge","markPSRdsAge","marktopDSR","marktopmod")

# dsr_name  <- c(truedsr_name,lexp_name,lexp_supp, mcmc_name, mark_name)

# dsr_name  <- c("parID","repID",truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
dsr_name  <- c(truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
print(dsr_name)
# print(length(dsr_name))
dsrMat <- array(NA,
                dim=c(length(dsr_name), nreps,length(pArrList)),
                dimnames=list(dsr_name,seq(nreps), seq(length(pArrList))))

# print(dim(dsrMat))
# print(dimnames(dsrMat))
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mfDSR","appDSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
                # 1       2     3       4     5       6     7     8     9       10      11    12      13    14        
nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc",
               "avfint","avk","aDSR","aPSR","mfDSR","appDSR")
# dsr_name <- c("leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
# allval_name <- c(nval_name, dsr_name,mcmc_name,mark_name)
nValMat <- array(NA,
                 dim=c(length(nval_name), nreps,length(pArrList)),
                 dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))
# allval_name <- c(nval_name, dsr_name,mark_name)
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","rmDSR","rmPSR")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1")
# nValMat <- array(NA, dim=c(length(nval_name), nreps,length(pArrList)), dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))

pred2 <- array(NA, dim=c(2, preDays, nparsets), dimnames=list(c("m2","m3"),seq(preDays), seq(nparsets)) ) # print(pred2)

# vnames <- c("trueDSR","discDSR","anDSR","lexpDSR", "diff1","diff2","diff3")
## for summary at end of test:
# vnames <- c("trueDSR","lexpDSR","mcmcDSR","markDSR","mayfDSR", "diff_lexp","diff_mcmc","diff_mark","diff_mayf")
# vnames <- c("true","lexp","lexp_top","mcmc","mark","mark_top","mayf")
# vnames <- c("true","app","lexp","mcmc","mark","mark_top","mayf")
if(config$mcmcOld){
  vnames <- c("true","app","lexp","mcmc","mcmc_old","mayf")
} else {
  vnames <- c("true","app","lexp","mcmc","mayf")
}
vnames2 <- paste0("dsr_",vnames)
diffnames <- paste0("diff_",vnames[-1])
allnames <- c("number_storms","storm_mortality","obs_interval","discovery_probability","evidence_decay_rate",
              # "dsr_given","number_discovered","number_excluded",vnames2,diffnames)
              "dsr_given","number_discovered","number_excluded","proportion_excluded","proportion_misclassified",
              vnames2,diffnames)
# print(allnames)
# valMat <- array(NA, dim=c(length(vnames), nreps,nparsets), dimnames=list(vnames,seq(nreps), seq(nparsets)))
valMat <- array(NA, dim=c(length(allnames), nreps,nparsets), dimnames=list(allnames,seq(nreps), seq(nparsets)))
# if(debug>=3) print(valMat)

Sys.setenv(printset=FALSE) 
