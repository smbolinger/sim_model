


startTime <- Sys.time()
# arg <- unlist(strsplit(commandArgs(trailingOnly=TRUE), split=" "))
arg <- unlist(strsplit(commandArgs(trailingOnly=FALSE), split=" "))
# print(arg)
# file_arg <- grep("(?<=^--file=)[A-Za-z]*\,[A-Za-z]", arg, value = TRUE)
# file_arg <- grep("(?<=file=)(\\w+\\.\\w+)", arg, perl=TRUE, value = TRUE)
if(length(arg)<2){
  # message("** NO atype ARGUMENT PROVIDED; USING DEFAULT")
  # atype=""
  file_arg <- ""
} else {
  # file_arg <- stringr::str_extract(arg, "(?<=file=)(\\w+\\.\\w+)")
  file_arg <- stringr::str_extract(arg, "(?<=file=)([A-Za-z0-9_/]+\\.\\w+)")
  file_arg <- file_arg[!is.na(file_arg)]
}
# print(file_arg)
# print( class(file_arg))
library(reticulate)
# Sys.setenv(script_name="all_dsr.R")
Sys.setenv(script_name=file_arg)
py_run_file("init.py")
library(MASS)
library(brglm2)
library(tidyr)
suppressPackageStartupMessages(library(dplyr)) # load dplyr last so as not to mask select?
# arg <- unlist(strsplit(commandArgs(trailingOnly=TRUE), split=" "))

## NOTE: these args should only be used to change config options that don't affect python, unlss I want to also apply them to pyconfig
arg <- unlist(commandArgs(trailingOnly=TRUE))

# library(jsonlite)
#NOTE could make a counter of all times at least one survey int == 0
# psrTrue = "date"
psrTrue = "a"
nruns = NA ## doesn't affect python
debug = "" ## NOTE: DOES affect python?
# seedArg = NA
rngSeed = NA ## doesn't affect python
# startParArg = NA
startParID = NA ## doesn't affect python
msg = "" ## doesn't affect python

source("/home/wodehouse/.local/bin/r_func.R")
options(width=1000, digits=5, scipen=999)
Sys.setenv(printset=TRUE) 
# arg <- unlist(strsplit(commandArgs(trailingOnly=TRUE), split=" "))
py_vars <- import_builtins()$vars ## function to turn class instance into dictionary
py_attr <- import_builtins()$setattr ## function to turn class instance into dictionary
mcType=0 ## default value
file.create("out/psr_plot.txt")
file.create("out/storm_plot.txt")
file.create("out/init_plot.txt")
# file_arg <- grep("^--file=", arg, value = TRUE)
print(arg)

if(length(arg)==0){
  message("** NO atype ARGUMENT PROVIDED; USING DEFAULT")
  atype=""
} else {
  for(a in arg){
    # cat("length(arg) = ", length(arg),"arg = ",a)
    cat("\t\tlength(arg) = ", length(arg))
    if(grepl("at\\w+", a)) atype <- stringr::str_extract(a, "(?<=at)\\w+") # extract words after "at"
    if(grepl("mc\\w+", a)) mcType <- stringr::str_extract(a, "(?<=mc)\\w+") # extract words after "at"
    if(grepl("r\\d+", a)) nruns <- stringr::str_extract(a, "(?<=r)\\d+") # extract words after "at"
    if(grepl("db\\d+", a)) debug <- as.numeric(stringr::str_extract(a, "(?<=db)\\d+")) # extract words after "db"
    if(grepl("par\\d+", a)) startParID <- stringr::str_extract(a, "(?<=par)\\d+") # extract words after "db"
    if(grepl("rng\\d+", a)) rngSeed <- stringr::str_extract(a, "(?<=rng)\\d+") # extract words after "db"
    if(grepl("msg:", a)) msg <- stringr::str_extract(a, "(?<=msg:).*") # extract words after "db"
    if(grepl("help", a)){
      cat("\n>> help-arg values: at<type> | mc<mctype> | r<nruns> | db<debug val> | par<start param id> | rng<start seed>")
      quit(save = "no", status = 1, runLast = FALSE)
    }

  }
  cat("\t\t>> atype =",atype)
  cat("\t\t>> nruns arg =",nruns)
  cat("\t\t>> debug arg =",debug,"type=",class(debug))
  cat("\t\t>> startParID arg =",startParID)
  cat("\t\t>> rngSeed arg =",rngSeed) ## NOTE: changing rng seed here does not affect output dir
  # cat("\t\t>> mcType arg =",mcType)
  cat("\t\t>> msg arg =",msg)
  # cat("\t\t>> debug =",debug)
  # if(a=="test") params$test <- TRUE
} # atype <- "full"

## doesn't work to pass vars to python:
# Sys.setenv(atypeR="norm", script_name="logexp.R")
# Sys.setenv(atypeR=atype, script_name="logexp.R")
mc_file = "MCmatrix"
# mc_file = "notr_MCmatrix"
Sys.setenv(atypeR=atype) 
Sys.setenv(mcTypeR=mcType) 
# py_run_file("rsettings.py")
# py_run_file("init.py")
## maybe just running this ^ starts the repl in python?
## after running it, loading rsettings below still loads the correct values
## but, it doesn't seem to load anything into the R environment, so still need to do that below?

##--------------- IMPORT PYTHON FUNCTIONS: ----------------------------------------------
if(TRUE){
  cat("\n>> importing python functions \t")
  json <- import("json")
  dc <- import("dataclasses")
  pickle <- import("pickle")
  np <- import("numpy")
  cat("\t>> importing custom functions \n\t")
  obs <- import("observer")
  nest <- import("makeNests")
  printFun <- import("print_func")
  dsr <- import("dsrCalc")
  funs <- import("helpers")
  # mod <- import("datsim")
  mlfun <- import("matlab_func")
  # mod <- import("scipy_datsim")
  # mod <- import("lmfit_datsim")
  # mod <- import("datsim_new")
  # logex <- import("log_exposure")
  cat("\t>> importing settings \t\n")
  # Sys.sleep(3) ## keep python from printing too soon
  sett <- import("rsettings", convert=FALSE) # sett <- import("settings", )
  # cat("\n\t>> extracting config \t")
  # print(unlist(as.list(config)))
  # withr::with_options( list(width=120), print(unlist( py_vars(config) )) )
  # cat(unlist(py_vars(config)))
  # print(py_vars(config))
  # cat(sprintf("\nusing %s\n", mc_file))
}
# cat("CONFIG:", paste(config,collapse=";"), "\n")
# if(debug>=4) cat("\n\t|>python-format config: ", class(pyconfig))

##--------------- CONFIG: ----------------------------------------------
if(TRUE){
  ## preset values might make things easier?
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
  pyconfig <- sett$config
  ## want to convert config to be readable in R, but not all the way to list, because harder to change types?
  config <- py_to_r(sett$config) # I DO want to convert config, but not the param dicts
  # config <- py_vars(py_to_r(sett$config)) # I DO want to convert config, but not the param dicts
  # cat("\t\t>> seed =", config$rngSeed)
  # print(class(config))
  if(debug!=""){
    config$testing="yes"
    cat("\t>> changing config$testing to ", config$testing)
    # config$debug = debug else debug = config$debug
  }
  # if(debug!="") config$testing=="yes"
  # cat("\n\t>> changing config$testing to ", config$testing)
  if(debug!="") config$debug = debug else debug = config$debug

  cat("\n>> debug =",debug,"type=",class(debug))
  cat("\t\t>> config$debug =",config$debug,"type=",class(config$debug))
  # if(!is.na(debug)) config.debug = debug else debug = config.debug
  # cat(":\nconfig:\n")
  # withr::with_options( list(width=120), print(unlist( py_vars(py_to_r(pyconfig)) )) )
  # withr::with_options( list(width=120), print(unlist( py_vars(config) )) )
  # cat("\t|>debug = ", debug)
  # cat("\n config type=", class(config))
  if(config$debug>=5) cat(sprintf("\n\t|> importing pyconfig <%s> & config <%s>\n", class(pyconfig)[1], class(config)[1]))
  # debugVals <- lapply(config, function(x) stringr::str_match(x, "debug"))
  # debugVals <- sapply(names(py_vars(config)), function(x) grep("debug",x,fixed=TRUE)) ## get indices of debug vals
  debugVals <- grep("debug",names(py_vars(config)),fixed=TRUE)
  # print(debugVals)
  debugNames = names(py_vars(config))[debugVals]
  # print(debugNames)
  # print(config[debugVals])
  # config <- lapply(debugVals, function(x){
  for(x in debugNames) if(as.numeric(config[[x]])>=as.numeric(debug)) py_attr(config,x,debug)
  cat("\ndebug vals & types:")
  for(x in debugNames) cat("x:",config[[x]],class(config[[x]]))
#   config <- lapply(debugNames, function(x){
#                      ## debugVals are indices
#                      # ifelse(as.numeric(config$x)>=as.numeric(debug), debug, config$x)
#                      # if(config[[x]]>=debug) config[[x]] = debug 
#
#                      if(as.numeric(config[[x]])>=as.numeric(debug)) py_attr(config,x,debug)
#                      # if(as.numeric(config[[x]])>=as.numeric(debug)) config[[x]] = debug 
# })
  # cat("\n config type=", class(config))
  # cat(":\nconfig after:\n")
  # withr::with_options( list(width=120), print(unlist( py_vars(config) )) )
  # withr::with_options( list(width=120), print(unlist( py_vars(config) )) )
}

debug <- as.numeric(debug)
# if (debug < 1) {
#     obs <- import("nodebug_observer")
#     nest <- import("nodebug_makeNests")
# }
cat("\nCONFIG")
## this changes vals for R config, but not py config

# ------------------- CHANGE CONFIG VALUES BEFORE PRINTING: -------------------------------------------------------------------------------


if (msg!="") config$msg = paste(config$msg, msg, sep=";")
rngSeed <- ifelse(!is.na(rngSeed), as.integer(rngSeed),as.integer(config$rngSeed))
rng <- np$random$default_rng(seed=rngSeed)
startParID <- ifelse(!is.na(startParID),as.integer(startParID), as.integer(config$startParID))
nreps <- ifelse(!is.na(nruns), as.integer(nruns),config$nreps)

  # config$msg = sprintf("%s\t\t%s",config$msg, paste(msg, sep=" "))

# if(config$nreps>20) cat(" [ !! NOTE: large # of reps - force debug values low unless overridden by CL arg] ")
if(nreps>20) cat(" [ !! NOTE: large # of reps - force debug values low unless overridden by CL arg] ")
# if(config$nreps>20){
#   cat(" [ !! NOTE: large # of reps - force debug values low ] ")
#   config$debug=2
#   if(config$debugObs>=2) config$debugObs=1 ## if val is 0, this won't change it to 1
#   if(config$debugNests>=2) config$debugNests=1
#   if(config$debugDSR>=2) config$debugDSR=1
#   if(config$debugLogEx>=2) config$debugLogEx=1
#   if(config$debugLL>=2) config$LL=1
#   if(config$debugSummary>=2) config$Summary=1
# } else if(config$nreps>50){
#   cat(" [ !! NOTE: VERY large # of reps - force debug values low ] ")
#   config$debug=1
#   config$debugObs=0
#   config$debugNests=0
#   config$debugDSR=0
#   config$debugLogEx=0
#   config$LL=0
#   config$Summary=0
# }
#
# cat("\nCONFIG:\n")
cat(":\n")
withr::with_options( list(width=120), print(unlist( py_vars(config) )) )
# withr::with_options( list(width=120), print(unlist( py_vars(py_to_r(pyconfig)) )) )
if(config$testing=="yes") library(ggplot2)

##--------------- SET SOME VALUES: ----------------------------------------------
# odir <- funs$mk_outdir(now)
# rng = np.random.default_rng(seed=config.rngSeed)

if(TRUE){
  # obsVar = "nobs" ## which obs variable to use to decide nest discovery
  obsVar = "totobs"
  obsVarNum = ifelse(obsVar=="nobs", 8, 10)

  # debug <- config$debug
  suff <- sprintf("%s%s", config$rngSeed, atype)
  # rng <- sett$rng
  # stormUnk <- py_to_r(sett$stormUnk)
  # if(debug>=4) cat(sprintf("\t|> debug: %s |>numpy rng: %s %s \n", debug, rng, class(rng)))
  homeDir <- "/home/wodehouse/Projects/sim_model"
  odir <- sett$odir
  dirName <- sprintf("%s%s_inc", config$rngSeed, atype) # homedir <- "/home/wodehouse/Projects/sim_model"
  outdir <- file.path(odir,dirName)
  if(!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)

  stormDat <- sett$stormDat
  initDat <- sett$initDat
  
  pLists <- sett$pLists
  # print(unlist(py_to_r(pLists)))
  staticPar <- sett$staticPar # this one IS a list
  paramsArray <- sett$paramsArray
  pArrList <- sett$pArrList ## should be able to use in R? # if(debug>=2) print(pArrList)
  vary     <- py_to_r(sett$vary)
  stormUnk <- py_to_r(staticPar$stormUnk)
  stormUnk <- as.integer(stormUnk)
  # cat("\n\t|>debug = ", debug)
  # cat("\tvary=", vary)
# +> automatic type conversion is NOT working for dicts. and neither is explicit conversion.
  parr <- py_to_r(paramsArray)[[1]] ## first set; breeding days is the same in all
  # if(debug>=2) print(unlist(parr))
  # parr <- py_to_r(paramsArray)
  # if(debug>=2) print(unlist(parr))
  # print(unlist(py_to_r(pLists)))
  nparsets <- length(pArrList)
  
  # preDays <- parr$brDays
  breDays <- py_to_r(staticPar$brDays)
  brDays <- seq(breDays)
  preDays <- 150
  prDays <- seq(preDays)
  ## column names for nest data in R:
  # colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
}
colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs', 'sint')

##--------------- MODEL LISTS: ----------------------------------------------
# mList <- c("Surv~1", "Surv~Date", "Surv~Date+I(Date^2)", "Surv~Age", "Surv~Age+Date", "Surv~avDate")
# mList <- c("Surv~1", "Surv~Date", "Surv~Age", "Surv~Age+Date", "Surv~avDate")
mList <- c("Surv~1", "Surv~Date", "Surv~Age", "Surv~Age+Date", "Surv~avDate")
mList_supp <- c("Surv~avDate","Surv~avAge","Surv~avAge+avDate")
# mNames <- c("m1", "m2", "m3", "m4","m5")
mNames <- c("m1", "m2", "m3", "m4")
nmod <- length(mList)
# initDateList <- as.data.frame(py_to_r(nest$initDat))
cat(sprintf("\n|>|> %s reps x %s param sets = %s rows", nreps, nparsets, nreps*nparsets))
if(config$msg!="") cat("\n|>MSG: ", config$msg)
cat("\nOTHER:")
# qvcalc::indentPrint(unlist(py_to_r(pLists)))
# cat("\tparam lists:")
# qvcalc::indentPrint(unlist(py_vars(pLists)))
# cat("\n\tusing randArgs2 function to create initial params for optim")
cat("\n\t|>debug = ", debug)
cat("\tvary=", vary)
cat("\texplicitly mark storm nests 'unknown'?", stormUnk)
cat("\n\tmList = ", mList)
cat(sprintf("\t\t| discovered nests = where %s > 0", obsVar))
cat(sprintf("\n\t| number of days for prediction: %s <%s>",preDays,class(preDays)))
cat(sprintf("\t\t| true number of reps: %s",nreps))
cat(sprintf("\t\t| num param sets: %s",nparsets))
cat(sprintf("\t\t| starting param set: %s",startParID))
cat(sprintf("\t\t| starting rngSeed: %s",rngSeed))
cat("\n\t>> check modules imported to datsim.py:")
# withr::with_options( list(width=120), print(unlist( py_vars(mod$imported) )) )
# if(vary!="decayRate"){
# if(all(vary!="decayRate")){
if(!any(vary=="decayRate")){
  evidProb <- sapply(c(3,5,7), function(x) funs$expDecay(1, staticPar$decayRate, x))
  cat(sprintf("\n\t\t>>->> probability of fate evidence after 3 days:%s; 5 days:%s; 7 days:%s",
              evidProb[1], evidProb[2], evidProb[3]))
}

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
# coef_list <- c("est","lcl","ucl")
coef_list <- c("est","se")
# mod_names <- c("_dot","_date_int","_date_b1","_datesq_int","_datesq_b1","_datesq_b2",
#                "age_int","age_b1","agedate_int","agedate_b1","agedate_b2")
mod_names <- c("_null","_date_int","_date_b1", "_age_int","_age_b1",
               "_agedate_int","_agedate_b1","_agedate_b2","_avdate_int","_avdate_b1")
# coef_names <- do.call(paste0, expand.grid(coef_list, mod_names))
coef_names <- unlist(lapply(coef_list, function(x) paste0(x,mod_names)))
coef_names <- c(coef_names, "avg_expos","avg_age","avg_avdate")
# if(debug>=0) print(seq(nreps))
# if(debug>=0) print(class(nreps))
coefsMat <- array(NA,
               dim=c(length(coef_names), nreps, length(pArrList)),
               dimnames=list(coef_names,seq(nreps), seq(length(pArrList))))
# if(debug>=0) qvcalc::indentPrint(dimnames(coefs))
if(debug>=0) cat("\n\tcoef names:", coef_names)
# if(debug>=0) qvcalc::indentPrint(coef_names)
coefList <- list()

# lexp_name <- c("leDSR1","lePSR1","leDSRdate","lePSRdate","lePSR2","lePSR3","lePSR4","lePSR5")
if(config$logex){
  # lexp_name <- c("leDSR1","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","lePSR6")
  # lexp_name <- c("leDSR1","lePSR1","leDSR2","leDSR3","leDSR4","leDSR5","leDSR6","lePSR2","lePSR3","lePSR4","lePSR5","lePSR6")
#   lexp_name_p <- c("leDSR1p","leDSR2p","leDSR3p","leDSR4p","leDSR5p","lePSR1p","lePSR2p","lePSR3p","lePSR4p","lePSR5p")
#   lexp_se_p <- c("seDSR1p","seDSR2p","seDSR3p","seDSR4p","seDSR5p")
#   lexp_name <- c("leDSR1","lePSR1","seDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR2","lePSR3","lePSR4","lePSR5")
#   lexp_se <- c("seDSR2","seDSR3","seDSR4","seDSR5")
# # lexp_supp <- c("leDSRava","lePSRava","leDSRavd","lePSRavd","leDSRavad","lePSRavad")
#   lexp_supp <- c("leDSRavd","lePSRavd","leDSRava","lePSRava")
#   lexp_all <- c(lexp_name,lexp_se,lexp_name_p,lexp_se_p)
#   lexpMat <- array(NA,
#                   dim=c(length(lexp_all), nreps,length(pArrList)),
#                   dimnames=list(lexp_all,seq(nreps), seq(length(pArrList))))
#   cat("\n\tlexp matrix dim:", dim(lexpMat))
#   cat("\t\t& lexp val names:", lexp_all)
  # lexp_name <- c("leDSR1","lePSR1","seDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR2","lePSR3","lePSR4","lePSR5")
  lexp_name <- c("leDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
  lexp_se <- c("seDSR1","seDSR2","seDSR3","seDSR4","seDSR5")
# lexp_supp <- c("leDSRava","lePSRava","leDSRavd","lePSRavd","leDSRavad","lePSRavad")
  lexp_supp <- c("leDSRavd","lePSRavd","leDSRava","lePSRava")
# truedsr_name <- c("tDSR","tPSR","tDSRdate","tPSRdate","tDSRage","tPSRage")
} else {
  lexp_name <- c()
  lexp_supp <- c()
}

truedsr_name <- c("tDSR","tPSR","tDSR_date","tPSR_date")
# truedsr_name <- c("tDSR","tPSR","tPSR_date","tDSR2","tPSR2","tPSR_date2")
# if(config$mcmcOld){
#   mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDFR","mcmcDSR_old","mcmcPSR_old","mcmcDFR_old")
#   mayfdsr_name <- c("mayfDSR","mayfDSR_old")
# } else {
mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcDSR_se","mcmcDPR_se")
# mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcSE")
mayfdsr_name <- c("mayfDSR","mayfPSR","mayfVar","mayfSE","simplePSR")
  # mayfdsr_name <- c("mayfDSR")
# }
mark_name <- c()
if(config$mark) mark_name <- c("markDSR","markPSR","markDSRdate","markPSRdate","markDSRdsAge","markPSRdsAge","marktopDSR","marktopmod")

# dsr_name  <- c(truedsr_name,lexp_name,lexp_supp, mcmc_name, mark_name)

# dsr_name  <- c("parID","repID",truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
# dsr_name  <- c(truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
dsr_name  <- c(truedsr_name,lexp_name,lexp_se, mcmc_name,mayfdsr_name, mark_name)
# print(dsr_name)
# print(length(dsr_name))
dsrMat <- array(NA,
                dim=c(length(dsr_name), nreps,length(pArrList)),
                dimnames=list(dsr_name,seq(nreps), seq(length(pArrList))))
cat("\n\tDSR matrix dim:", dim(dsrMat))
cat("\t\t& DSR val names:", dsr_name)

# print(dim(dsrMat))
# print(dimnames(dsrMat))
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mfDSR","appDSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
                # 1       2     3       4     5       6     7     8     9       10      11    12      13    14        
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc",
# nval_name <- c("parID","repID","fld", "hat", "sNest", "dsc", "excl","unk","mc",
               # "avfint","avk","maxi","aDSR","aPSR","mfDSR","appDSR")
nval_name <- c("parID","repID","fld", "hat","fld_dsc","hat_dsc","fld_an","hat_an", "sNest",
               "dsc", "excl","unk","mc","mc2", "avfint","avk","maxi","lint","aDSR","aPSR","mfDSR","appDSR")
cat("\n\tnVal names:", nval_name)
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

# pred2 <- array(NA, dim=c(2, preDays, nparsets), dimnames=list(c("m2","m3"),seq(preDays), seq(nparsets)) ) # print(pred2)
# par_names <- c("storm_fate", "num_nests", "flsurv_given", "MCtype", "propMC", "propUnk",
               # "storm_dur", "storm_freq", "obs_int", "hatch_time", 
# )

# vnames <- c("trueDSR","discDSR","anDSR","lexpDSR", "diff1","diff2","diff3")
## for summary at end of test:
# vnames <- c("trueDSR","lexpDSR","mcmcDSR","markDSR","mayfDSR", "diff_lexp","diff_mcmc","diff_mark","diff_mayf")
# vnames <- c("true","lexp","lexp_top","mcmc","mark","mark_top","mayf")
# vnames <- c("true","app","lexp","mcmc","mark","mark_top","mayf")
if(config$mcmcOld){
  vnames <- c("true","app","lexp","mcmc","mcmc_old","mayf")
} else {
  # vnames <- c("true","app","lexp","mcmc","mayf")
  # vnames <- c("true","true2","app","lexp","lexp-pr","mcmc","mayf")
  vnames <- c("true","true2","app","lexp","mcmc","mayf")
}
# vnames2 <- paste0("dsr_",vnames)
vnames2 <- paste0("psr_",vnames)
diffnames <- paste0("diff_",vnames[-1])
allnames <- c("number_storms","storm_mortality","obs_interval","discovery_probability","evidence_decay_rate",
              # "dsr_given","number_discovered","number_excluded",vnames2,diffnames)
              "dsr_given","total_number_hatched", "total_number_flooded","number_discovered","number_excluded",
              "proportion_excluded","proportion_misclassified",
              vnames2,diffnames)
# print(allnames)
# valMat <- array(NA, dim=c(length(vnames), nreps,nparsets), dimnames=list(vnames,seq(nreps), seq(nparsets)))
valMat <- array(NA, dim=c(length(allnames), nreps,nparsets), dimnames=list(allnames,seq(nreps), seq(nparsets)))
# if(debug>=3) print(valMat)
psrPlot <- array(NA, dim=c(nreps, preDays))
psrPlot_true <- array(NA, dim=c(nreps, preDays))

stormDates <- c()
obsLength <- c()
obsIntList <- c()

# trueDSRmat <- array(NA, dim=c(150,nreps,nparsets))

# save propInit in same matrix? should be same length
trueDSRmat <- array(NA, dim=c(2,preDays,nreps,nparsets))
# propInitmat <- array(NA, dim=c(preDays,nreps,nparsets))
# dateDSRmat <- array(NA, dim=c(preDays,nreps,nparsets))
modDSRmat <- array(NA, dim=c(nmod+1,preDays,nreps,nparsets))
# print(dim(modDSRmat))
# print(dim(trueDSRmat))

Sys.setenv(printset=FALSE) 
