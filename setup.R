
startTime <- Sys.time()
library(reticulate)
library(MASS)
library(brglm2)
library(tidyr)
suppressPackageStartupMessages(library(dplyr)) # load dplyr last so as not to mask select?

source("/home/wodehouse/.local/bin/r_func.R")
options(width=1000, digits=5, scipen=999)
py_list <- import_builtins()$list
py_print <- import_builtins()$print
py_vars <- import_builtins()$vars ## function to turn class instance into dictionary
py_attr <- import_builtins()$setattr ## function to turn class instance into dictionary

vals <- list(
  "homeDir" = "/home/wodehouse/Projects/sim_model",
  "obsVar" = "totobs",
  "nowstr"  = format(Sys.time(), "%Y%m%d"),
  "initFromFile" = TRUE,
  "stormFromFile" = TRUE,
  "psrTrueVal" = "t",
  "dsrTrueVal" = "date", #mc_file = "MCmatrix"
  "uniqueStorm"=TRUE, # "uniqueStorm"=FALSE,
  "preDays" = 120
)

tests     <- c('norm','range','test2','small','small2','xtrastorm',
               'test100', 'testmaxmin','testctrl','fixedtest','nstest1',
               'nstest2','nostormtest', 'test500','setn','snrange')
fullList  <- c('nostorm','full','supp','subset')
ctrlList  <- c('control','ctlstorm')
rangeList <- c('range','snrange')

# parser <- optparse::OptionParser() |> add_option()
# arg <- unlist(commandArgs(trailingOnly=TRUE))
# NOTE: there are some short flags that are reserved for R's use
spec <- matrix(c(
                 'nruns', 'R', 1, "integer",
                 'rng', 'S', 1, "integer",
                 'par', 'P', 1, "integer",
                 'nnest', 'N', 1, "integer",
                 'db', 'D', 1, "integer",
                 'nomc', 'C', 0, "logical",
                 'nolx', 'X', 0, "logical",
                 'savend', 'E', 0, "logical", ## save nest data to file
                 # 'nodsr', 'U', 0, "logical",   ## don't save DSR values to file
                 'pred', 'U', 2, "logical", ## 2 means default is TRUE
                 'msg', 'M', 1, "character",
                 'atype', 'A', 1, "character",
                 # 'at', 'A', 1, "character",
                 'oval', 'O', 1, "character"
                 ), byrow = TRUE, ncol = 4L)

# print(spec)
cat("\n*.*.*.*.*.*.*.*.*.*.SETUP.*.*.*.*.*.*.*.*.*")
opt <- getopt::getopt(spec) # print(opt)
opt$atype <- gsub("^['\"]|['\"]$", "", opt$atype)
othVal <- ifelse(is.null(opt$oval), "", opt$oval)
cat("\n\t>> other vals =", othVal)
atype <- ifelse(is.null(opt$atype), "default", opt$atype) # atype <- ifelse(is.null(opt$at), "default", opt$at)
cat("\t>> atype =", atype)
cat("\tatype class:", class(atype))

py_imports <- matrix(c(
                       "np", "numpy", TRUE,
                       "obs", "observer", TRUE,
                       "nest", "makeNests", TRUE,
                       "dsr", "dsrCalc", TRUE,
                       "mlfun", "matlab_func", TRUE,
                       "printFun", "print_func", TRUE,
                       "funs", "helpers", TRUE,
                       "pyfuns", "helpers", FALSE,
                       "allParList", "paramLists", FALSE
                 ), byrow = TRUE, ncol = 3L)

cat("\n\t>> importing python functions \t")
# print(nrow(py_imports))
for (r in seq(nrow(py_imports))){
  # print(dim(py_imports))
  modName <- py_imports[r,2]
  rName <- py_imports[r,1]
  conv <- as.logical(py_imports[r,3])
  assign(rName, import(modName, convert=conv))
  cat(sprintf("\t>>> imported %s as %s, convert=%s", modName, rName, conv))
}

## unpack param lists
if(TRUE){
  parLists <- allParList$parLists
  plTest <- allParList$plTest
  plControl <- allParList$plControl
  plTestRange <- allParList$plTestRange
  plNoStorm <- allParList$plNoStorm
  plTest2 <- allParList$plTest2
  plCtlTest <- allParList$plCtlTest
  staticPar <- allParList$staticPar

}

inits = c(4,74,67,48,51,33,42,41,34,28,36,10,7)
cat("\n\t> class(inits)=", class(inits))
weeks = np$arange(3,15,1)
weeks=np$arange(0,13,1)
# initProb = list(inits / np$sum(inits)) # make them into probabilities again
initProb = r_to_py(inits / np$sum(inits)) # make them into probabilities again
cat("\t> class(initProb)=", class(inits))
initWeek = weeks * 7
initWeek = r_to_py(as.integer(initWeek))
cat("\t> class(initWeek)=", class(initWeek))
# initDat  = py_list(c(initProb, initWeek))
initDat  = r_to_py(list(initProb, initWeek)) ## should become python list automatically
# py_print(f"{class(initDat)=}")
cat("\t> class(initDat)=", class(initDat))

stormNum  = c(1,4,5,3,3,4,4,5,5,2,9,4,2)
stormProb = r_to_py(stormNum/np$sum(stormNum))
stormWeek = np$arange(1,14,1)
cat("storm weeks:", stormWeek)
weekStart = stormWeek*7
weekStart = r_to_py(as.integer(weekStart))
stormDat  = r_to_py(list(stormProb, weekStart))
# py_print(f"{class(stormDat)=}")
cat("\t> class(stormDat)=", class(stormDat))

cat("\n\t>> importing config ")
  # if(config$debug>=5) cat(sprintf("\n\t|> importing pyconfig <%s> & config <%s>\n", class(pyconfig)[1], class(config)[1]))
configOut <- pyfuns$choose_config(atype,tests,ctrlList,fullList)
# print(configOut)
# print(class(configOut))
# pyconfig <- configOut[1]
pyconfig <- py_get_item(configOut,0L)
cat("\t>-> pyconfig - class:", class(pyconfig))
config <- py_to_r(pyconfig)
cat("\t>-> config - class:", class(config))
# configMsg <- configOut[2]
configMsg <- py_get_item(configOut,1L)
# print(configMsg)
  # if(debug!=""){
    # config$testing="yes"
    # cat("\t>> changing config$testing to ", config$testing)
    #  # config$debug = debug else debug = config$debug
  # }

debugVals <- grep("debug",names(py_vars(config)),fixed=TRUE)
debugNames = names(py_vars(config))[debugVals]
debug = ifelse(is.null(opt$db), config$debug, opt$db)
# cat("\ndebug value:", debug)
for(x in debugNames) if(as.numeric(config[[x]])>=as.numeric(debug)) py_attr(config,x,debug)
if (as.numeric(debug)>=4){
  cat("\n\t>> debug =",debug,"type=",class(debug))
  cat("\t\t>> config$debug =",config$debug,"type=",class(config$debug))
  cat("\t\tdebug vals & types:")
  for(x in debugNames) cat(sprintf("%s",debugNames[x]),config[[x]],class(config[[x]]))
}
debug <- as.numeric(debug)

saveNestData <- ifelse(!is.null(opt$savend), opt$savend,config$saveNData)
startParID   <- ifelse(!is.null(opt$par),as.integer(opt$par), as.integer(config$startParID))
rngSeed <- ifelse(!is.null(opt$rng), as.integer(opt$rng),as.integer(config$rngSeed))
rng <- np$random$default_rng(seed=rngSeed)
nreps        <- ifelse(!is.null(opt$nruns), as.integer(opt$nruns),config$nreps)

if (!is.null(opt$nomc)) config$mcmc <- FALSE
if (!is.null(opt$nolx)) config$logex <- FALSE
# if (!is.null(opt$nodsr)) config$survSave <- "trueVal"
if (isTRUE(opt$pred)) config$predict <- TRUE
if (!is.null(opt$msg)) config$msg <- paste(config$msg, opt$msg, sep=";")

cat("\n\tCONFIG")
if(nreps>20) cat(" [ !! NOTE: large # of reps - force debug values low unless overridden by CL arg] ")

cat(":\n")

withr::with_options( list(width=120), print(unlist( py_vars(pyconfig) )) )
# withr::with_options( list(width=120), print(unlist( py_vars(py_to_r(pyconfig)) )) )
if(config$testing=="yes") library(ggplot2)

like_f_dir = "/home/wodehouse/Dropbox/Models/ch2_analysis/py_out"
odir = sprintf("%s/%s",like_f_dir,vals$nowstr)
if (!dir.exists(odir)) {
  dir.create(odir, recursive = TRUE)
}
cat("\t\t>> importing param lists \n")
pListOut <- pyfuns$choose_parlist(atype, config, otherVal=othVal,debug=TRUE)
pLists <- py_get_item(pListOut,0L) # pLists <- pListOut[1]
pListMsg <- py_get_item(pListOut,1L) # pListMsg <- pListOut[2]
vary <- py_to_r(py_get_item(pListOut,2L)) # vary <- pListOut[3]

suff <- sprintf("%s%s", rngSeed, atype) # cat("\n\t>> suff:", suff) # cat("\t>> params array \n\t")
paramsArray = pyfuns$mk_param_list_list(parL=pLists,pStatic=staticPar, fdir=odir, suf=suff, debug=FALSE) # cat("\t>> params list \n\t")
pArrList = pyfuns$mk_param_list_list(parL=pLists,pStatic=staticPar,fdir=odir,suf=suff,debug=FALSE, listRet=TRUE)
# print_settings(config, atype, initFromFile, paramsArray, pListMsg, confMsg )

dirName <- sprintf("%s%s_inc", rngSeed, atype) # homedir <- "/home/wodehouse/Projects/sim_model"
outdir <- file.path(odir,dirName)

obsVarNum = ifelse(vals$obsVar=="nobs", 8, 10)
stormUnk <- as.integer(py_to_r(staticPar$stormUnk))
breDays <- py_to_r(staticPar$brDays)
brDays <- seq(breDays)
prDays <- seq(vals$preDays)
colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs', 'sint')
nparsets <- length(pArrList)

##--------------- MODEL LISTS: ----------------------------------------------

mList <- c("Surv~1", "Surv~Date", "Surv~Age", "Surv~Age+Date", "Surv~avDate")
mList_supp <- c("Surv~avDate","Surv~avAge","Surv~avAge+avDate")

mList_true <- c("Surv~1", "Surv~Date")                      ## models to fit
# mList_true <- c("Surv~1", "Surv~Date", "Surv~Date+Age")                      ## models to fit
# mNames <- c("m1", "m2", "m3", "m4","m5")

mNames <- c("m1", "m2", "m3", "m4")
nmod <- length(mList)
nmod_true <- length(mList_true)

# print(length(colnames))
cat("\n\t|> overall output directory:", odir)
cat("\t>> seed output directory:", outdir)
if (TRUE){
  if(config$msg!="") cat("\n\t|>MSG: ", config$msg)
  cat("\n\tOTHER:")
  cat("\n\t\t|>debug = ", debug)
  cat("\tvary=", vary)
  cat("\t\t>>| save nest data:",saveNestData)
  cat(sprintf("\t\t| discovered nests = where %s > 0", vals$obsVar))
  cat("\n\tmList = ", mList)
  cat("\t| explicitly mark storm nests 'unknown'?", stormUnk)
  cat(sprintf("\n\t\ttrue number of reps: %s",nreps))
  cat(sprintf("\t\t| num param sets: %s",nparsets))
  cat(sprintf("\t\t| starting param set: %s",startParID))
  cat(sprintf("\t\t| starting rngSeed: %s",rngSeed))
  cat("\n\t\t>> unique storm days for each replicate? ", vals$uniqueStorm)
  cat(sprintf("\t\t| number of days for prediction: %s <%s>",vals$preDays,class(vals$preDays)))
  # cat("\n\t>> breeding days:", brDays)
  # cat("\n\t>> prediction days:", prDays)
  cat("\n\t\t>> check modules imported to datsim.py:")
}
cat("\n\tCREATE ARRAYS FOR STORAGE")
source("/home/wodehouse/Projects/sim_model/array_setup.R")
cat(sprintf("\n|>|> %s reps x %s param sets = %s rows", nreps, nparsets, nreps*nparsets))

if(config$obsSave){
  file.create("out/psr_plot.txt")
  file.create("out/dsr_plot.txt")
  psr_head <- c("htime", "stormfrq", "stormdur", "flmort", "psurv", paste("day",prDays))
  print(psr_head)
  write(psr_head, file="out/dsr_plot.txt", sep="\t", append=TRUE, ncolumns=130)

  file.create("out/storm_plot.txt")
  file.create("out/init_plot.txt")
}
