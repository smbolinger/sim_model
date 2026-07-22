

# mk_true_dsr <- function(nData, modList, preDat, fname, par, config){
mk_true_dsr <- function(nData, modList, preDat, par, config){
  nData <- nData |>
    mutate(Survival = end - init) ## total survival days

  fitData <- nData[rep(1:nrow(nData), times=nData$Survival),]
  fitData$Day <- unlist(lapply(nData$Survival, function(x) seq(1,x))) 
  fitData$Date <- fitData$init + fitData$Day
  fitData <- fitData |> mutate(Surv = ifelse(fate %in% c(1,2) & Date==end, 0, 1))
  ## returns a list:
  out <- lapply(modList, function(x){
                     form <- as.formula(x)
                     modFit <- glm(form, data=fitData, family=binomial)
                     # if(config$debugDSR>=4) qvcalc::indentPrint(modFit,indent=12)
                     predict(modFit, newdata=preDat, type="response")
  })

  # psrOut <- lapply(out, function(x) x ^ par$hatchTime)
  #
  # allInits    <- nData$init 
  # numInit     <- sapply(preDat$Date, function(x) sum(allInits==x))
  # propInit    <- numInit/par$numNests
  # propInitScl <- propInit/sum(propInit)
  # psrScl      <- sapply(psrOut,function(x) sum(propInitScl * x))
  # dsr         <- out[[1]][1]

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(config$debugDSR>=5){
    #   cat(sprintf("\n\t\t>> mk_true_dsr: propInitScl (length=%s):\n\t\t", length(propInitScl)), unlist(propInitScl))
    #   cat(sprintf("\n\t\t>> mk_true_dsr: psrOut (length=%s):\n\t\t", sapply(psrOut,length)))
    #   qvcalc::indentPrint(psrOut)
    #   cat(sprintf("\n\t\t>> mk_true_dsr: psrScl (length=%s):\n\t\t", length(psrScl)), unlist(psrScl))
      cat(sprintf("\n\t\t>> mk_true_dsr: fitData (nrow=%s):\n\t\t", nrow(fitData)))
      qvcalc::indentPrint(fitData)
    }
    if(config$debugDSR>=4){
      if(config$debugDSR<5){
        cat(sprintf("\n\t\t>> mk_true_dsr: 50 rows of fitData (nrow total=%s):\n", nrow(fitData)))
        # cat("\n\t\t>>>> mk_true_dsr:  data for calculating true DSR:\n")
        qvcalc::indentPrint(head(fitData,50), indent=12) # cat("\n\t\t\t> model output:\n") qvcalc::indentPrint(modFit)
      }
      cat("\n\t\t\t>->mk_true_dsr(): max init date:", max(preDat$Date))
      qvcalc::indentPrint(out) # cat("\n\t\t\t> DSR for null model:\n") qvcalc::indentPrint(psrOut)
      cat("\n\t\t\t>->mk_true_dsr(): DSR for all models:\n")
      qvcalc::indentPrint(out) # cat("\n\t\t\t> DSR for null model:\n") qvcalc::indentPrint(psrOut)
      # cat("\n\t\t\t> PSR for all models:\n")
      # qvcalc::indentPrint(psrOut)

    }
    # if(config$debugDSR>=4){
    #   cat("\n\t\t\t>mk_true_dsr: true DSR (null): ", dsr)
    #   cat("\n\t\t\t>mk_true_dsr: true PSR, weighted average: ", unlist(psrScl))
    # }
    # write(psrOut[[2]], file="out/psr_plot.txt", sep="\t", append=TRUE, ncolumns=130)
    # write(psrOut[[2]], file=fname, sep="\t", append=TRUE, ncolumns=130)
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # return(c(dsr, psrScl))
  return(out) ## return list of DSR vals
}

mk_logex_data <- function(nestData,survey,pyconfig,expoVal=0){

  #' take reduced nest data + survey info and pass to python functions to
  #' create df to pass to logex function

  config = py_to_r(pyconfig)
  nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
  nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) #
  # if(config$debugLogEx>=4) qvcalc::indentPrint(head(nestObs))
  ##--- 1. choose dates to pass - different for daily exposure vs. normal ---------------------------------------------------------
  if(expoVal==1){
    #NOTE: should this be up until the final active day of any nest?
    svyDays <- np_array(seq(max(survey[[1]]))) ## better than using as.matrix?
    svyInts <- np_array(rep(1, length(svyDays)))
    ## sum happens inside function; also needs to be all nests, not just discovered:
    # if(config$debugLogEx>=4) cat("exposure=1\n")
    numObs  <- nestData[,"end"] - nestData[,"init"]
    first   <- nestData[,"init"]
    last    <- nestData[,"end"]
    exp1    <- TRUE
    # if(config$debugLogEx>=4) cat("\t\texp1=", exp1)
  } else {
    svyDays <- survey[[1]]
    svyInts <- survey[[2]]
    first   <- nestData[,"i"]
    last    <- nestData[,"k"]
    numObs  <- nestData[, "totobs"]
    exp1    <- FALSE
    # if(config$debugLogEx>=4) cat("\t\texp1=", exp1)
  }
  ##--- 2. get exposure days from survey info -----------------------------------------
  # expoList   <- dsr$calc_daily_expo(numNests=nNest, surveyDays=svyDays,
                                   # surveyInts=svyInts, firstDay=first,
  expoList   <- dsr$calc_daily_expo(numNests=nNest, survey=survey, firstDay=first,
                                   lastDay=last, config=pyconfig)
  # if(dbug>=3) cat("\n\tmaking log exp dataframe\n")
  # if(debug>=3) cat("\n\tpass obs data to make_daily_logex_df:\n")
  # if(debug>=3) qvcalc::indentPrint(head(nestData,30))
  ##--- 3. make a new df w/exposure & covars for glm -----------------------------------
  dat2S <-  withCallingHandlers(
    {dsr$make_daily_logex_df(obsData=nestObs,
                                     nObs=numObs,
                                     survey=survey,
                                     config=pyconfig,
                                     expoList=expoList,
                                     # expos=expoList[[1]],
                                     # covar1=expoList[[2]], # all survey dates for all nests
                                     exp1 = exp1)},
                                     # db=config$debugLogEx) },
    error = function(e) { reticulate::py_last_error() } )
  # return(list(expoList,dat2S))

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(config$debugLogEx>=3){
      if(config$debugLogEx>=4) cat("\t\t>> mk_logex_data: exp1=", exp1)
      cat("\n\t\t\t>> mk_logex_data: >->num obs:", numObs)
      cat(sprintf("\n\t\t\t>> mk_logex_data: >-> survey end days (len %s) & survey ints (len %s) to pass to logexp functions:\n",
                  length(svyDays),length(svyInts)))
      # print(class(svyInts)) 
      # print(class(svyDays))
      # print(py_to_r(svyInts))
      # print(py_to_r(svyInts)[-1])
      # prvec(py_to_r(svyInts), nms=py_to_r(svyDays)[-1])
      prvec(py_to_r(svyInts), nms=py_to_r(svyDays)[-length(svyDays)])
      # qvcalc::indentPrint(svyDays)
      # qvcalc::indentPrint(svyInts)
      # cat("\n\t\t>>> calculating daily exposure\n")
      cat("\n\t\t>> mk_logex_data: |>dat2S:\n")
      qvcalc::indentPrint(head(dat2S,40),indent=12)
    }
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return(dat2S)
}

calc_logexp <- function(modList,dat2S,exp=0,config){

  #' RETURNS: list of model obj from fit_glm, OR 
  #'   "exception" if error, if model did not converge, or if unclear

  excpt <- FALSE
  warn  <- FALSE
  ## take the dataframe made in make_logex_data and pass to glm in fitglm
  ## 08 Jun: change to return glm object and not coefs

  if (exp==1) {modList=modList[c(1,2)]} # withCallingHandlers({ modOut <- fit_glm(modList,dat=dat2S,debug=config$debugLL) }, modOut <- tryCatch({
  ## why is the whole thing also wrapped in trycatch? I guess to get either warning or error as "excpt == TRUE"?
  # fates <- sapply(c(0,1,2), function(x) sum(dat2S$fate==x))
  # fates <- sapply(c(0,1,2), function(x){
  #                   f <- dat2S |> group_by(Nest.ID) |> summarize(fate=first(aFate))
  #                   sum(f$fate==x) })
  # message(sprintf("\t >> fate counts: H:%s|D:%s|Fl:%s\n", fates[1], fates[2], fates[3]))
  tryCatch({
    modOut <- withCallingHandlers({
      fit_glm(modList,dat=dat2S,debug=config$debugLogEx)
    },
      error = function(e) { 
        message("\t!! error in glm:", e) # tryCatch({modOut <- fit_glm}) coefsArray <- -999 coefs[,r,i] <- -999
        # fates <- sapply(c(0,1,2), function(x){
        #                   f <- dat2S |> group_by(Nest.ID) |> summarize(fate=first(aFate))
        #                   sum(f$fate==x) })
        # message(sprintf("\t >> fate counts: H:%s|D:%s|Fl:%s\n", fates[1], fates[2], fates[3]))
      },
      warning = function(w) { # message("!! warning in glm:", w, "go to next") 
        message("\t!! warning in glm:", w) 
        # fates <- sapply(c(0,1,2), function(x) sum(dat2S$fate==x))
        # fates <- sapply(c(0,1,2), function(x){
        #                   f <- dat2S |> group_by(Nest.ID) |> summarize(fate=first(aFate))
        #                   sum(f$fate==x) })
        # message(sprintf("\t >> fate counts: H:%s|D:%s|Fl:%s\n", fates[1], fates[2], fates[3]))
        # if (modOut$converged==FALSE){ excpt <<- TRUE
        warn <<- TRUE
        # coefs[,r,i] <- coefsArray coefs[,r,i] <- -999 coefsArray <- -999
    })
   },

   error=function(e){
      cat("\n\t~~ exception - error ~~")
      excpt <<- TRUE # return("exception")
   })
  if(warn) {
    message("~~ exception - warning but no error ~~") # if(length(modOut>0)){ if(is.list(modOut) & length(modOut>0)){
    tryCatch({
      if (modOut$converged==FALSE){
        cat("\n\t~~ exception - model did not converge ~~")
        return("exception") # conv <- modOut$converged
      }
      },
      error=function(e){
        cat("unclear if converged - go to next")
        excpt <<- TRUE # return("exception")
      })
  } # if (modOut$converged==FALSE){ # message("~~ exception ~~") }
  if(excpt) return("exception")
# #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(config$debugLogEx>=6) cat("\n\t\t\t>-> calc_logexp: modList = ", modList)
    # if(config$debugLogEx>=6) cat(sprintf("\n\t>-> calc_logexp: coefsArray <class:%s> =\n", class(coefsArray)))
    # if(config$debugLogEx>=6) qvcalc::indentPrint(coefsArray)
  }
  # #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(modOut)
}

fit_glm <- function(modList, dat, expoVal = 1, debug=F){

  #' modList contains the formulas for the models
  #' RETURNS: list of model obj

  out <- list()
  for(m in seq_along(modList)){
    vars         <- stringr::str_extract_all(modList[m], "[\\w()^]{2,}")
    vars         <- vars[[1]][-1]
    form <- as.formula(modList[m]) # start <- c(1, rep(0,m-1))
    start <- c(1, rep(0,length(vars))) # out[[m]] <- glm(modList[m], data=dat,
    out[[m]] <- glm(form, data=dat, start=start,
                    family=binomial(link=logexp(dat$Exposure)))

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if(config$testing=="yes"){
    #   if(debug>=5) cat("\n\t\t\tvars: ",paste(vars, collapse=","))
    #   if(debug>=5) cat("\t\tmodel: ",modList[m])
    #   if(debug>=5) cat("\t\tstart: ",start)
    #   if(debug>=6) qvcalc::indentPrint(summary(out[[m]]),indent=8)
    # }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }
    ## move trycatch outside of function so you can skip entire iteration
  return(out)
}

make_preds <- function(coefArr,vcovMat,mod,newDat,db=0){

  #' make predictions manually from equations
  #' RETURNS: nested list of: 1) list of DSR vals; 2) list of SE vals

  vars     <- stringr::str_extract_all(mod, "[\\w()^]{2,}") ## returns a LIST
  # cat("\n\t\t>> make_preds: VARS:", unlist(vars))
  # print(vars)
  vars     <- unlist(vars)[-1]
  betas <- unlist(coefArr)
  # int      <- coefArr[1]
  # if(length(vars)>1){
  #   betas  <- c(coefArr[2],coefArr[3])
  # } else if(length(vars)<1) {
  #   # betas        <- c(0)
  #   betas  <- c()
  # } else {
  #   betas  <- c(coefArr[2])
  # }
  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    # if(db>=3) 
    if(db>=3){
      cat("\n\t>>>make_preds: getting predictions from GLM")
      cat("\n\t\t>> make_preds: MODEL:", mod)
      cat("\t | VARS:", unlist(vars))
      # cat(sprintf("\n\t\t>> make_preds: MODEL: %s ; VARS: %s", mod, unlist(vars)))
    }
    if(db>=5) {
      cat(sprintf("\n\t\t>> make_preds: newDat - first 10 rows (type=%s, nrow=%s)\n", class(newDat), length(newDat)))
      qvcalc::indentPrint(head(newDat,10))
      cat("\n\t\t\t>> make_preds: coefArr: ")
      qvcalc::indentPrint(coefArr)
      cat("\n\t\t\t>> make_preds: vcov matrix: ")
      qvcalc::indentPrint(vcovMat)
      # cat("\n\t\tVARS: ")
      # qvcalc::indentPrint(vars)
    }
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # mod_eq   <- str2expression(make_pr_eq(int,betas,vars,db))
  # se_eq    <- str2expression(make_pr_eq(int,betas,vars,db,outType="se"))
  mod_eq   <- str2expression(make_pr_eq(betas,vars,db))
  se_eq    <- str2expression(make_pr_eq(betas,vars,db,outType="se"))
  dsr_vals <- eval(mod_eq, envir=newDat)
  se_vals  <- eval(se_eq, envir=newDat)
  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(db>=4){
      cat("\n\t\t\t>> make_preds: predictor as expression:")
      withr::with_options( list(width=150), qvcalc::indentPrint(mod_eq))
      cat("\n\t\t\t>> make_preds: standard error as expression:")
      withr::with_options( list(width=120), qvcalc::indentPrint(se_eq))
      cat("\n\t\t\t>> make_preds: vcovMat")
      qvcalc::indentPrint(vcovMat)
      cat("\n\t\t\t>> make_preds: Date")
      qvcalc::indentPrint(newDat[['Date']])
    }
    if(db>=4){
      cat("\n\t\t\t>> make_preds: dsr vals <length=",length(dsr_vals),">")
      qvcalc::indentPrint(dsr_vals)
      cat("\n\t\t\t>> make_preds: se vals <length=",length(se_vals),">")
      qvcalc::indentPrint(se_vals)
    }
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ret <- list(dsr_vals,se_vals)
  return(ret)
}

# make_pr_eq <- function(intercept, betas, pred,db, outType="resp"){
make_pr_eq <- function( allBeta, pred,db, outType="resp"){

  #' make prediction equations 
  #' form: 1/( 1 + exp(- intercept + betas*predvals) )

  if (db>=4) cat("\n\t\t\t>>pass to make_pr_eq: allBeta:", allBeta)
  intercept <- allBeta[1]
  betas <- allBeta[-1]
  if (db>=4) cat("\n\t\t\t>>make_pr_eq: betas=", betas, "intercept=",intercept)
  if(length(betas)<1){
    beta_expand <- 0
    vars_expand <- 0
  }else{
    vars <- paste0("x",seq(2,length(betas)))
    beta_expand <- paste(betas,pred,sep="*")
    vars_expand <- paste(betas,vars,sep="*")
    # cat("\n>> vars=",vars,"\t>> vars_expand=",vars_expand)
  }
  # if(length(betas)>1){
  #   beta_expand <- sapply(betas, function(i) paste(betas[i],x[i],sep="*"))
  # }else{
  # if(betas[1]==0){
  # eq <- sprintf("qlogis(%s + %s)", intercept, paste(beta_expand, collapse="+"))
  if(outType=="resp"){
    eq <- sprintf("1/(1+exp(-(%s + %s)))", intercept, paste(beta_expand, collapse="+"))
    # eta <- sprintf("%s + %s", intercept, paste(beta_expand, collapse="+"))
    # eq <- sprintf("(exp(%s))/(1+exp(%s))", eta, eta)
  } else if(outType=="pred"){
    eq <- sprintf("%s + %s", intercept, paste(beta_expand, collapse="+"))
  } else if(outType=="se"){
    # eq <- sprintf("1/(1+exp(-(%s + %s)))", "x1", paste(vars_expand, collapse="+"))
    # val <- sprintf("c(1,%s)", paste(beta_expand, collapse=","))
    # get_col <- sprintf("x$%s", pred)
    # print(get_col)
    # val <- sprintf("c(1,%s)", paste(pred, collapse=","))
    get_col <- sprintf("x[['%s']]", pred)
    val <- sprintf("c(1,%s)", paste(get_col, collapse=","))
    # # if(beta_expand<1) val <- "c(1)"
    if(length(betas)<1) val <- "c(1)"
    if (db>=4) cat("\n\t\t\t>> val:", val)
    if (db>=4) cat("\n\t\t\t>> t(val):") 
    if (db>=4) print(t(val))
    betaVal <- sprintf("c(%s)", allBeta)
    # val2 <- sprintf("%s*%s",t(val),allBeta)
    # val2 <- sprintf("%s%%*%%%s",(t(val)),c(allBeta))
    # val2 <- sprintf("%s%%*%%%s",(t(val)),betaVal)
    ## the value of "betas" will be inserted when evaluated, i.e. in the outer function?
    val2 <- sprintf("%s%%*%%betas",(t(val)),betaVal)
    # if (db>=4) cat("\n\t\t\t>> val2:", val2, eval(val2))
    if (db>=4) cat("\n\t\t\t>> val2:")
    if (db>=4) print(val2)
    # if (db>=4) cat("\n\t\t\t>> dlogis(val2):") 
    # print(dlogis(eval(val2)))
    # eq <- sprintf("sqrt(t(%s) %%*%% vcovList[[m]] %%*%% %s)", val, val)
    # eq <- sprintf("sapply(dat2S,function(x) sqrt(t(c(1,%s)) %%*%% vcovList[[m]] %%*%% c(1,%s)))", paste(get_col,collapse=","), paste(get_col,collapse=","))
    # eq <- sprintf("sapply(dat2S,function(x) sqrt(t(%s) %%*%% vcovList[[m]] %%*%% %s))", val, val)
    # eq <- sprintf("apply(dat2S,MARGIN=1,FUN=function(x) sqrt(t(%s) %%*%% vcovList[[m]] %%*%% %s))", val, val)
    # eq <- sprintf("apply(dat2S,MARGIN=1,FUN=function(x) sqrt(t(%s) %%*%% vcovMat %%*%% %s))", val, val)
    # eq <- sprintf("apply(newDat,MARGIN=1,FUN=function(x) sqrt(t(%s) %%*%% vcovMat %%*%% %s))", val, val)
    # apply(newDat,MARGIN=1,FUN=function(x){
    #                 print(dlogis(val2))
    #                 print(val)
    #                 print(t(val))
    #                 print(vcovMat)
    #                 sqrt(dlogis(%s) %%*%% t(%s) %%*%% vcovMat %%*%% %s %%*%% dlogis(%s))
    #               } )",val2, val, val,val2)
    eq <- sprintf("apply(newDat,MARGIN=1,FUN=function(x){
                    sqrt(dlogis(%s) %%*%% t(%s) %%*%% vcovMat %%*%% %s %%*%% dlogis(%s))
                  } )",val2, val, val,val2)
    # eq <- sprintf("apply(newDat, MARGIN=1,
    #               FUN=function(x){
    #                 print(%s)
    #                 sqrt(t(%s) %%*%% vcovMat %%*%% %s)
    #               })", val, val,val)
    # eq <- sprintf("apply(newDat,MARGIN=1,FUN=function(x) 1/(1+exp(sqrt(t(%s) %%*%% vcovMat %%*%% %s)))", val, val)
    # eq <- sprintf("apply(newDat,MARGIN=1,FUN=function(x) sqrt(diag(summary(mod)$cov.unscaled)*summary(mod)$dispersion))", val, val)
    # sqrt(diag(summary(model)$cov.unscaled)*summary(model)$dispersion)
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(db>=4) cat(sprintf("\n\t\tpassed to make_pr_eq: intercept:%s betas:%s x:%s\n",intercept, betas,x))
    if(db>=4) cat(sprintf("\n\t\tmake_pr_eq: beta_expand:%s ; vars_expand:%s\n",beta_expand,vars_expand))
    # if(db>=3) cat(sprintf("\n\t\tpass to function: intercept=%s ; betas=%s \n",intercept, paste(betas,x, sep=" ")))
    if(db>=4) cat("\n\t\t\t>> equation:", eq)
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  return(eq)
}

make_weighted <- function(dsrList, seList,allInits, allDates, par, config, type="resp"){
  numInit     <- sapply(allDates, function(x) sum(allInits==x,na.rm=TRUE))
  # numInit     <- sapply(allDates, function(x) {
  #                         cat(sprintf("\n%s:",x))
  #                         qvcalc::indentPrint(allInits==x)
  #                         # print(sum(allInits==x))
  #                         return(sum(allInits==x))
  #                               })
  propInit    <- numInit/par$numNests
  propInitScl <- propInit/sum(propInit,na.rm=TRUE) ## make sure it sums to 1
  # psrList <- lapply(dsrList, function(x) x^par$hatchTime)
  # psr <- lapply(psrList, function(x) sum(x*propInitScl))
  if(type=="resp"){
    psrList <- unlist(dsrList)^par$hatchTime
    # psrList <- unlist(unwtList)^par$hatchTime
    psr <- sum(psrList*propInitScl,na.rm=TRUE)
  } else {
    psr <- c()
  }

  dsr <- sum(dsrList*propInitScl,na.rm=TRUE)
  if(length(seList)>1) serr <- sum(seList*propInitScl,na.rm=TRUE) else serr <- list()
  ## psrList is dsrList ^ hatchTime; prop_nests is proportion of nests initiated on day j
  ## this could either be the true number or some estimate by the observer; for now, stick with the true number
  # psrOut <- lapply(psrList, function(x) sum(x*prop_nests))

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(config$debugLogEx>=5){
  #   #   # cat("\nlength of dsr2:\n", length(dsr2))
      cat(sprintf("\n\t\t>->make_weighted: inits (%s) & dates (%s):\n", class(allInits), class(allDates)))
      qvcalc::indentPrint(allInits, indent=8)
      qvcalc::indentPrint(allDates, indent=8)
      # cat("\nnum inits before date:\n")
      cat("\n\t\t\t>->make_weighted: num inits on date:\n")
      qvcalc::indentPrint(numInit, indent=8)
      cat("\n\t\t\t>-> proportion inits on date:\n")
      qvcalc::indentPrint(propInit, indent=8)
    }
    if(config$debugLogEx>=4){
      cat("\n\t\t\t|>make_weighted: scaled proportion inits on date:\n")
      qvcalc::indentPrint(propInitScl, indent=8)
    }
  #   # if(config$debugDSR>=3) cat("\n\t\t>> calculating weighted PSR")
    if(config$debugLogEx>=4){
      cat("\n\t\t\t|>make_weighted: <input> dsrList:")
      qvcalc::indentPrint(dsrList, indent=8)
      cat("\n\t\t\t|>make_weighted: <input> psrList:")
      qvcalc::indentPrint(psrList, indent=8)
      cat("\n\t\t\t|>make_weighted: <input> seList:")
      qvcalc::indentPrint(seList, indent=8)
      cat("\n\t\t\t|>make_weighted: <output> dsr:")
      qvcalc::indentPrint(dsr, indent=8)
      cat("\n\t\t\t|>make_weighted: <output> psr:")
      qvcalc::indentPrint(psr, indent=8)
      cat("\n\t\t\t|>make_weighted: <output> std err:")
      qvcalc::indentPrint(serr, indent=8)
  #
  #     # cat("\n")
    }
  }
  #   if(config$debugDSR>=4) cat("\n\t\t>> psr (avg psr weighted by nest initiation per day): ", psr, "\n")
  # # if(db>=3){
  # #   cat(sprintf("\n\t>>> calculate for first psr list (lengths= %s, %s):\n",
  # #               length(psrList[[1]]), length(prop_nests)))
  # #   qvcalc::indentPrint(psrList)
  # #   qvcalc::indentPrint(prop_nests)
  # #   # print(sum(psrList[[1]]*prop_nests))
  # #   # return(sum(psrList*prop_nests))
  # #   cat("\n\t|>output of make_psr:\n")
  # #   qvcalc::indentPrint(psrOut)
  # # }
  # }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # return(psrOut)
  # return(psr)
  return(list(dsr,psr,serr))
}

save_current <- function(parStart, parEnd, odir, coef=TRUE, nval=TRUE, suff=""){

  parEnd <- parEnd - 1
  dirname <- sprintf("%s/%s%s",odir,config$rngSeed,atype)
  # dirpath <- sprintf
  cat(sprintf("\n |> saving matrices from par ID %s-%s to %s",parStart,parEnd, dirname))
  if(!dir.exists(dirname)) dir.create(dirname)

  fname <- sprintf("%s/dsr%s%s_%sto%s%s.rds",dirname,config$rngSeed,atype,parStart, parEnd, suff )
  saveRDS(dsrMat, fname)

  if(nval){
    fname <- sprintf("%s/nval%s%s_%sto%s%s.rds",dirname,config$rngSeed,atype,parStart, parEnd, suff )
    saveRDS(nValMat, fname)
  }
  if(coef){
    fname <- sprintf("%s/coef%s%s_%sto%s%s.rds",dirname,config$rngSeed,atype,parStart, parEnd, suff )
    # nvalname <- sprintf("%s/nval%s%s.rds", odir,config$rngSeed,atype)
    saveRDS(coefsMat, fname)
  }
}

logexp <- function(exposure = 1) {

  #' function from Bolker

  get_exposure <- function() { ## hack to help with visualization, post-prediction etc etc
    if (exists("..exposure", env=.GlobalEnv))
      return(get("..exposure", envir=.GlobalEnv))
    exposure
  }
  # cat("\n\t\texposure=", exposure)
  linkfun <- function(mu) qlogis(mu^(1/get_exposure()))
  ## FIXME: is there some trick we can play here to allow
  ##   evaluation in the context of the 'data' argument?
  linkinv <- function(eta) plogis(eta)^get_exposure()
  logit_mu_eta <- function(eta) {
    ifelse(abs(eta)>30,.Machine$double.eps,
           exp(eta)/(1+exp(eta))^2)
  }
  mu.eta <- function(eta) {       
    get_exposure() * plogis(eta)^(get_exposure()-1) *
      logit_mu_eta(eta)
  }
  valideta <- function(eta) TRUE
  link <- paste("logexp(", deparse(substitute(exposure)), ")",
                sep="")
  structure(list(linkfun = linkfun, linkinv = linkinv,
                 mu.eta = mu.eta, valideta = valideta, 
                 name = link),
            class = "link-glm")
}

#-------------------------------------------------------------------------------------------

mk_par_storm_survey <- function(paramsArray, staticPar){

  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  qvcalc::indentPrint(par) # if(debug) print(par$stormFrq)
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur)
  survey    <- withCallingHandlers(
                                   {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=config)},
                                   error=function(e){ 
                                     reticulate::py_last_error() 
                                     # print(sys.calls()) # doesn't help if error in python
                                   }  )
}

## Begin Example 1
## logistic exposure model, following the Example in ?family. See,
## Shaffer, T. 2004. Auk 121(2): 526-540.
# Definition of the link function
logexp_brglm <- function(exposure = 1) {
  get_exposure <- function() {
    if (exists("..exposure", env=.GlobalEnv))
      return(get("..exposure", envir=.GlobalEnv))
    exposure
  }
  linkfun <- function(mu) qlogis(mu^(1/get_exposure()))
  linkinv <- function(eta) plogis(eta)^get_exposure()
  logit_mu_eta <- function(eta) {
    ifelse(abs(eta)>30,.Machine$double.eps,
           exp(eta)/(1+exp(eta))^2)
  }
  mu.eta <- function(eta) get_exposure() * plogis(eta)^(get_exposure()-1) *
    logit_mu_eta(eta)
  # binomial()$mu.eta(eta)
  valideta <- function(eta) TRUE
  link <- paste("logexp(", deparse(substitute(exposure)), ")", sep="")
  structure(list(linkfun = linkfun, linkinv = linkinv,
          mu.eta = mu.eta, valideta = valideta, name = link),
          class = "link-glm")
}

br.custom.family <- function(p) {
  etas <- binomial(logexp(.days))$linkfun(p)
  list(ar=0.5*p/p, # so that to fix the length of ar
  at=0.5+exp(etas)*(1-p)/(2*p*.days))
}

make_dsr_list <- function(dat2S,prDat, survey, mList,par, pyconfig,newList=FALSE,out=NULL){
  config = py_to_r(pyconfig)
  # dat2S <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, expoVal=0) 
  if(is.null(out)){
    out <- calc_logexp(mList,dat2S,config=config)
  }
  if(any(out=="exception")) return("exception")
  coefOut <- sapply(out, function(x) coef(x))
  coefM1 <- coef(out[[1]])
  # vcOut <- sapply(out, function(x) vcov(x)) # get variance-covariance matrix
  vcOut <- lapply(out, function(x) vcov(x)) # get variance-covariance matrix
  # sqrt of diagonal of vcov is SE

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(config$debugLogEx>=6){
      message("\n\t\t>-> make_dsr_list: calc_logexp out:")
      qvcalc::indentPrint(out)
      cat("\n\t\t>-> make_dsr_list: coef - model 1:")
      qvcalc::indentPrint(coefM1)
      cat("\n\t\t>-> make_dsr_list: coef - all:")
      qvcalc::indentPrint(coefOut)
    }
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  dsr1        <- 1/(1+exp(-coefM1[1])) ## coef output is a vcetor?

  psr1        <- dsr1 ^ par$hatchTime
  nmod <- length(mList)
  # dsrList  <- make_pred(coefOut, vcOut,nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
  dsrList  <- make_pred(coefOut, vcOut, nmod, mList, newDat=prDat, hTime=par$hatchTime,
                        newls=newList, db=config$debugLogEx)
  # dsrList  <- make_pred(coefOut, nmod, mList, newDatList=prDat,
                        # hTime=par$hatchTime, db=config$debugLogEx)
  # dsrList  <- dsrList[-1]
  # allInits <- nestData$init
  # numInit     <- sapply(dat2S$Date, function(x) sum(allInits==x))
  # propInit    <- numInit/par$numNests
  # propInitScl <- propInit/sum(propInit) ## make sure it sums to 1
  # dsrScl <- sum(unlist(dsrList)*propInitScl)

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(config$testing=="yes"){
    if(config$debugLogEx>=4){
      cat("\n\t\t>->make_dsr_list: dsrList (head):")
      # qvcalc::indentPrint(dsrList)
      qvcalc::indentPrint(lapply(dsrList, head))
    #   # cat("\nlength of dsr2:\n", length(dsr2))
      # cat("\n\t\t\t\t>-> inits & dates:\n")
      # qvcalc::indentPrint(allInits, indent=8)
      # qvcalc::indentPrint(allDates, indent=8)
      # # cat("\nnum inits before date:\n")
      # cat("\n\t\t\t\t>-> num inits on date:\n")
      # qvcalc::indentPrint(numInit, indent=8)
      # cat("\n\t\t\t\t>-> proportion inits on date:\n")
      # qvcalc::indentPrint(propInit, indent=8)
    }
    # if(config$debugLogEx>=4){
    #   cat("\n\t\t\t\t>-> proportion inits on date:\n")
    #   qvcalc::indentPrint(propInitScl, indent=8)
    #   cat("\n\t\tDSR scaled:")
    #   qvcalc::indentPrint(dsrScl)
    # }
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(dsrList)

}


## do I ever use this??
get_logex <- function(nestData,coefsArray,mList,dat,config){
  cat("\nUSING GET_LOGEX()\n")
  debug = config$debugLogEx
  dsr1 <-  1/(1+exp(-coefsArray[[1]][1,1]))
  psr1 <- dsr1 ^ par$hatchTime
  allInits <- nestData$init
  numInit <- sapply(dat$Date, function(x) sum(allInits==x))
  propInit <- numInit/par$numNests
  propInitScl <- propInit/sum(propInit)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if(config$testing=="yes"){
    # if(debug>=4){
    #   cat("\n\t\tinits & dates:\n")
    #   qvcalc::indentPrint(allInits)
    #   qvcalc::indentPrint(dat2S$Date)
    #   # cat("\nnum inits before date:\n")
    #   cat("\n\t\tnum inits on date:\n")
    #   qvcalc::indentPrint(numInit)
    #   cat("\n\t\tproportion inits on date:\n")
    #   qvcalc::indentPrint(propInit)
    #   qvcalc::indentPrint(propInitScl)
    # }
  # }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
  dsrList <- dsrList[-1]
  psrList <- lapply(dsrList, function(x) x^par$hatchTime)
  if(debug>=4){
    cat("\n\t\toutput of make_pred (dsrList & psrList):\n", length(dsrList), length(psrList))
    qvcalc::indentPrint(dsrList)
    qvcalc::indentPrint(psrList)
  }
  psr <- make_psr(psrList, propInitScl)
}

# make_weighted <- function(unwtList,allInits, allDates, par, config, type="DSR"){
# make_psr <- function(nestData, survey, mList, par, config){
  ## Make weighted PSR values 
  ##  - dsrList = single list of DSR values (not nested)
  ##  - multiply daily PSR by proportion of nests initiated for each day of season  
  ##    > sum of number of nests initiated on day x
  ##    > make into proportion
  ##    > multiply proportion by predicteed PSR for that date (from GLM)
  # if(config$debugLogEx>=4){
  #   cat("\n\t\t\t|>make_weighted: scaled proportion inits on date:\n")
  #   qvcalc::indentPrint(propInitScl, indent=8)
  # }
make_psr_list <- function(nestData, survey, mList, par, pyconfig,expo=0){
  config = py_to_r(pyconfig)
  dat2S <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, expoVal=expo) 
  ## don't need to get CIs for this since you have many rplicates:
  ## i think this is a relic of when I thought I might need to calculate predictions by hand?
  ## also maybe to get around the "hack" for exposure in the log exp function?
  # coefsArray <- calc_logexp(mList,dat2S,config=config)
  #08 Jun: changed output of calc_logexp
  ## can't go to next from within this function
  out <- calc_logexp(mList,dat2S,config=config)
  if(any(out=="exception")) return("exception")
  # print(out)
  # dsr1        <- 1/(1+exp(-coefsArray[[1]][1,1]))
  # nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
  # numObs <- nestData[,"totobs"]
  # if(any(coefsArray=="exception")){
  # if(any(out=="exception")){
  #   cat("  go to next ~~")
  #   #     # coefs[,r,i] <- coefsArray
  #   next
  # }
  # coefOut <- coef(out)
  coefOut <- sapply(out, function(x) coef(x))
  coefM1 <- coef(out[[1]])

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if(config$testing=="yes"){
  #   # cat("\n\t\texpo=",expo)
  #   message("\n\t\tout:")
  #   qvcalc::indentPrint(out)
  #   cat("\n\t\tcoef - model 1:")
  #   qvcalc::indentPrint(coefM1)
  #   cat("\n\t\tcoef - all:")
  #   qvcalc::indentPrint(coefOut)
  # }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  dsr1        <- 1/(1+exp(-coefM1[1])) ## coef output is a vcetor?
  psr1        <- dsr1 ^ par$hatchTime
  
  if(FALSE){
    prAge    <- seq(28)
    newDat <- expand.grid(prDays_true, prAge)
    dsrList2 <- make_pred(out, nmod, mList, newDat=newDat,hTime=par$hatchTime, db=config$debugLogEx)
    dsrList2 <- dsrList2[-1]
    psr_test <- sapply(dsrList2, function(x){
                     make_psr(x, allInits,newDat_true$Date, par, config)
                                  })
    if(config$debugLogEx>=3) cat("\n\t\t>> TEST logistic exposure PSR (avg weighted by inits per day), excl null model: ",class(psr_test), unlist(psr_test), "\n")
  }
  # if(config$coefSave!="none") coefs[,r,i] = unlist(coefsArray)
  # dsrList  <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
  # message("\nmaking DSR list")

  nmod <- length(mList)
  dsrList  <- make_pred(coefOut, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
  dsrList  <- dsrList[-1]
  # print(dsrList)
  # message("\nmaking DSR list 2")
  # dsrList2 <- list()
  # for(m in seq(2,nmod)){# why get rid of the first one when already starting at 2??
  #   dsrList2[[m]] <- predict(out[[m]],newdata=dat2S, type="response")
  # }
  # cat("\t\t>>> *** TEST predictions from GLM:")
  # print(dsrList2)
  allInits <- nestData$init

  # message("\nmaking PSR list ")
  psrList <- sapply(dsrList, function(x){
                   make_psr(x, allInits, dat2S$Date, par, config)
                                })
  ret      <- list(dsr1,psr1,psrList)
  
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if(config$testing=="yes"){
  #   if(config$debugLogEx>=4){
  #     cat("\nDSR list")
  #     print(dsrList)
  #   }
  #   if(config$debug>=4){
  #     cat("\n\t\t>> ret:")
  #     qvcalc::indentPrint(ret)
  #   }
  # }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # return(psrList)
  return(ret)
  }

get_coef <- function(modOut, debug=0){
  if(debug>=3) cat("\t\t>>> getting coefficients from models")
  coefsArray = sapply(modOut, function(x){
                        if(debug>=4) cat("\n\t\t\tcoefs input:\n")
                        if(debug>=4) qvcalc::indentPrint(x,indent=8)
                        # print(coef(x))
                                sapply(seq_along(coef(x)), function(y){
                                # sapply(seq_along(x), function(y){
                                         ## R STILL trying to return conf instead of coef_arr?
                                         # print(y)
                                         if (is.matrix(confint.default(x))){ 
                                            # if (debug>=5) cat(sprintf("\n\t\t\t\t|>%s-coefs&confint:\n",y))
                                            # if (debug>=5) qvcalc::indentPrint(c(coef(x)[y], confint.default(x)[y,]),indent=8)
                                            return(c(coef(x)[y], confint.default(x)[y,]))
                                         } else {
                                            # if (debug>=5) cat(sprintf("\n\t\t\t\t|>%s-coefs&confint:\n",y))
                                            # # if (debug>=4) cat("\n\t\tcoefs&confint:\n")
                                           # if (debug>=5) qvcalc::indentPrint(c(coef(x)[y], confint.default(x)[y]),indent=8)
                                           return(c(coef(x)[y], confint.default(x)[y]))
                                         }
                                         })
                 })
  if (debug>=4) cat("\n\t\t\tcoefs output:\n")
  if (debug>=4) qvcalc::indentPrint(coefsArray)
  return(coefsArray)
}

make_pred_se <- function(){
  # vcMod <- vcov(modFit)
  ## standard error on predictor scale should be sqrt(betas*coefs * vcov * newdata)?
  modSE <- sqrt(vcMod * pred)
  return(modSE)
}

make_pr_data <- function(prDays,dat){

  #' RETURNS: list of new data, one df for each model

  newDat1 <- expand.grid(Date=prDays,
  # newDat1 <- expand.grid(Date=seq(1,preDays,by=5),
                         Age=mean(dat2S$Age),
                         # Age=seq(1,par$hatchTime,by=byVal),
                         avDate=mean(dat2S$avDate)) ## null
  newDat2 <- expand.grid(Date=prDays,
                         # Age=seq(1,par$hatchTime,by=byVal),
                         Age=mean(dat2S$Age),
                         avDate=mean(dat2S$avDate)) ## date-only
  # newDat3 <- expand.grid(Date=seq(1,preDays,by=5),
  # newDat3 <- expand.grid(Date=mean(dat2S$Date),
  newDat3 <- expand.grid(Date=prDays,
                         Age=mean(dat2S$Age),
                         # Age=seq(1,par$hatchTime), ## don't have to include all these
                         avDate=mean(dat2S$avDate)) ## age-only
  newDat4 <- expand.grid(Date=seq(1,preDays),
                         Age=mean(dat2S$Age),
                         # Age=seq(1,par$hatchTime),
                         avDate=mean(dat2S$avDate)) ## age+date
  # newDat5 <- expand.grid(Date=mean(dat2S$Date),
  newDat5 <- expand.grid(Date=prDays,
                         Age=mean(dat2S$Age),
                         avDate=mean(dat2S$avDate)) ## age+date
                         # avDate=prDays) ## age+date
                         # avDate=seq(1,preDays,by=10)) ## age+date
  newList <- list(newDat1,newDat2,newDat3,newDat4,newDat5)
  return(newList)

}

make_pred <- function(coefArr,vcovList,nmod,mods,newDat,hTime,newls=FALSE,db=0){

  #' make predictions manually from equations

  dsrList <- list()
  # if (db>=4) cat("\n\t>> make_pred: newDat\n", class(newDat))
  # if (db>=4) qvcalc::indentPrint(newDat)
  ## mods is just the model names!
  # vcMod <- lapply(mods, vcov)
  # nmod=length(mods)
  ## already calculated for constant model in the main script
  if(db>=3) cat("\t\t>>>make_pred: getting predictions from GLM")
  # for(m in seq(2,nmod)){# why get rid of the first one when already starting at 2??
  for(m in seq(1,nmod)){# why get rid of the first one when already starting at 2??

    if (newls) newDat <- newDat[[m]]
    if (db>=4) cat("\n\t>> make_pred: newDat", class(newDat), length(newDat),"\n")
    if (db>=4) qvcalc::indentPrint(head(newDat,30))
    vars         <- stringr::str_extract_all(mods[m], "[\\w()^]{2,}")
    vars         <- vars[[1]][-1]
    # cat("\nvars=",vars)
    # int          <- coefArr[[m]][1,1]
    int          <- coefArr[[m]][1]
    # print(length(vars))
    if(length(vars)>1){
      betas        <- c(coefArr[[m]][2],coefArr[[m]][3])
    } else if(length(vars)<1) {
      # betas        <- c(0)
      betas        <- c()
    } else {
      betas        <- c(coefArr[[m]][2])
    }
    ## response-scale:
    mod_eq       <- str2expression(make_pr_eq(int,betas,vars,db))
    ## predictor-scale:
    # lin_pr       <- str2expression(make_pr_eq(int,beta,vars,db,scale="pred"))
    # dsrList[[m]] <- eval(mod_eq, envir=newDat)
    dsr_vals <- eval(mod_eq, envir=newDat)
    # Date = newDat$Date avDate = newDat$avDate Age = newDat$Age

    # se_vals <- msm::deltamethod(mod_eq, mean = coefArr[[m]], cov = vcovList[[m]])
    # print(se_vals)
    ## standard errors:
    # se_vals      <- t(coefArr[[m]]) %*% vcovList[[m]] %*% coefArr[[m]]
    # se_vals <- sapply()
    se_eq       <- str2expression(make_pr_eq(int,betas,vars,db,outType="se"))
    se_vals <- eval(se_eq, envir=newDat)
    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(config$testing=="yes"){
      if(db>=3){
        cat(sprintf("\n\t\t\t>> make_pred: MODEL %s: %s ; VARS: %s",m, mods[m], unlist(vars)))
      }
      if(db>=6) {
        cat("\n\t\t\t>> make_pred: coefArr: ")
        qvcalc::indentPrint(coefArr[[m]])
        cat("\n\t\t\t>> make_pred: vcov matrix: ")
        qvcalc::indentPrint(vcovList[[m]])
        # cat("\n\t\tVARS: ")
        # qvcalc::indentPrint(vars)
      }
      if(db>=5){
        cat("\n\t\t\t>> make_pred: predictor as expression:")
        withr::with_options( list(width=120), qvcalc::indentPrint(mod_eq))
        # cat("\n\t\t\t>> make_pred: standard error as expression:")
        # withr::with_options( list(width=120), qvcalc::indentPrint(se_eq))
      }
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    dsrList[[m]] <- list(dsr_vals, se_vals)

    ## standard error on predictor scale should be sqrt(betas*coefs * vcov * newdata)?
    # seList[[m]] <- dsrList[[m]] * vcMod[[m]]
    # if(db>=3) qvcalc::indentPrint (dsrList[[m]])
    # dsrList[[m]] <- plogis(mod_eq[[m]])
    # psrList[[m]] <- dsrList[[m]]^hTime
  }
  # psrList <- sapply(dsrList, function(x) x^hTime)
  return(dsrList)
}

