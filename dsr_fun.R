
mk_true_dsr <- function(nData, modList, preDat, par, config){
  ## returns output from predict.glm for each model
  nData <- nData |>
    mutate(Survival = end - init) ## total survival days

  
  fitData <- nData[rep(1:nrow(nData), times=nData$Survival),]
  if(config$debugDSR>=3){
    cat("\n\t>>mk_true_dsr: nrow nData=",nrow(nData),"nrow fitData=",nrow(fitData))
    qvcalc::indentPrint(cumsum(nData$Survival))
    cat("\n\t>>mk_true_dsr: nData columns:", names(nData))
    cat("\n\t>>mk_true_dsr: nData=")
    qvcalc::indentPrint(head(nData,30))

  }
  # if(config$debugDSR>=3){
  #   cat("\n\t>>mk_true_dsr: fitData=")
  #   qvcalc::indentPrint(fitData)
  # }
  # fitData$Day <- unlist(lapply(nData$Survival, function(x) seq(1,x))) 
  fitData$Age <- unlist(lapply(nData$Survival, function(x) seq(1,x))) 
  # if(config$debugDSR>=3){
  #   cat("\n\t>>mk_true_dsr: fitData w/Day=")
  #   qvcalc::indentPrint(fitData)
  # }
  fitData$Date <- fitData$init + fitData$Age
  fitData <- fitData |> mutate(Surv = ifelse(fate %in% c(1,2) & Date==end, 0, 1))
  ## returns a list:
  if(config$debugDSR>=4){
    cat(sprintf("\n\t\t>> mk_true_dsr: fitData (nrow=%s):\n\t\t", nrow(fitData)))
    qvcalc::indentPrint(head(fitData,30))
  }
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
    if(config$debugDSR>=4){
    #   cat(sprintf("\n\t\t>> mk_true_dsr: propInitScl (length=%s):\n\t\t", length(propInitScl)), unlist(propInitScl))
    #   cat(sprintf("\n\t\t>> mk_true_dsr: psrOut (length=%s):\n\t\t", sapply(psrOut,length)))
    #   qvcalc::indentPrint(psrOut)
    #   cat(sprintf("\n\t\t>> mk_true_dsr: psrScl (length=%s):\n\t\t", length(psrScl)), unlist(psrScl))
    }
    if(config$debugDSR>=4){
      if(config$debugDSR<5){
        cat(sprintf("\n\t\t>> mk_true_dsr: 50 rows of fitData (nrow total=%s):\n", nrow(fitData)))
        # cat("\n\t\t>>>> mk_true_dsr:  data for calculating true DSR:\n")
        qvcalc::indentPrint(head(fitData,50), indent=12) # cat("\n\t\t\t> model output:\n") qvcalc::indentPrint(modFit)
      }
      cat("\n\t\t\t>->mk_true_dsr(): max init date:", max(preDat$Date))
      # qvcalc::indentPrint(out) # cat("\n\t\t\t> DSR for null model:\n") qvcalc::indentPrint(psrOut)
      cat("\n\t\t\t>->mk_true_dsr(): DSR for all models:\n")
      qvcalc::indentPrint(out) # cat("\n\t\t\t> DSR for null model:\n") qvcalc::indentPrint(psrOut)
      # cat("\n\t\t\t> PSR for all models:\n")
      # qvcalc::indentPrint(psrOut)

    }
    # if(config$debugDSR>=3){
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
  # print(length(nestObs))
  expoList   <- dsr$calc_daily_expo(numNests=nNest, survey=survey, firstDay=first,
                                   lastDay=last, config=pyconfig)
  # print(length(expoList))
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

make_preds <- function(coefArr,vcovMat,mod,newDat,se=TRUE,db=0){
  

  #' make predictions manually from equations
  #' RETURNS: nested list of: 1) list of DSR vals; 2) list of SE vals
  ##  ADD? if se==TRUE, returns two lists; if FALSE, just returns DSR list

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
    if(db>=1){
      cat("\n\t\t>>>make_preds: getting predictions from GLM")
      # cat("\n\t\t\t>> make_preds: MODEL:", mod)
      cat("\t>> make_preds: MODEL:", mod)
      cat("\t | VARS:", paste(unlist(vars), collapse=" "))
      # cat("\t>> make_preds: coefArr: ", paste(unlist(coefArr), collapse=" "))
      cat("\t>> make_preds: coefArr: ", paste(unlist(coefArr), collapse=" "))
      # cat("\n")
      # qvcalc::indentPrint(coefArr)
      # qvcalc::indentPrint(class(coefArr))
      # cat(sprintf("\n\t\t>> make_preds: MODEL: %s ; VARS: %s", mod, unlist(vars)))
    }
    if(db>=3) {
      cat(sprintf("\n\t\t>> make_preds: newDat - first 10 rows (type=%s, nrow=%s)\n", class(newDat), length(newDat)))
      qvcalc::indentPrint(head(newDat,10))
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
    if(db>=2){
      cat("\n\t\t\t>> make_preds: predictor as expression:")
      withr::with_options( list(width=150), qvcalc::indentPrint(mod_eq))
      cat("\n\t\t\t>> make_preds: standard error as expression:")
      withr::with_options( list(width=120), qvcalc::indentPrint(se_eq))
    }
    if(db>=3){
      cat("\n\t\t\t>> make_preds: vcovMat")
      qvcalc::indentPrint(vcovMat)
      cat("\n\t\t\t>> make_preds: Date")
      qvcalc::indentPrint(newDat[['Date']])
    }
    if(db>=2){
      cat("\n\t\t\t>> make_preds-output: dsr vals <length=",length(dsr_vals),">")
      qvcalc::indentPrint(head(dsr_vals, 20))
      cat("\n\t\t\t>> make_preds-output: se vals <length=",length(se_vals),">")
      qvcalc::indentPrint(head(se_vals, 20))
    }
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ret <- list(dsr_vals,se_vals)
  return(ret)
}

make_pr_eq <- function( allBeta, pred,db, outType="resp"){

  #' make prediction equations 
  #' form: 1/( 1 + exp(- intercept + betas*predvals) )

  if (db>=3) cat("\n\t\t\t>>pass to make_pr_eq: allBeta:", allBeta)
  intercept <- allBeta[1]
  betas <- allBeta[-1]
  if (db>=3) cat("\n\t\t\t>>make_pr_eq: betas=", betas, "intercept=",intercept)
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
    # if(db>=4) cat(sprintf("\n\t\tpassed to make_pr_eq: intercept:%s betas:%s x:%s\n",intercept, betas,x))
    # if(db>=4) cat(sprintf("\n\t\tmake_pr_eq: beta_expand:%s ; vars_expand:%s\n",beta_expand,vars_expand))
    # if(db>=3) cat(sprintf("\n\t\tpass to function: intercept=%s ; betas=%s \n",intercept, paste(betas,x, sep=" ")))
    if(db>=4) cat("\n\t\t\t>> equation:", eq)
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  return(eq)
}

make_weighted <- function(mod,dsrList,seList,allInits,allDates,par,config,newDat=NULL,propInitScl=NULL,outtype="resp"){
  if(config$testing=="yes"){
    if(config$debugLogEx>=2){
      cat("\n  |>make_weighted: model:", mod)
      # cat("\n\t\t\t|>make_weighted: <input> dsrList:")
      # qvcalc::indentPrint(head(dsrList, 5), indent=8)
      # qvcalc::indentPrint(tail(dsrList, 15), indent=8)
      cat("\n\t<input> dsrList:")
      cat( paste(unlist(head(dsrList, 5)), collapse=" "))
      cat(". . . . ")
      cat( paste(unlist(tail(dsrList, 5)), collapse=" "))
      cat("\tlength=", length(dsrList))
      # cat("\n\t\t\t|>make_weighted: <input> psrList:")
      # qvcalc::indentPrint(psrList, indent=8)
      cat("\n\t<input> seList:")
      cat( paste(unlist(head(seList, 5)), collapse=" "))
      # qvcalc::indentPrint(head(seList, 5), indent=8)
      cat(". . . . ")
      # qvcalc::indentPrint(tail(seList, 15), indent=8)
      cat( paste(unlist(tail(seList, 5)), collapse=" "))
      cat("\tlength=", length(seList))
    }
  }
  if(is.null(propInitScl)){
    numInit     <- sapply(allDates, function(x) sum(allInits==x,na.rm=TRUE))
    propInit    <- numInit/par$numNests
    propInitScl <- propInit/sum(propInit,na.rm=TRUE) ## make sure it sums to 1
  }
  vars     <- unlist(stringr::str_extract_all(mod, "[\\w()^]{2,}")) ## returns a LIST
  if(config$debugLogEx>=2) cat("\n\t\t\tvars=", vars, "length=", length(vars))

  if(length(vars)<2){
    if(config$debugLogEx>=2) cat("\n\t\t\tnull model - not weighting")
    if(config$debugLogEx>=2) cat("\tdsrList:")
    if(config$debugLogEx>=2) print(dsrList)
    dsr <- dsrList[1]
    # print()
    # cat(sprintf("\n dsr: %s (%s)", dsr, class(dsr)))
    psr <- dsr ^ as.numeric(par$hatchTime)
    serr <- seList[1]
    # cat(sprintf("// psr: %s // se: %s", psr, serr))
  } else if(length(vars)==2 & vars[2]=="Age"){

    # dsr <- sum(dsrList*propInitScl,na.rm=TRUE)
    ## date-specific
    # dsr <- dsrList*propInitScl

    if(config$debugLogEx>=2) cat("\n\t\t\tweighting by age")
    # if(config$debugLogEx>=2) cat("\tdsrList:")
    # if(config$debugLogEx>=2) print(dsrList)
    ## where does "dsr" come from?
    dsr <- mean(dsrList)
    psr <- prod(dsrList)

    ## this standard error is much too small
    if(length(seList)>1) serr <- mean(seList) else serr <- list()
    
  }else if(length(vars)>2){
      # this should only target age+date model
      # df <- cbind(newDat, dsrList)
      # df <- data.frame(Age=newDat$Age, dsr=dsrList)
      if(config$debugLogEx>=2) cat("\n\t\t\tweighting by age+date")
      # df <- data.frame(Date=newDat$Date, dsr=dsrList)
      if(is.null(newDat)){
        df <- data.frame(Date=rep(prDays, times=as.numeric(parVals['hatch_time']), dsr=dsrList))
        # cat("\nprediction dates:", df$Date, length(df$Date))
      }else{
        df <- data.frame(Date=newDat$Date, dsr=dsrList, se=seList)
      }
      # dsrList <- lapply(split(df, df$Date), function(x) prod(x$dsr))
      # first get product of all age estimates for each value of Date
      if(config$debugLogEx>=2) cat("\t >> get mean DSR across all nest ages for each value of date")
      dsrList <- sapply(split(df, df$Date), function(x) mean(x$dsr))
      dsr <- sum(dsrList*propInitScl,na.rm=TRUE)
      # if(config$debugLogEx>=2) cat("\t output:\n")
      if(config$debugLogEx>=2){
        cat( paste(unlist(head(dsrList, 5)), collapse=" "))
        cat(". . . . ")
        cat( paste(unlist(tail(dsrList, 5)), collapse=" "))
      }
      if(config$debugLogEx>=2) cat("\t >> get age-weighted PSR for each value of date")
      psrList <- sapply(split(df, df$Date), function(x) prod(x$dsr))

      if(config$debugLogEx>=2){
        cat( paste(unlist(head(psrList, 5)), collapse=" "))
        cat(". . . . ")
        cat( paste(unlist(tail(psrList, 5)), collapse=" "))
      }
      psr <- sum(psrList*propInitScl,na.rm=TRUE)
      if(config$debugLogEx>=2) cat("\t >> get mean standard error for each value of date")
      if(length(seList)>1){
        # df <- data.frame(Date=newDat$Date, se=seList)
        # seList <- sapply(split(df, df$Date), function(x) prod(x$se))
        seList <- sapply(split(df, df$Date), function(x) mean(x$se))
        serr <- sum(seList*propInitScl,na.rm=TRUE)
      }else{
        serr <- list()
      }
      if(config$debugLogEx>=2){
        cat("\n\t\t\t>> psrList for age+date model, after grouping by age and taking product:")
        qvcalc::indentPrint(psrList)
        cat("\tlength=", length(psrList))
      }

  } else {

    # numInit     <- sapply(allDates, function(x) sum(allInits==x,na.rm=TRUE))
    # propInit    <- numInit/par$numNests
    # propInitScl <- propInit/sum(propInit,na.rm=TRUE) ## make sure it sums to 1
    # psrList <- lapply(dsrList, function(x) x^par$hatchTime)
    # psr <- lapply(psrList, function(x) sum(x*propInitScl))
    if(config$debugLogEx>=2) cat("\n\t\t\tweighting by date")

    if(outtype=="resp"){
      # psrList <- unlist(unwtList)^par$hatchTime
      ## why did it not make me add "as.numeric" when I was running it in the all_dsr script?
      if(config$debugLogEx>=2) cat("\t >> get date-weighted PSR")
      psrList <- unlist(dsrList)^as.numeric(par$hatchTime)
      if(config$debugLogEx>=2) {
        cat( paste(unlist(head(psrList, 5)), collapse=" "))
        cat(". . . . ")
        cat( paste(unlist(tail(psrList, 5)), collapse=" "))
      }
      psr <- sum(psrList*propInitScl,na.rm=TRUE)
    } else {
      psr <- c()
    }

    if(config$debugLogEx>=2) cat("\t >> get date-weighted DSR")
    dsr <- sum(dsrList*propInitScl,na.rm=TRUE)
    if(length(seList)>1) serr <- sum(seList*propInitScl,na.rm=TRUE) else serr <- list()

    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(config$testing=="yes"){
      if(config$debugLogEx>=2){
        # cat( "DSR at average date:", )
        cat("\t\t\t>>> compare PSR values:", psr, dsr^as.numeric(par$hatchTime))
      }
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
      if(config$debugLogEx>=3){
        cat("\n\t\t\t|>make_weighted: scaled proportion inits on date:\n")
        qvcalc::indentPrint(propInitScl, indent=8)
      }
    }

  }
  # numInit     <- sapply(allDates, function(x) {
  #                         cat(sprintf("\n%s:",x))
  #                         qvcalc::indentPrint(allInits==x)
  #                         # print(sum(allInits==x))
  #                         return(sum(allInits==x))
  #                               })
  ## psrList is dsrList ^ hatchTime; prop_nests is proportion of nests initiated on day j
  ## this could either be the true number or some estimate by the observer; for now, stick with the true number
  # psrOut <- lapply(psrList, function(x) sum(x*prop_nests))

  #   # if(config$debugDSR>=3) cat("\n\t\t>> calculating weighted PSR")

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(config$debugLogEx>=3){
      # cat("\n\t\t\t|>make_weighted: <input> dsrList:")
      # qvcalc::indentPrint(dsrList, indent=8)
      # cat("\n\t\t\t|>make_weighted: <input> psrList:")
      # qvcalc::indentPrint(psrList, indent=8)
      # cat("\n\t\t\t|>make_weighted: <input> seList:")
      # qvcalc::indentPrint(seList, indent=8)
      cat("\n\t\t\t|>make_weighted: <output> dsr:")
      qvcalc::indentPrint(dsr, indent=8)
      cat("\n\t\t\t|>make_weighted: <output> psr:")
      qvcalc::indentPrint(psr, indent=8)
      cat("\n\t\t\t|>make_weighted: <output> std err:")
      qvcalc::indentPrint(serr, indent=8)
  #
  #     # cat("\n")
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

print_ndata <- function(nData,nData_disc,nData_an,debug){
      if(debug>=3) cat("\n\t[*] [*] [*] [*] [*]  NEST DATA [*] [*] [*] [*] [*] [*] [*] \n")
      if(debug>=4) cat("\n\t\t** all nest data:\n")
      if(debug>=4 & debug<6) qvcalc::indentPrint(head(nData,30), indent=8)
      if(debug>=6) qvcalc::indentPrint(nData, indent=8)

      all_fld      <- sum(nData$fate==2, na.rm=TRUE)
      all_hatch    <- sum(nData$fate==0, na.rm=TRUE)
      all_dep      <-  sum(nData$fate==1, na.rm=TRUE)
      all_longfin  <- sum(nData$fint>par$obsFreq, na.rm=TRUE)
      all_hatch_longfin <- sum(nData$fint>par$obsFreq, na.rm=TRUE)
      all_misclass <- sum(nData$fate!=nData$afate,na.rm=TRUE)
      all_unk      <- sum(nData$afate==7, na.rm=TRUE)
      apparent_all <- dsr$calc_dsr(nData=nData,nestType="all", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
      num_disc <- nrow(nData_disc)
      num_excl      <- sum(nData_disc$afate==7, na.rm=TRUE)
      num_misclass  <- sum(nData_disc$fate!=nData_disc$afate,na.rm=TRUE)
      num_misclass  <- num_misclass-num_excl
      num_an        <- num_disc - num_excl

      mayfield_disc <- dsr$calc_dsr(nData=nData_disc,nestType="discovered", calcType="mayfield",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      apparent_disc <- dsr$calc_dsr(nData=nData_disc,nestType="discovered", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      prop_excl     <- num_excl/num_disc
      prop_misclass <- num_misclass/num_disc
      disc_fld      <- sum(nData_disc$fate==2, na.rm=TRUE)
      disc_hatch    <- sum(nData_disc$fate==0, na.rm=TRUE)
      disc_longfin  <- sum(nData_disc$fint>par$obsFreq, na.rm=TRUE)
      if(debug>=3) cat(sprintf("\n\t\t>>> discovered nests (length=%s):\n", num_disc))
      if(debug>=3 & debug<6) qvcalc::indentPrint(head(nData_disc,25), indent=12)
      if(debug>=6) qvcalc::indentPrint(nData_disc)
      if(debug>=2) cat(sprintf("\t\t\tfor discovered nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_disc, apparent_disc))
      mayfield_an <- dsr$calc_dsr(nData=nData_an,nestType="analysis", calcType="mayfield",
                                    # conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debug)

      apparent_an <- dsr$calc_dsr(nData=nData_an,nestType="analysis", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      an_fld <- sum(nData_an$fate==2, na.rm=TRUE)
      an_hatch <- sum(nData_an$fate==0, na.rm=TRUE)
      an_misclass <- sum(nData_an$fate!=nData_an$afate,na.rm=TRUE)
      an_longfin <- sum(nData_an$fint>par$obsFreq, na.rm=TRUE)

      if(debug>=5){
        cat("\n\t\t>> NAs after excluding unknown fate nests:\n")
        print(colSums(is.na(nData_an))) ## this is actually indented somewhat
        cat("\n")
      }

      if(debug>=3) cat(sprintf("\n\t\t>>> analyzed nests (length=%s ; num excluded=%s):\n",nrow(nData_an), num_excl))
      if(debug>=3 & debug<6) qvcalc::indentPrint(head(nData_an,25), indent=12)
      if(debug>=6) qvcalc::indentPrint(nData_an, indent=8)
      if(debug>=3) cat(sprintf("\t\t\tfor analyzed nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_an, apparent_an))

      if(debug>=1){
        # allVal <- c("all_fld","all_hatch","all_longfin","all_unk","all_misclass")
        # allVal <- c(all_fld,all_hatch,all_longfin,all_misclass,all_unk)
        allVal <- c(all_fld,all_hatch,all_longfin,all_misclass,num_disc)
        allProp <- allVal/par$numNests
        discVal <- c(disc_fld,disc_hatch,disc_longfin,num_misclass,num_excl)
        discProp <- discVal/num_disc
        anVal <- c(an_fld,an_hatch,an_longfin,an_misclass)
        anProp <- anVal/num_an
        # cat("\n all depredated=",all_dep)
        # cat(sprintf("\n\tall: flooded=%s, hatched=%s, long final=%s, unknown=%s, miclassified=%s ",allVal))
        cat(do.call(sprintf,c("\n\tall: flooded=%s, hatched=%s, long final=%s, miclassified=%s, discovered=%s ",as.list(allVal))))
        cat(do.call(sprintf,c("\t\t\t\t| prop flooded=%s, prop hatched=%s, prop long final=%s, prop miclassified=%s, prop discovered=%s ",as.list(allProp))))
        cat(do.call(sprintf,c("\n\tdiscovered: flooded=%s, hatched=%s, long final=%s, miclassified=%s, excluded=%s ",as.list(discVal))))
        cat(do.call(sprintf,c("\t | prop flooded=%.3f, prop hatched=%.3f, prop long final=%.3f, prop miclassified=%.3f, prop excluded=%.3f ",as.list(discProp))))
        cat(do.call(sprintf,c("\n\tanalyzed: flooded=%s, hatched=%s, long final=%s, misclassified=%s",as.list(anVal))))
        cat(do.call(sprintf,c("\t\t\t\t\t\t\t\t\t | prop flooded=%.3f, prop hatched=%.3f, prop long final=%.3f, prop miclassified=%.3f ",as.list(anProp))))
      }
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

make_summary <- function(dsrVal, nVal, truePSRcovar="date", debug=0){
  if(debug>=5) message("\n\n\t>> saving vals to matrix for summary")
  aDSR = nVal["aDSR"]
  aPSR = aDSR ^ par$hatchTime
  mayfDSR <- dsrVal[length(dsrVal)-4]
  mayfPSR = mayfDSR ^ par$hatchTime
  if(vals$psrTrueVal=="date") {
    psrT = dsrVal[4]
  } else {
    psrT = dsrVal[3]
  }
  if(debug>=2) cat(sprintf("\n\t\tpsrT (covar = %s): %s", truePSRcovar,psrT))
  if(config$logex & config$predict) {leDSR = dsr1 } else {leDSR = 0}
  if(config$logex & config$predict) {lePSR = psr1 } else {lePSR = 0}
  if(config$mcmc) mcmcPSR = dsrMat['mcmcPSR',r,i] else mcmcPSR=0
  if(debug>=1){
    cat(sprintf( "\n\n\t<> <> PSR vals: true= %.5f, true (date covar)= %.5f, MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ",
                psrT, psrT_date, mcmcPSR,  lePSR, mayfPSR))
    cat(sprintf( "\n\t<> <> <> <> <> <> <> <> diff from true (date covar): MCMC=%.5f, logEx=%.5f, Mayfield=%.5f \n",
                mcmcPSR-psrT_date, lePSR-psrT_date, mayfPSR-psrT_date))
  }
  vals         <- c(psrT_date,lePSR,mcmcPSR,mayfPSR,psrT,aPSR)
  diffs        <- vals[-1] - psrT_date
  ret <- c(par$stormFrq,par$pMortFl,par$obsFreq,par$discProb,par$decayRate,par$probSurv,
                    dsrT_date,all_hatch,all_fld,num_disc,num_excl,prop_excl,prop_misclass,vals,diffs)
  return(ret)

}

prBoxPl <- function(val1, valMatList, deb=FALSE){
    valStr <- stringr::str_extract(val1, "(?<=_)\\w+")
    cat(sprintf("\n>> difference between true PSR and %s PSR for each param set: \n\n", valStr))
    boxPlList  <- lapply(valMatList, function(x) {
                           unname(x[val1,])
          })
    names(boxPlList) <- paste0("parSet", seq(boxPlList))
    if(deb) cat(sprintf("\nboxPlList <%s>:\n", class(boxPlList)))
    if (deb) print(boxPlList) # txtplot::txtboxplot(boxPlList)
    if (deb) cat("\nfunction call for txtboxplot:\n")
    callList <- list(as.name("txtplot::txtboxplot"), c(boxPlList, list(width=70)))
    if (deb) print(as.call(callList))

    ## need to pass the list vals as individual arguments:
    do.call(txtplot::txtboxplot, c(boxPlList, list(width=70, legend=FALSE)))
}
