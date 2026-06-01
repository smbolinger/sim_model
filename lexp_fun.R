

mk_true_dsr <- function(nData, modList, preDat, par, config){
# mk_true_dsr <- function(nData, modForm, preDat, par, config){
  # if(config$debugDSR>=3){
    # print(preDat)
  # }
  nData <- nData |>
    mutate(Survival = end - init) ## total survival days

  fitData <- nData[rep(1:nrow(nData), times=nData$Survival),]
  # print(unlist(lapply(nData$survival, function(x) seq(1,x))) )
  # print(fitData$init + fitData$day)
  # dayz <- unlist(lapply(nData$survival, function(x) seq(1,x))) 
  # datez <- fitData$init + fitData$day
  # cat("\nlength of days & dates; nrow of data: ", length(dayz), length(datez), nrow(fitData))
  # print(nData$survival)
  # print(nData$init)
  # print(dayz)
  # print(datez)
  fitData$Day <- unlist(lapply(nData$Survival, function(x) seq(1,x))) 
  fitData$Date <- fitData$init + fitData$Day
  # fitData <- fitData |> mutate(status = ifelse(fate %in% c(1,2) & Date==end, 0, 1))
  fitData <- fitData |> mutate(Surv = ifelse(fate %in% c(1,2) & Date==end, 0, 1))

  # form <- as.formula(modForm)
  # modFit <- glm(form, data=fitData, family=binomial)
  # out <- predict(modFit, newdata=preDat, type="response")
  # psrOut <- out ^ par$hatchTime
  #
  # allInits    <- nestData1$init 
  # numInit     <- sapply(preDat$Date, function(x) sum(allInits==x))
  # propInit    <- numInit/par$numNests
  # propInitScl <- propInit/sum(propInit)
  # psrScl      <- sum(propInitScl * psrOut)
  ## you can alwways get the null even w/o it bing in modList
  # if(config$debugDSR>=4) qvcalc::indentPrint(modFit)

  # modFit <- lapply(modList, function(x){
  ## returns a list:
  out <- lapply(modList, function(x){
                     # form <- as.formula(modList[x])
                     form <- as.formula(x)
                     modFit <- glm(form, data=fitData, family=binomial)
                     if(config$debugDSR>=4) qvcalc::indentPrint(modFit)
                     predict(modFit, newdata=preDat, type="response")
                     # out <- predict(modFit, newdata=preDat, type="response")
                     # psrOut <- out ^ par$hatchTime
                     #
                     # allInits    <- nestData1$init 
                     # numInit     <- sapply(preDat$Date, function(x) sum(allInits==x))
                     # propInit    <- numInit/par$numNests
                     # propInitScl <- propInit/sum(propInit)
                     # # psrScl      <- sum(propInitScl * psrOut)
                     # sum(propInitScl * psrOut)
  })

  # psrOut <- out ^ par$hatchTime
  psrOut <- lapply(out, function(x) x ^ par$hatchTime)

  allInits    <- nData$init 
  numInit     <- sapply(preDat$Date, function(x) sum(allInits==x))
  propInit    <- numInit/par$numNests
  propInitScl <- propInit/sum(propInit)
  # psrScl      <- sum(propInitScl * psrOut)
  psrScl      <- sapply(psrOut,function(x) sum(propInitScl * x))
  dsr         <- out[[1]]
  # dsrNull <- 


  if(config$debugDSR>=3){
    cat("\n\t\t>> data for calculating true DSR:\n")
    qvcalc::indentPrint(head(fitData,80))
    # cat("\n\t\t\t> model output:\n")
    # qvcalc::indentPrint(modFit)
    cat("\n\t\t\t> DSR for all models:\n")
    qvcalc::indentPrint(out)
    # cat("\n\t\t\t> DSR for null model:\n")
    # qvcalc::indentPrint(psrOut)
    cat("\n\t\t\t> PSR for all models:\n")
    qvcalc::indentPrint(psrOut)

    cat("\n\t\t\t> DSR (null): ", dsr)
    cat("\n\t\t\t> PSR, weighted average: ", psrScl)
    # cat("\n\t\tcoefficients:\n")
    # qvcalc::indentPrint(coef(modFit))
    # writeLines(as.character(psrOut), )
    # conn <- file("psr_plot.txt", open="a")
    # writeLines(psrOut, con=conn)
    ## should append vector to file as a single line:
    ## will just keep appending (doesn't reset) - need to create blank file elsewhere in script
    ## or just take the last 100 lines written to the file and plot those?
    write(psrOut, file="out/psr_plot.txt", ncolumns=length(psrOut), append=TRUE)
  }

  
  # return(list(c(psrOut, psrScl)))
  ## return DSR & PSR from null model, and weighted PSR from Date model:
  return(c(dsr, psrScl))
  # return(psrScl)
  # return(psrOut)
  # fitData <- lapply(nData, function(x) )
  # colPivot <- c("")
  # fitData <- nData |>
    # tidyr::pivot_longer()
    # pivot_longer()
           
           # trials = 

}

logexp <- function(exposure = 1) {
  ## function from Bolker
  ## hack to help with visualization, post-prediction etc etc
  get_exposure <- function() {
    if (exists("..exposure", env=.GlobalEnv))
      return(get("..exposure", envir=.GlobalEnv))
    exposure
  }
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

## probably makes more sense to do in python bc of search functions
get_exposure <- function(numNests, survey, firstDay, lastDay, config){
  db=config$debugLogEx
  surveyDays = survey[[1]]
  surveyInts = survey[[2]]
}

## take reduced nest data + survey info and create df to pass to logex function
mk_logex_data <- function(nestData,survey,pyconfig,exposure=0){

  config = py_to_r(pyconfig)
  debug <- config$debugLogEx
  nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
  nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) # print(head(nestObs))


  ##--- 1. choose dates to pass - different for daily exposure vs. normal ---------------------------------------------------------
  if(exposure==1){
    #NOTE: should this be up until the final active day of any nest?
    svyDays <- np_array(seq(max(survey[[1]]))) ## better than using as.matrix?
    svyInts <- np_array(rep(1, length(svyDays)))
    ## sum happens inside function; also needs to be all nests, not just discovered:
    numObs  <- nestData[,"end"] - nestData[,"init"]
    first   <- nestData[,"init"]
    last    <- nestData[,"end"]
    exp1    <- TRUE
  } else {
    svyDays <- survey[[1]]
    svyInts <- survey[[2]]
    first   <- nestData[,"i"]
    last    <- nestData[,"k"]
    numObs  <- nestData[, "totobs"]
    exp1    <- FALSE
  }

  ##--- 2. get exposure days from survey info -----------------------------------------
  expoList   <- dsr$calc_daily_expo(numNests=nNest, surveyDays=svyDays,
                                   surveyInts=svyInts, firstDay=first,
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
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(debug>=3){
    cat("\n\t>->num obs:", numObs)
    cat(sprintf("\n\t>-> survey end days (len %s) & survey ints (len %s) to pass to logexp functions:\n",
                length(svyDays),length(svyInts)))
    # print(class(svyInts)) 
    # print(class(svyDays))
    # print(py_to_r(svyInts))
    # print(py_to_r(svyInts)[-1])
    # prvec(py_to_r(svyInts), nms=py_to_r(svyDays)[-1])
    prvec(py_to_r(svyInts), nms=py_to_r(svyDays)[-length(svyDays)])
    # qvcalc::indentPrint(svyDays)
    # qvcalc::indentPrint(svyInts)
    cat("\n\t\t>>> calculating daily exposure\n")
  }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return(dat2S)
}

## take the dataframe made in make_logex_data and pass to glm
calc_logexp <- function(modList,dat2S,exp=0,config){

  debug <- config$debugLogEx
  # if(dbug>=3) cat("\n\t>>> calculating logistic exposure\n")

  excpt <- FALSE
  warn  <- FALSE
  if (exp==1) {modList=modList[c(1,2)]}
    # cat("\n|> made obs data\n") print(obsDat)
  # tryCatch({ modOut <- fit_glm(modList,dat=dat2S,debug=config$debugLL) },
  # withCallingHandlers({ modOut <- fit_glm(modList,dat=dat2S,debug=config$debugLL) },
   # modOut <- tryCatch({
  if(debug>=3) cat("\n\t>-> modList = ", modList)
  tryCatch({

    modOut <- withCallingHandlers({
      fit_glm(modList,dat=dat2S,debug=config$debugLogEx) },

      error = function(e) { 
      message("\t!! error in glm:", e) 
      # tryCatch({modOut <- fit_glm})
      # message("!! error in glm:", e, "go to next") 
      # excpt <<- TRUE
      # coefsArray <- rep(-999, length(coef_names))
      # coefs[,r,i] <- coefsArray
      # coefsArray <- -999
      # coefs[,r,i] <- -999
      },

      warning = function(w) { 
        # message("!! warning in glm:", w, "go to next") 
        message("\t!! warning in glm:", w) 
        # if (modOut$converged==FALSE){
        # excpt <<- TRUE
        warn <<- TRUE
        # coefsArray <- rep(-999, length(coef_names))
        # coefs[,r,i] <- coefsArray
        # coefs[,r,i] <- -999
        # coefsArray <- -999
    })
   },

   error=function(e){
      cat("\n\t~~ exception - error ~~")
      excpt <<- TRUE
      # return("exception")
   })
  if(warn) {
    message("~~ exception - warning but no error ~~")
    # if(length(modOut>0)){
    # if(is.list(modOut) & length(modOut>0)){
    tryCatch({
      if (modOut$converged==FALSE){
        cat("\n\t~~ exception - model did not converge ~~")
        return("exception")
      # conv <- modOut$converged
      }
      },
      error=function(e){
        cat("unclear if converged - go to next")
        excpt <<- TRUE
        # return("exception")
      })
  }
      # if (modOut$converged==FALSE){
      # if (conv==FALSE){
      #   # message("~~ exception ~~")
      #   cat("\n\t~~ exception - model did not converge ~~")
      #   return("exception")
      #   }
  if(excpt) return("exception")
  coefsArray <- get_coef(modOut, debug=config$debugLogEx)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if(debug>=3) cat(sprintf("\n\t|> coefsArray <class:%s> =\n", class(coefsArray)))
  # if(debug>=3) qvcalc::indentPrint(coefsArray)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(coefsArray)
}

fit_glm <- function(modList, dat, exposure = 1, debug=F){
  ## modList contains the formulas for the models
  # modList <- rlang::parse_exprs(modList)
  if(debug>=3) cat("\n\t\t>>> FITTING MODELS\n")
  out <- list()
  for(m in seq_along(modList)){
    vars         <- stringr::str_extract_all(modList[m], "[\\w()^]{2,}")
    vars         <- vars[[1]][-1]
    # if(debug>=4) cat("\n\t\t\tvars: ",paste(vars, collapse=","))
    form <- as.formula(modList[m])
    # if(debug>=4) cat("\t\tmodel: ",modList[m])
    # start <- c(1, rep(0,m-1))
    start <- c(1, rep(0,length(vars)))
    # if(debug>=4) cat("\t\tstart: ",start)
    # out[[m]] <- glm(modList[m], data=dat,
    out[[m]] <- glm(form, data=dat, start=start,
                    family=binomial(link=logexp(dat$Exposure)))
    # if(debug>=4) qvcalc::indentPrint(summary(out[[m]]),indent=8)
  }

    ## move trycatch outside of function so you can skip entire iteration

  return(out)
}

## do I ever use this??
get_logex <- function(nestData,coefsArray,mList,dat,config){
  debug = config$debugLogEx
  dsr1 <-  1/(1+exp(-coefsArray[[1]][1,1]))
  psr1 <- dsr1 ^ par$hatchTime
  allInits <- nestData$init
  numInit <- sapply(dat$Date, function(x) sum(allInits==x))
  propInit <- numInit/par$numNests
  propInitScl <- propInit/sum(propInit)
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

get_coef <- function(modOut, debug=0){
  # if(debug>=3) cat("\n\t\t>>> getting coefficients from models\n")
  coefsArray = sapply(modOut, function(x){
                        # if(debug>=5) cat("\n\t\t\tcoefs input:\n")
                        # if(debug>=5) qvcalc::indentPrint(x,indent=8)
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
  # if (debug>=4) cat("\n\t\t\tcoefs output:\n")
  # if (debug>=4) qvcalc::indentPrint(coefsArray)
  return(coefsArray)
}

make_pred <- function(coefArr,nmod,mods,newDat,hTime,db=0){
  # intercepts <- array(NA, dim=c(nmod))
  # betas
  # mod_eq <- list()
  dsrList <- list()
  
  # psrList <- list()
  # mod_eq[[1]] <- plogis(coefArr[[]])
  ## already calculated for constant model in the main script
  # for(m in 2:nmod){
  # if(db>=3) cat("\n\t\tnewDat:\n")
  # if(db>=3) qvcalc::indentPrint(head(newDat))
  for(m in seq(2,nmod)){# why get rid of the first one when already starting at 2??
    # vars         <- stringr::str_extract_all(mods, "\\w{2,}")[-1]
    # vars         <- stringr::str_extract_all(mods, "(?<=~)[\\w()^]{2,}")
    vars         <- stringr::str_extract_all(mods[m], "[\\w()^]{2,}")
    # cat("\n\t\tcoefArr: ")
    # qvcalc::indentPrint(coefArr)
    # cat("\n\t\tVARS: ")
    # qvcalc::indentPrint(vars)
    # vars         <- sapply(vars, function(x) x[-1])
    vars         <- vars[[1]][-1]
    # if(db>=2) cat(sprintf("\n\t\tMODEL %s: %s ; VARS: %s",m, mods[m], vars))
    int          <- coefArr[[m]][1,1]
    # betas        <- sapply(coefArr, function(x) )
    if(length(vars)>1){
      betas        <- c(coefArr[[m]][1,2],coefArr[[m]][1,3])
    } else {
      betas        <- c(coefArr[[m]][1,2])
    }
    # if(db>=3) print(vars)
    # newD         <- sapply(vars, function(x) sprintf("newDat[[%s]]", x))
    # if(db>=3) print(newD)
    # mod_eq       <- make_pr_eq(int,betas,vars,db)
    # mod_eq       <- str2expression(make_pr_eq(int,betas,newD,db))
    mod_eq       <- str2expression(make_pr_eq(int,betas,vars,db))
    # if(db>=3) cat("\n\t\tas expression:")
    # if(db>=3) qvcalc::indentPrint(mod_eq)
    dsrList[[m]] <- eval(mod_eq, envir=newDat)
    # if(db>=3) qvcalc::indentPrint (dsrList[[m]])
    # dsrList[[m]] <- plogis(mod_eq[[m]])
    # psrList[[m]] <- dsrList[[m]]^hTime
  }
  # psrList <- sapply(dsrList, function(x) x^hTime)
  return(dsrList)
}

make_pr_eq <- function(intercept, betas, x,db){
  # if(db>=3) cat(sprintf("\npass to function: %s %s %s\n",intercept, betas,x))
  # if(db>=3) cat(sprintf("\n\t\tpass to function: %s %s \n",intercept, paste(betas,x, sep=" ")))
  # if(length(betas)>1){
  #   beta_expand <- sapply(betas, function(i) paste(betas[i],x[i],sep="*"))
  # }else{
  beta_expand <- paste(betas,x,sep="*")
  # }
  # if(db>=3) cat(sprintf("\n\t\tbetas: %s\n", beta_expand))
  # eqs <- sapply(beta_expand, function(x)cat( intercept,"+", paste(x, collapse="+")))
  # eq <- cat( intercept,"+", paste(beta_expand, collapse="+"))
  # eq <- sprintf("%s + %s", intercept, paste(beta_expand, collapse="+"))
  # eq <- sprintf("qlogis(%s + %s)", intercept, paste(beta_expand, collapse="+"))
  eq <- sprintf("1/(1+exp(-(%s + %s)))", intercept, paste(beta_expand, collapse="+"))
  # if(db>=3) cat("\t\t>> equation:", eq)

  return(eq)
}

make_psr <- function(psrList, prop_nests,db=0){
  ## Make weighted PSR values 
  ## psrList is dsrList ^ hatchTime; prop_nests is proportion of nests initiated on day j
  ## this could either be the true number or some estimate by the observer
  ## for now, stick with the true number
  psrOut <- lapply(psrList, function(x) sum(x*prop_nests))
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if(db>=3){
  #   cat(sprintf("\n\t>>> calculate for first psr list (lengths= %s, %s):\n",
  #               length(psrList[[1]]), length(prop_nests)))
  #   qvcalc::indentPrint(psrList)
  #   qvcalc::indentPrint(prop_nests)
  #   # print(sum(psrList[[1]]*prop_nests))
  #   # return(sum(psrList*prop_nests))
  #   cat("\n\t|>output of make_psr:\n")
  #   qvcalc::indentPrint(psrOut)
  # }
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(psrOut)
}

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
