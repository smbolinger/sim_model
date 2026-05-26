
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

logexp <- function(exposure = 1) {
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

mk_logex_data <- function(nestData,survey,pyconfig,exposure=0){
  config = py_to_r(pyconfig)
  debug <- config$debugLogEx
  # cat(sprintf("debug: %s ; type: %s ; length: %s", debug, class(debug), length(debug)))
  nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
  # nestObs <- nestData |> dplyr::select(ID, init, i, j, k, afate, totobs) # print(head(nestObs))
  nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) # print(head(nestObs))
  # if(debug>=3) cat("\n\n\t<*><*> Logistic exposure ")
  if(exposure==1){
    # if(debug>=2) cat("- exposure=1 <*><*><*><*>\n")
    #NOTE: should this be up until the final active day of any nest?
    svyDays <- np_array(seq(max(survey[[1]]))) ## better than using as.matrix?
    svyInts <- np_array(rep(1, length(svyDays)))
    # numObs <- sum(nestData[,"k"] - nestData[,"i"])
    ## sum happens inside function
    # numObs <- nestData[,"k"] - nestData[,"i"]
    ## also needs to be all nests, not just discovered:
    numObs  <- nestData[,"end"] - nestData[,"init"]
    first   <- nestData[,"init"]
    last    <- nestData[,"end"]
    exp1    <- TRUE
  } else {
    # if(debug>=2) cat("- exposure=exposure <*><*><*><*>\n")
    svyDays <- survey[[1]]
    svyInts <- survey[[2]]
    first   <- nestData[,"i"]
    last    <- nestData[,"k"]
    numObs  <- nestData[, "totobs"]
    exp1    <- FALSE
  }
  # expoList   <- logex$calc_daily_expo(numNests=nNest, surveyDays=svyDays,
  #                                  surveyInts=svyInts, firstDay=nestData$i,
  #                                  lastDay=nestData$k, db=config$debugLL)

  expoList   <- logex$calc_daily_expo(numNests=nNest, surveyDays=svyDays,
                                   surveyInts=svyInts, firstDay=first,
                                   lastDay=last, config=pyconfig)
  # if(dbug>=3) cat("\n\tmaking log exp dataframe\n")
  # if(debug>=3) cat("\n\tpass obs data to make_daily_logex_df:\n")
  # if(debug>=3) qvcalc::indentPrint(head(nestData,30))
  dat2S <-  withCallingHandlers(
    {logex$make_daily_logex_df(obsData=nestObs,
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
  }
  if(debug>=3) cat("\n\t\t>>> calculating daily exposure\n")
  # if(debug>=3) cat("\n\t>-> dat2S:\n")
  # if(debug>=3) qvcalc::indentPrint(head(dat2S, 25))
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(dat2S)
}

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

    # tryCatch( { out[[m]] <- glm(form, data=dat, start=start,
    #                 family=binomial(link=logexp(dat$Exposure)))
    #              },
    #              error=function(e){
    #                message("!! error in glm:", e, "re-run")
    #                reticulate::py_last_error()
    #                out[[m]] <- glm(form, data=dat, start=start,
    #                    family=binomial(link=logexp(dat$Exposure)),
    #                    method=brglm2::brglmFit,
    #                    control=brglmControl(maxit=500))
    #
    #                cat("\n**converged? ",out[[m]]$converged, "\n")
    #              },
    #              warning = function(w){
    #                message("!! warning in glm:", w, "re-run")
    #                out[[m]] <- glm(form, data=dat, start=start,
    #                    family=binomial(link=logexp(dat$Exposure)),
    #                    method=brglm2::brglmFit,
    #                    control=brglmControl(maxit=500))
    #
    #                cat("\n**converged? ",out[[m]]$converged, "\n")
    #              }
    # )
    # if(debug) print(summary(out[[m]]))
  # }

  # out[[1]] <- glm(modList[1], data=dat,
  #                 family=binomial(link=logexp(dat$exposure)))
  # if(debug) print(summary(out[[1]]))
  #
  # out[[2]] <- glm(modList[2], data=dat,
  #                 family=binomial(link=logexp(dat$exposure)))
  # if(debug) print(summary(out[[2]]))
  #
  # out[[3]] <- glm(modList[3], data=dat,
  #                 family=binomial(link=logexp(dat$exposure)))
  # if(debug) print(summary(out[[3]]))

  # just explicitly specify the formulas, since it doesn't like anything else...
  # dat <- na.omit(dat)
  # out[[1]] <- glm(Surv~1, data=dat, start=c(1),control=glm.control(maxit=1000),
  # # out[[1]] <- glm2(Surv~1, data=dat, start=c(1),control=glm.control(maxit=1000),
  #                 family=binomial(link=logexp(dat$Exposure)))
  # if(debug>=4) qvcalc::indentPrint(summary(out[[1]]),indent=8)
  #
  # out[[2]] <- glm(Surv~Date, data=dat, start=c(1,0),control=glm.control(maxit=1000),
  #                 family=binomial(link=logexp(dat$Exposure)))
  # if(debug>=4) qvcalc::indentPrint(summary(out[[2]]),indent=8)
  #
  # out[[3]] <- glm(Surv~Date+I(Date^2), data=dat, start=c(1,0,0),control=glm.control(maxit=1000),
  #                 family=binomial(link=logexp(dat$Exposure)))
  # if(debug>=4) qvcalc::indentPrint(summary(out[[3]]),indent=8)
  #
  # out[[4]] <- glm(Surv~Age, data=dat,start=c(1,0),control=glm.control(maxit=1000),
  #                 family=binomial(link=logexp(dat$Exposure)))
  # if(debug>=4) qvcalc::indentPrint(summary(out[[4]]),indent=8)
  #
  # out[[5]] <- glm(Surv~Date+Age, data=dat,start=c(1,0,0),control=glm.control(maxit=1000),
  #                 family=binomial(link=logexp(dat$Exposure)))
  # if(debug>=4) qvcalc::indentPrint(summary(out[[5]]),indent=8)
  #
  # out[[6]] <- glm(Surv~avDate, data=dat,start=c(1,0),control=glm.control(maxit=1000),
  #                 family=binomial(link=logexp(dat$Exposure)))
  # if(debug>=4) qvcalc::indentPrint(summary(out[[5]]),indent=8)

  return(out)
}

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

# get_trueDSR <- function(nestData){
# }

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

# real_MARK <- function(inp,nocc,modList){
real_MARK <- function(nData, more=F , db=0){
  inp <- make_inp(nData)
  noc <- max(inp$LastChecked)
  if(db>=5) cat(sprintf("\n\t\t\t|> made 'inp' (nocc=%s):\n",noc))
  if(db>=5) qvcalc::indentPrint(inp)
  ## should output a list of arrays of DSR values & CIs:
  ret <- make_outp(inp, noc,more, db)
  if (db>=1) cat("\n\toooo|> MARK: returning model results for ", paste(names(ret),collapse=" ; "))
  return(ret)

}

make_inp <- function(dat,db=0){
  inp <- dat %>%
  mutate(Name        = sprintf("/*sim_%s*/", nest),
         FirstFound  = i,
         LastPresent = j,
         LastChecked = k,
         Fate        = fate,
         avDate      = (k-i)/2) %>%
  mutate(avAge       = avDate -init, 
         Freq        = 1) %>%
  select(Name, FirstFound, LastPresent, LastChecked, Fate, avDate, avAge, Freq)
  return(inp)

}

make_outp <- function(inp, nocc,more=F,db=0){
  # res <- invisible(run_mark_models(inp,nocc, more))
  if(db>=2) cat("\n<*><*><*> Run RMark <*><*><*><*><*>\n")
  res <- invisible(make_mark_models(inp,nocc,db))
  if(db>=3) cat(sprintf("\n\t\tRMark OUTPUT <type: %s> :\n",class(res)))
  if(db>=3) qvcalc::indentPrint(res)
  # if(db>=3) cat("\n\t\tAICc scores:\n") ## outputs a list
  # if(db>=3) print(sapply(res,function(x) x$results$AICc))
  top <- which.min(unlist(sapply(res,function(x) x$results$AICc)))
  # print(top)
  # expre <- paste("res",names(top),sep="$")
  expre <- paste(c("res",names(top),"results","real[,1:4]"),collapse="$")
  # print(expre)
  # print(rlang::expr(expre))
  # dsr <- eval(parse(text=expre))
  # print(res[top])
  dsrVals <- list()
  dsrVals[["dot"]]   <- res$S.dot$results$real[,1:4]
  dsrVals[["date"]]  <- res$S.date$results$real[,1:4]
  dsrVals[["dsAge"]] <- res$S.dsAge$results$real[,1:4]
  dsrVals[["top"]]   <- eval(parse(text=expre))
  dsrVals[["topname"]] <- top
  # dsrVals[["dot"]] <- res$Dot$results$real[,1:4]
  # if(more){
  #   dsrVals[["age"]] <- res$Age$results$real[,1:4]
  #   dsrVals[["date"]] <- res$Date$results$real[,1:4]
  #   dsrVals[["date_sq"]] <- res$Datesq$results$real[,1:4]
  #   dsrVals[["age_date"]] <- res$AgeDatesq$results$real[,1:4]
  # }
  # if(db>=3) print(dsrVals[["dot"]])
  return(dsrVals)
}

make_mark_models <- function(dat,noc,db=0){
  # out <- dat |> RMark::process.data(model="Nest",nocc=noc) |> RMark::make.design.data() 
  markDat <- RMark::process.data(dat, model="Nest",nocc=noc) 
  # desDat  <- RMark::make.design.data(markDat) 
  mark.ddl  <- RMark::make.design.data(markDat) 
  # Dotf    <- list(formula=~1)
  # Agef    <- list(formula=~avAge)
  # Datef   <- list(formula=~avDate)
  S.dot    <- list(formula=~1)
  S.age    <- list(formula=~avAge)
  S.date   <- list(formula=~avDate)
  S.datesq   <- list(formula=~avDate + I(avDate^2))
  S.dsAge   <- list(formula=~avAge + avDate + I(avDate^2))
  # mods    <- create.model.list("Nest")
  mark.model.list    <- create.model.list("Nest")
  sink("/dev/null")
  # mark.results <- mark.wrapper(mark.model.list, data=markDat, ddl=mark.ddl,
  res <- mark.wrapper(mark.model.list, data=markDat, ddl=mark.ddl,
                               invisible=TRUE,silent=TRUE,delete=TRUE)
  sink()
  # Dot     <- RMark::make.mark.model(markDat, desDat, parameters=list(Dot))
  # out   <- RMark::mark.wrapper(mods, data=markDat, ddl=desDat)
  # return(out)
  # res <- mark.results$results
  if(db>=4) qvcalc::indentPrint(res)
  if(db>=4) qvcalc::indentPrint(str(res))
  # return(mark.results)
  return(res)
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

run_mark_models <- function(dat,noc,runMore=F,inv=T,mod=NULL){
  # Dot <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,output=FALSE, silent=TRUE, model.parameters=list(S=list(formula=~1)) )
  #
  # if (runMore){
  #   Age <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
  #                      model.parameters=list(S=list(formula=~Age)) )
  #
  #   Date <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE, 
  #                       model.parameters=list(S=list(formula=~Date)) )
  #
  #   Datesq <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
  #                         model.parameters=list(S=list(formula=~Date + I(Date^2))) )
  #
  #   AgeDatesq <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
  #                            model.parameters=list(S=list(formula=~Age + Date + I(Date^2))) )
  # }

  # if(inv){
  #   Dot <- 
  # } else {
    invisible(capture.output(Dot <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
                               model.parameters=list(S=list(formula=~1)) )))

    if (runMore){
      invisible(Age <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
                                  model.parameters=list(S=list(formula=~Age)) ))

      invisible(Date <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE, 
                                   model.parameters=list(S=list(formula=~Date)) ))

      invisible(Datesq <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
                                     model.parameters=list(S=list(formula=~Date + I(Date^2))) ))

      invisible(AgeDatesq <- RMark::mark( dat, nocc = noc , model='Nest', se=TRUE,
                                        model.parameters=list(S=list(formula=~Age + Date + I(Date^2))) ))
      }
      # }

  return(RMark::collect.models())
}

# logexp_brglm_old <- function(days = 1) {
#   linkfun <- function(mu) qlogis(mu^(1/days))
#   linkinv <- function(eta) plogis(eta)^days
#   mu.eta <- function(eta) days * plogis(eta)^(days-1) *
#   binomial()$mu.eta(eta)
#   valideta <- function(eta) TRUE
#   link <- paste("logexp(", days, ")", sep="")
#   structure(list(linkfun = linkfun, linkinv = linkinv,
#           mu.eta = mu.eta, valideta = valideta, name = link),
#           class = "link-glm")
# }
#
#
# br.custom.family.old <- function(p) {
#   etas <- binomial(logexp(.days))$linkfun(p)
#   list(ar=0.5*p/p, # so that to fix the length of ar
#   at=0.5+exp(etas)*(1-p)/(2*p*.days))
# }
#
