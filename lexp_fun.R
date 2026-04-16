
## Begin Example 1
## logistic exposure model, following the Example in ?family. See,
## Shaffer, T. 2004. Auk 121(2): 526-540.
# Definition of the link function
logexp_brglm <- function(days = 1) {
  linkfun <- function(mu) qlogis(mu^(1/days))
  linkinv <- function(eta) plogis(eta)^days
  mu.eta <- function(eta) days * plogis(eta)^(days-1) *
  binomial()$mu.eta(eta)
  valideta <- function(eta) TRUE
  link <- paste("logexp(", days, ")", sep="")
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

fit_glm <- function(modList, dat, exposure = 1, debug=F){
  ## modList contains the formulas for the models
  # modList <- rlang::parse_exprs(modList)
  if(debug>=3) cat("\n\t\tFITTING MODEL\n")
  out <- list()
  # for(m in seq_along(modList)){
  #   form <- as.formula(modList[m])
  #   start <- c(1, rep(0,m-1))
  #   if(debug) print(start)
  #   if(debug) print(form)
  #   # out[[m]] <- glm(modList[m], data=dat,
  #   out[[m]] <- glm(form, data=dat, start=start,
  #                   family=binomial(link=logexp(dat$Exposure)))

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
  out[[1]] <- glm(Surv~1, data=dat,
                  family=binomial(link=logexp(dat$Exposure)))
  if(debug>=4) qvcalc::indentPrint(summary(out[[1]]))

  out[[2]] <- glm(Surv~Date, data=dat,
                  family=binomial(link=logexp(dat$Exposure)))
  if(debug>=4) qvcalc::indentPrint(summary(out[[2]]))

  out[[3]] <- glm(Surv~Date+I(Date^2), data=dat,
                  family=binomial(link=logexp(dat$Exposure)))
  if(debug>=4) qvcalc::indentPrint(summary(out[[3]]))

  out[[4]] <- glm(Surv~Age, data=dat,
                  family=binomial(link=logexp(dat$Exposure)))
  if(debug>=4) qvcalc::indentPrint(summary(out[[4]]))

  out[[5]] <- glm(Surv~Date+Age, data=dat,
                  family=binomial(link=logexp(dat$Exposure)))
  if(debug>=4) qvcalc::indentPrint(summary(out[[5]]))

  return(out)
}

get_logex <- function(nestData,coefsArray,mList,dat){
  dsr1 <-  1/(1+exp(-coefsArray[[1]][1,1]))
  psr1 <- dsr1 ^ par$hatchTime
  allInits <- nestData$init
  numInit <- sapply(dat$Date, function(x) sum(allInits==x))
  propInit <- numInit/par$numNests
  propInitScl <- propInit/sum(propInit)
  if(debug>=4){
    cat("\n\t\tinits & dates:\n")
    qvcalc::indentPrint(allInits)
    qvcalc::indentPrint(dat2S$Date)
    # cat("\nnum inits before date:\n")
    cat("\n\t\tnum inits on date:\n")
    qvcalc::indentPrint(numInit)
    cat("\n\t\tproportion inits on date:\n")
    qvcalc::indentPrint(propInit)
    qvcalc::indentPrint(propInitScl)
  }
  dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugSummary)
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
  coefsArray = sapply(modOut, function(x){
                        if(debug>=4) qvcalc::indentPrint(x)
                        # print(coef(x))
                                sapply(seq_along(coef(x)), function(y){
                                         ## R STILL trying to return conf instead of coef_arr?
                                         # print(y)
                                         if (is.matrix(confint.default(x))){ 
                                            if (debug>=4) cat("\n\t\tcoefs&confint:\n")
                                            if (debug>=4) qvcalc::indentPrint(c(coef(x)[y], confint.default(x)[y,]))
                                            return(c(coef(x)[y], confint.default(x)[y,]))
                                         } else {
                                            if (debug>=4) cat("\n\t\tcoefs&confint:\n")
                                           if (debug>=4) qvcalc::indentPrint(c(coef(x)[y], confint.default(x)[y]))
                                           return(c(coef(x)[y], confint.default(x)[y]))
                                         }
                                         })
                 })
  if (debug>=4) cat("\n\t\t\tcoefs output:\n")
  if (debug>=4) qvcalc::indentPrint(coefsArray)
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
  if(db>=3) cat("\n\t\tnewDat:\n")
  if(db>=3) qvcalc::indentPrint(head(newDat))
  for(m in seq(2,nmod)){
    # vars         <- stringr::str_extract_all(mods, "\\w{2,}")[-1]
    # vars         <- stringr::str_extract_all(mods, "(?<=~)[\\w()^]{2,}")
    vars         <- stringr::str_extract_all(mods[m], "[\\w()^]{2,}")
    # vars         <- sapply(vars, function(x) x[-1])
    vars         <- vars[[1]][-1]
    if(db>=2) cat(sprintf("\n\t\tMODEL %s: %s ; VARS: %s",m, mods[m], vars))
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
    if(db>=3) cat("\n\t\tas expression:\n")
    if(db>=3) qvcalc::indentPrint(mod_eq)
    dsrList[[m]] <- eval(mod_eq, envir=newDat)
    if(db>=3) qvcalc::indentPrint (dsrList[[m]])
    # dsrList[[m]] <- plogis(mod_eq[[m]])
    # psrList[[m]] <- dsrList[[m]]^hTime
  }
  # psrList <- sapply(dsrList, function(x) x^hTime)
  return(dsrList)
  # dsrArr1 <-  1/(1+exp(-(coefsArray[[2]][1,1] + coefsArray[[2]][1,2] * prDays)))
  # if (debug>=3) cat("\ndsr & psr from model 1:\n")
  # if (debug>=3) print(dsrArr1)
  # psrArr1 <- dsrArr1 ^ par$hatchTime
  # if (debug>=3) print(psrArr1)
  # dsrArr2 <-  1/(1+exp(-(coefsArray[[3]][1,1] + coefsArray[[3]][1,2] * prDays + coefsArray[[3]][1,3] * prDays^2)))
  # psrArr2 <- dsrArr2 ^ par$hatchTime
  # if (debug>=3) cat("\ndsr & psr from model 2:\n")
  # if (debug>=3) print(dsrArr2)
  # if (debug>=3) print(psrArr2)
  # plogis(intercept )
}

make_pr_eq <- function(intercept, betas, x,db){
  # if(db>=3) cat(sprintf("\npass to function: %s %s %s\n",intercept, betas,x))
  if(db>=3) cat(sprintf("\n\t\tpass to function: %s %s \n",intercept, paste(betas,x, sep=" ")))
  # if(length(betas)>1){
  #   beta_expand <- sapply(betas, function(i) paste(betas[i],x[i],sep="*"))
  # }else{
  beta_expand <- paste(betas,x,sep="*")
  # }
  if(db>=3) cat(sprintf("\n\t\tbetas: %s\n", beta_expand))
  # eqs <- sapply(beta_expand, function(x)cat( intercept,"+", paste(x, collapse="+")))
  # eq <- cat( intercept,"+", paste(beta_expand, collapse="+"))
  # eq <- sprintf("%s + %s", intercept, paste(beta_expand, collapse="+"))
  # eq <- sprintf("qlogis(%s + %s)", intercept, paste(beta_expand, collapse="+"))
  eq <- sprintf("1/(1+exp(-(%s + %s)))", intercept, paste(beta_expand, collapse="+"))
  if(db>=3) cat("\t\t>> equation:", eq)

  return(eq)
}

make_psr <- function(psrList, prop_nests){
  ## Make weighted PSR values 
  ## psrList is dsrList ^ hatchTime; prop_nests is proportion of nests initiated on day j
  ## this could either be the true number or some estimate by the observer
  ## for now, stick with the true number
  psrOut <- lapply(psrList, function(x) sum(x*prop_nests))
  # print(psrList)
  # print(prop_nests)
  # cat("\ncalculate for first psr list:\n", length(psrList[[1]]), length(prop_nests))
  # print(sum(psrList[[1]]*prop_nests))
  # return(sum(psrList*prop_nests))
  # cat("\noutput of make_psr:\n")
  # print(psrOut)
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
  if (db>=1) cat("\nMARK: returning model results for ", names(ret))
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
  res <- invisible(make_mark_models(inp,nocc,db))
  if(db>=4) cat("\n\t\tRMark OUTPUT:\n")
  if(db>=4) qvcalc::indentPrint(res)
  dsrVals <- list()
  dsrVals[["dot"]] <- res$S.dot$results$real[,1:4]
  dsrVals[["date"]] <- res$S.date$results$real[,1:4]
  dsrVals[["dsAge"]] <- res$S.dsAge$results$real[,1:4]
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

