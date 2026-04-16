
## attempting to streamline
## takes parts of logexp.R and datsim.py 


library(MASS)
library(RMark)
library(dplyr) # load dplyr last so as not to mask select?
library(brglm2)
library(reticulate)

#---- LOAD FUNCTIONS & VARIABLES --------------------------------------------------------------
source("lexp_fun.R")
source("lexp_setup.R")

parID = 0
for(i in seq(length(pArrList))){
  if(debug) cat("\n.....................................................i=",i, ".............................................................\n")

#---- Make params, storms, surveys: --------------------------------------------------------------
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  print(par) # if(debug) print(par$stormFrq)
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur)
  survey    <- withCallingHandlers(
                                   {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=config)},
                                   error=function(e){ 
                                     reticulate::py_last_error() 
                                     # print(sys.calls()) # doesn't help if error in python
                                   }  )


#----------------------------------------------------
  repID=0
  if (config$testing=="yes"){

    initDF <- data.frame(day=as.numeric(names(initDateList)),prop=unlist(initDateList,use.names=F))
    initDF$init <- initDF$prop * par$numNests
    cat("\n\tINIT DF:\n")
    qvcalc::indentPrint(initDF)
    # plotFile <- sprintf("%s/figs/%s_inits_density.png",outdir,parID)
    plotFile <- sprintf("figs/%s_inits_density.png",parID)
    cat("\n\tMAKING PLOT\n")
    pl <- ggplot2::ggplot()
    ## file gets closed in between reps:
    # cat("\n\tMAKING PNG\n")
    # png(plotFile, width=800,height=500)
    # # plot(density(initDateList), main="Simulated Nest Initiation Dates")
    # cat("\n\tMAKING PLOT\n")
    # plot(density(initDF$init), main="Simulated Nest Initiation Dates")
  }
  for(r in seq(nreps)){
    cat(sprintf("\n:::::::::::::::::::::::::::::: rep %s-%s :::::::::::::::::::::::::::::::::::::::::::\n",i,r))

  #---- Full nest data: ----------------------------------------------------
    skiptoNext <- FALSE
    nestData1 <- withCallingHandlers({
      nweeks = round(par$brDays/7)-1
      obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }
    colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
    nestData <- nestData1 |> as.data.frame() |> setNames(colnames)
    if (config$debugNests>=3) cat("\nall nest data:\n")
    # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
    if (config$debugNests>=3) qvcalc::indentPrint(nestData)
    # if (config$testing=="yes") lines(density(nestData1$init),col="green",)
    if (config$testing=="yes"){
      cat("\n\tADDING TO PLOT\n")
      # pl <- pl + ggplot2::geom_density(data=initDF,ggplot2::aes(x=!!rlang::sym(init)),color="darkturquoise",alpha=0.5) ## force evaluation 
      # pl <- pl + ggplot2::geom_density(data=nestData,ggplot2::aes(x=!!rlang::enquo(init)),color="darkturquoise",alpha=0.5) ## force evaluation 
      # pl <- pl + ggplot2::geom_density(data=nestData,ggplot2::aes(x=!!enquo(init)),color="darkturquoise",alpha=0.5) ## force evaluation 
    # if (config$testing=="yes") lines(density(nestData1$init),col=rgb(50,166,168,alpha=0.5))
    }
    if(config$debugNests>=3) cat("\ncreating nVals")
    if(config$debugNests>=3) cat("(par,rep,flood,hatch,disc,excl,unkn,misclass\n")
    if(config$debugNests>=3) cat("avFint,avK,true DSR, true PSR, mayfield, apparent)\n")
    nVal <- mod$calc_nests(nestData1, par, repID, parID,db=config$debugNests)
    names(nVal) <- nval_name
    if(debug>=2) cat("\nnVal:\n")
    if(debug>=2) qvcalc::indentPrint(nVal)

  #---- Reduced nest data: ----------------------------------------------------
    # nestData <- nestData1 |> as.data.frame() |> setNames(colnames) |> filter(totobs!=0) # remove undiscovered nests
    # nestData <- nestData1 |> as.data.frame() |> setNames(colnames) |> filter(totobs>0) # remove undiscovered nests
    nestData <- nestData |> filter(totobs>0) # remove undiscovered nests
    if (config$debugNests>=3) cat("\ndiscovered nests (1 to 15):\n")
    if (config$debugNests>=3) qvcalc::indentPrint(head(nestData,15))
    nestData <- nestData |> filter(afate!=7) # remove undiscovered nests
        # lVal = rep_loop(par=par, nData=nestData, storm=stormDays,
        #            survey=survey,config=config)
        # # llDSR = lVal[0]
        # llDSR,llPSR,llDFR = lVal
    if(debug>=3) cat("\nanalyzed nests (1 to 15):\n")
    if(debug>=3) qvcalc::indentPrint(head(nestData,15))
    if(debug>=4) qvcalc::indentPrint(nestData)

  #---- Program MARK: ----------------------------------------------------
    if(T){
      if (debug>=2) cat("\n<*><*><*> Run RMark <*><*><*><*><*>\n")
      RMark_out <- real_MARK(nestData, db=config$debugM)
      if(debug>=2) cat("\n|> RMark output:\n")
      if(debug>=2) qvcalc::indentPrint(RMark_out)
      if(debug>=2) qvcalc::indentPrint(class(RMark_out))
      RM_dot <-  RMark_out[[1]][,1]
      dotPSR <- RM_dot ^ par$hatchTime
      RM_dsr <- RMark_out[[2]][,1]
      RM_psr <- RM_dsr ^ par$hatchTime
      markVal <- c(RM_dsr,RM_psr)
    }

  #---- MCMC model: ------------------------------------------------------------------------------------------------------

    if(T){
      lVal = withCallingHandlers({mod$rep_loop(par=par,
                          nData=nestData,
                          storm=stormDays,
                          survey=survey,
                          config=config
                          # to_r=TRUE
      )},
      error=function(e){
        skiptoNext <<- TRUE # need to use super-assignment
        message("error in MCMC model: ", e) # print(sys.calls())
        reticulate::py_last_error()
      })
      if(debug>=2) cat("\n|> MCMC model output:\n") # print(class(lVal))
      if(debug>=2) qvcalc::indentPrint(lVal)

      llVal <- py_to_r(lVal$astype("float64")) # when it's a np ndarray, this dosn't work
      if(debug>=3) qvcalc::indentPrint(class(llVal))
      if(debug>=3) qvcalc::indentPrint(llVal)
      # llDSR <- as.numeric(llVal[0]) llPSR <- as.numeric(llVal[1]) llDFR <- as.numeric(llVal[2])

      llDSR <- py_to_r(llVal[[1]]) # list does convert, & needs to be 1-indexed?
      llPSR <- llVal[[2]]
      llDFR <- llVal[[3]]
      if(debug>=3){
        qvcalc::indentPrint(class(llVal))
        qvcalc::indentPrint(llVal)
        qvcalc::indentPrint(class(llDSR))
        qvcalc::indentPrint(llDSR)
      }
      mcmcVal <- c(llDSR,llPSR,llDFR)
  }

  #---- Mayfield: -----------------------------------------------------------------------------------------------
    # if(debug>=3) cat("\n<*><*><*> Mayfield estimate <*><*><*><*><*>\n")
    # mayfDSR <- calc_dsr(nData=nestData,
    #                            nestType="analysis",
    #                              calcType="mayfield",
    #                              conf=config,
    #                              incTime=par.hatchTime,
    #                              psurv=par.probSurv,
    #                              debug=config.debugSummary) 
    #

  #---- Logistic exposure: -----------------------------------------------------------------------------------------------
    if(debug>=2) cat("\n<*><*> Logistic exposure <*><*><*><*>\n")
    excpt <- FALSE
    nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
    nestObs <- nestData |> dplyr::select(ID, init, i, j, k, afate, totobs) # print(head(nestObs))
    expoList   <- logex$calc_daily_expo(numNests=nNest, surveyDays=survey[[1]],
                                     surveyInts=survey[[2]], firstDay=nestData$i,
                                     lastDay=nestData$k, db=config$debugLL)
    dat2S <- logex$make_daily_logex_df(nestObs, expos=expoList[[1]], covar1=expoList[[2]], # all survey dates for all nests
                                        db=config$debugLL) # cat("\n|> made obs data\n") print(obsDat)
    tryCatch({ modOut <- fit_glm(mList,dat=dat2S,debug=config$debugSummary) },
      error = function(e) { 
        message("!! error in glm:", e, "go to next") 
        excpt <<- TRUE
        # coefsArray <- rep(-999, length(coef_names))
        # coefs[,r,i] <- coefsArray
        coefs[,r,i] <- -999
      },
      warning = function(w) { 
        message("!! warning in glm:", w, "go to next") 
        excpt <<- TRUE
        # coefsArray <- rep(-999, length(coef_names))
        # coefs[,r,i] <- coefsArray
        coefs[,r,i] <- -999
      })
    if(excpt) {
      print("exception")
      next
    }
    coefsArray <- get_coef(modOut, debug=config$debugSummary)
    if(config$coefSave!="none") coefs[,r,i] = unlist(coefsArray)
    if (debug>=4) cat("\n\t<> coefficients:\n")
    # if (debug>=4) qvcalc::indentPrint(coefs[,r,i])
    if (debug>=4) qvcalc::indentPrint(coefsArray)

  #---- Logexp DSR & PSR: -----------------------------------------------------------------------------------------------
    dsr1 <-  1/(1+exp(-coefsArray[[1]][1,1]))
    psr1 <- dsr1 ^ par$hatchTime
    dsr2 <- 1/(1+exp(-coefsArray[[6]][1,1] + coefsArray[[6]][1,2] * dat2S$avDate))
    psr2 <- dsr2 ^ par$hatchTime
    allInits <- nestData$init
    numInit <- sapply(dat2S$Date, function(x) sum(allInits==x))
    propInit <- numInit/par$numNests
    propInitScl <- propInit/sum(propInit)
    if(debug>=4){
      cat("\ninits & dates:\n")
      print(allInits)
      print(dat2S$Date)
      # cat("\nnum inits before date:\n")
      cat("\nnum inits on date:\n")
      print(numInit)
      cat("\nproportion inits on date:\n")
      print(propInit)
      print(propInitScl)
    }
    dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugSummary)
    dsrList <- dsrList[-1]
    psrList <- lapply(dsrList, function(x) x^par$hatchTime)
    if(debug>=4){
      cat(sprintf("\noutput of make_pred (dsrList:%s & psrList:%s):\n", length(dsrList), length(psrList)))
      qvcalc::indentPrint(dsrList)
      qvcalc::indentPrint(psrList)
    }
    psr <- make_psr(psrList, propInitScl)
    # nVal <- c(nVal, dsr1, psr1,psr[[1]], psr[[2]],psr[[3]],psr[[4]])
    if(debug>=2){
      cat("\n|> logistic exposure DSR & PSR (no covars):", dsr1, psr1)
      cat("\n|> logistic exposure DSR & PSR (average date):", dsr2,psr2)
      cat("\n|> logistic exposure PSR (mods 2-5):", paste(psr,collapse=" ; "))
    }

    # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
    logexVal <- c(dsr2,psr2,psr[[1]],psr[[2]],psr[[3]],psr[[4]])


  #---- Add to nVal: -----------------------------------------------------------------------------------------------
    nVal <- c(nVal,logexVal,mcmcVal,markVal)
    # nVal <- c(nVal,logexVal,markVal)
    if (debug>=2) cat("\n\n|>|>nVal:\n")
    if(debug>=2) qvcalc::indentPrint(nVal)
    nValMat[,r,i] <- nVal
    if(debug>=2) qvcalc::indentPrint(nValMat[,r,i])


  #---- Finish replicate: -----------------------------------------
    if(config$testing=="yes"){
      # valMat[,r,i]= c(nVal["aDSR"],nVal["leDSR"],nVal['leDSR']-nVal['aDSR'])
      #                       11            13             19              
      # valMat[,r,i]= c(nVal["aDSR"],nVal["leDSR"],nVal["mcmcDSR"],nVal['leDSR']-nVal['aDSR'],nVal["mcmcDSR"]-nVal["aDSR"])
      # pl <- pl + ggplot2::geom_density(data=initDF,ggplot2::aes(x=init),color="darkblue",alpha=0.9)
      # ggplot2::ggsave(plotFile, plot=pl, device="png", width=6,height=4,units="in")
      # dev.off() ## should save the plot opened at beginning of rep

      ## nVal is not named
      aDSR = nVal[11]
      # leDSR = nVal[13]
      leDSR = nVal[15]
      # mcmcDSR = nVal[19]
      # mcmcDSR = nVal[21]
      mcmcDSR = llDSR
      markDSR = RM_dsr
      mayfDSR = nVal["mfDSR"]

      valMat[,r,i]= c(aDSR,leDSR,mcmcDSR,markDSR,mayfDSR,leDSR-aDSR,mcmcDSR-aDSR,markDSR-aDSR,mayfDSR-aDSR)
      # valMat[,r,i]= c(aDSR,leDSR,leDSR-aDSR)
      if(debug>=4)  cat("\nstore summary vals:\n")
      if(debug>=4)  qvcalc::indentPrint(valMat)
    }
    repID = repID + 1
  }

#---- Finish param set: -----------------------------------------
  # if (as.numeric(parID) %% 5 == 0){
  if (as.numeric(parID) %% 50 == 0){
    parStart = parID - 49
    # parStart = as.numeric(parID) - 4
    fname <- sprintf("%s/nval_%sto%s.rds", outdir, parID-49,parID)
    saveMat <- nValMat[,,c(parStart:parID)]
    # print("incremental save:")
    # print(fname)
    cat(sprintf("\n** incremental save from param set %s to %s (%s)", parStart,parID, fname))
    # print(saveMat)
    saveRDS(saveMat, fname)
  }
  parID = parID + 1
}

nvalname <- sprintf("%s/nval_r.rds", outdir)
saveRDS(nValMat, nvalname)

if (debug>=2) cat("\nCoefficients:\n")
if (debug>=2) qvcalc::indentPrint(coefs)

if (debug>=2) cat("\nN Val:\n")
if (debug>=2) qvcalc::indentPrint(nValMat)

summ <- apply(valMat,c(1,3),mean)
if(debug>=4)  qvcalc::indentPrint(valMat)
if(debug>=2) cat("\nSummary:\n")
if(debug>=2)  qvcalc::indentPrint(summ)
