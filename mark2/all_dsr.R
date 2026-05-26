
## attempting to streamline
## takes parts of logexp.R and datsim.py 

library(MASS)
library(RMark)
library(dplyr) # load dplyr last so as not to mask select?
library(brglm2)
library(reticulate)
#NOTE could make a counter of all times at least one survey int == 0

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
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur, stFromFile=sett$stormFromFile)
  survey    <- withCallingHandlers(
                                   {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=config)},
                                   error=function(e){ 
                                     reticulate::py_last_error() 
                                     # print(sys.calls()) # doesn't help if error in python
                                   }  )


#----------------------------------------------------
  repID=0
  for(r in seq(nreps)){
    cat(sprintf("\n:::::::::::::::::::::::::::::: rep %s-%s :::::::::::::::::::::::::::::::::::::::::::\n",i,r))

  #---- Full nest data: ----------------------------------------------------
    skiptoNext <- FALSE
    nestData1 <- withCallingHandlers({
      nweeks = round(par$brDays/7)-2
      # nweeks = floor(par$brDays/7)
      # nweeks = par$brDays//7
      obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }
    if (config$debugNests>=3){
      cat("\n\t|>creating nVals")
      cat("(par,rep,flood,hatch,disc,excl,unkn,misclass\n")
      cat("\t\t\tavFint,avK,true DSR, true PSR, mayfield, apparent)\n")
    }
    nVal <- mod$calc_nests(nestData1, par, repID, parID,db=config$debugNests)
    colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)
    if(debug>=3) cat("\n<*><*><*> NEST DATA: <*><*><*><*><*>\n") # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
    if(debug>=4) cat("\n\t** all nest data:\n") # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
    if(debug>=4) qvcalc::indentPrint(nestData1) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)
    names(nVal) <- nval_name
    if(debug>=2) cat("\n\t** nVal:\n")
    if(debug>=2) qvcalc::indentPrint(nVal)

  #---- Reduced nest data: ----------------------------------------------------
    nestData <- nestData1 |> filter(totobs>0) # remove undiscovered nests
    if (config$debugNests>=3) cat("\n\t** discovered nests (1 to 15):\n")
    if (config$debugNests>=3) qvcalc::indentPrint(head(nestData,15))
    nestData <- nestData |> filter(afate!=7) |> na.omit() # remove undiscovered nests
    if(debug>=3) cat("\n\t** analyzed nests (1 to 15):\n")
    if(debug>=3) qvcalc::indentPrint(head(nestData,15))
    if(debug>=4) qvcalc::indentPrint(nestData)


  #---- Program MARK: ----------------------------------------------------
    if(T){
      # if (debug>=2) cat("\n<*><*><*> Run RMark <*><*><*><*><*>\n")
      RMark_out <- real_MARK(nestData, db=config$debugM)
      mdotDSR <-  RMark_out[[1]][,1]
      mdotPSR <- mdotDSR ^ par$hatchTime
      mdateDSR <- RMark_out[[2]][,1]
      mdatePSR <- mdateDSR ^ par$hatchTime
      mdsAgeDSR <- RMark_out[[3]][,1]
      mdsAgePSR <- mdsAgeDSR ^ par$hatchTime
      marktop <- RMark_out[["top"]][,1]
      topname <- RMark_out[["topname"]]
      markVal <- c(mdotDSR,mdotPSR,mdateDSR,mdatePSR,mdsAgeDSR,mdsAgePSR,marktop,topname)
      # if(debug>=2) cat("\n\t|> RMark output:",mdotDSR,mdotPSR,mdateDSR,mdatePSR,mdsAgeDSR,mdsAgePSR,"\n")
      if(debug>=2) cat("\n\t|> RMark output:",mdotDSR,mdotPSR,mdateDSR,mdatePSR,mdsAgeDSR,mdsAgePSR)
      if(debug>=3) qvcalc::indentPrint(RMark_out)
      if(debug>=3) qvcalc::indentPrint(class(RMark_out))
    }

  #---- MCMC model: ------------------------------------------------------------------------------------------------------

    if(TRUE){
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
      if(debug>=3) cat("\n\t\t|>all MCMC model output:\n") # print(class(lVal))
      if(debug>=3) qvcalc::indentPrint(lVal)

      llVal <- py_to_r(lVal$astype("float64")) # when it's a np ndarray, this dosn't work
      if(debug>=3) qvcalc::indentPrint(class(llVal))
      if(debug>=3) qvcalc::indentPrint(llVal)
      # llDSR <- as.numeric(llVal[0]) llPSR <- as.numeric(llVal[1]) llDFR <- as.numeric(llVal[2])

      llDSR <- py_to_r(llVal[[1]]) # list does convert, & needs to be 1-indexed?
      llPSR <- llVal[[2]]
      llDFR <- llVal[[3]]
      if(debug>=5){
        qvcalc::indentPrint(class(llVal))
        qvcalc::indentPrint(llVal)
        qvcalc::indentPrint(class(llDSR))
        qvcalc::indentPrint(llDSR)
      }
      if(debug>=2) cat("\n\t|> MCMC DSR & PSR = ", llDSR,llPSR)
      mcmcVal <- c(llDSR,llPSR,llDFR)
  }

  #---- Calculate true DSR: ----------------------------------------------------
    if(TRUE){
      truePSR <- nVal["hat"] / par$numNests
      # if(debug>=2) cat(sprintf("\n\t|>|> true PSR: %s [num hatched] / %s [num total] = %s):", nVal["hat"],par$numNests,truePSR))
      if(debug>=2) cat(sprintf("\n\toooo|> true PSR: %s [num hatched] / %s [num total] = %s):", nVal["hat"],par$numNests,truePSR))

      ## needs to be all nests and all days (not just observed)
      # coef_out <- calc_logexp(mList, nestData, survey=survey, config=config, exposure=1, debug=config$debug) 
      modData <- mk_logex_data( nestData1, survey=survey, config=config, exposure=1 ) 
      # coef_out <- calc_logexp(mList, nestData1, survey=survey, config=config, exposure=1, debug=config$debug) 
      coef_out <- calc_logexp(mList, modData, config=config) 
      if(debug>=3) cat("\n\t\t>-> coef_out:\n")
      if(debug>=3) qvcalc::indentPrint(coef_out)

      dsrT <-  1/(1+exp(-coef_out[[1]][1,1]))
      psrT <- dsrT ^ par$hatchTime
      dsrTrue <- c(dsrT,psrT)
      if(debug>=2) cat("\n\t|> true DSR & PSR: logistic exposure (no covars):", dsrT, psrT)

    }

  #---- Logistic exposure: -----------------------------------------------------------------------------------------------
    if(TRUE){
      modData <- mk_logex_data( nestData, survey=survey, config=config, exposure=0) 
      modOut <- calc_logexp(mList,modData,config=config)
      # if(debug>=2) cat("\n<*><*> Logistic exposure <*><*><*><*>\n")
      # excpt <- FALSE
      nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
      # # nestObs <- nestData |> dplyr::select(ID, init, i, j, k, afate, totobs) # print(head(nestObs))
      nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) # print(head(nestObs))
      numObs <- nestData[,"totobs"]
      dat2S = modData

      ## use coefs from full data (true DSR) and 
      # dsrT2        <- 1/(1+exp(-coef_out[[6]][1,1] + coef_out[[6]][1,2] * dat2S$avDate))
      # psrT2        <- dsr2 ^ par$hatchTime
      # psrT2        <- psr2[1]
      # dsrT3        <- 1/(1+exp(-coef_out[[6]][1,1] + coef_out[[6]][1,2] * dat2S$avAge))
      # psrT3        <- dsr3 ^ par$hatchTime
      # psrT3        <- psr3[1]
      # if(debug>=2) cat("\n|> true DSR & PSR: logistic exposure (av date; av age):", dsrT2, psrT2, dsrT3, psrT3)
      # dsrTrue <- c(dsrT,psrT,dsrT2,psrT2,dsrT3,psrT3)

      # expoList   <- logex$calc_daily_expo(numNests=nNest, surveyDays=survey[[1]],
      #                                  surveyInts=survey[[2]], firstDay=nestData$i,
      #                                  lastDay=nestData$k, db=config$debugLL)
      # # # dat2S <- logex$make_daily_logex_df(nestObs, expos=expoList[[1]], covar1=expoList[[2]], # all survey dates for all nests
      # # #                                     db=config$debugLL) # cat("\n|> made obs data\n") print(obsDat)
      # dat2S <- logex$make_daily_logex_df(nestObs,
      #                                    nObs=numObs,
      #                                    expos=expoList[[1]],
      #                                    covar1=expoList[[2]], # all survey dates for all nests
      #                                    db=config$debugLL) # cat("\n|> made obs data\n") print(obsDat)
      # tryCatch({ modOut <- fit_glm(mList,dat=dat2S,debug=config$debugSummary) },
      #   error = function(e) { 
      #     message("!! error in glm:", e, "go to next") 
      #     excpt <<- TRUE
      #     # coefsArray <- rep(-999, length(coef_names))
      #     # coefs[,r,i] <- coefsArray
      #     # coefs[,r,i] <- -999
      #     modOut <-
      #   },
      #   warning = function(w) { 
      #     message("!! warning in glm:", w, "go to next") 
      #     excpt <<- TRUE
      #     # coefsArray <- rep(-999, length(coef_names))
      #     # coefs[,r,i] <- coefsArray
      #     coefs[,r,i] <- -999
      #   })
      # if(excpt) {
      #   print("exception")
      #   next
      # }
      if(any(modOut=="exception")){
        cat("  go to next ~~")
      #     # coefs[,r,i] <- coefsArray
        next
      }
      ## output is already coefsArray
      # coefsArray <- get_coef(modOut, debug=config$debugSummary)
      # if(config$coefSave!="none") coefs[,r,i] = unlist(coefsArray)
      if(debug>=3) cat("\n\tmodOut:")
      if(debug>=3) qvcalc::indentPrint(modOut)
      if(config$coefSave!="none") coefs[,r,i] = unlist(modOut)
      coefsArray = modOut
      # if (debug>=4) cat("\n\t<> coefficients:\n")
      # # if (debug>=4) qvcalc::indentPrint(coefs[,r,i])
      # if (debug>=4) qvcalc::indentPrint(coefsArray)
    }

  #---- Logexp DSR & PSR: -----------------------------------------------------------------------------------------------
    dsr1        <-  1/(1+exp(-coefsArray[[1]][1,1]))
    psr1        <- dsr1 ^ par$hatchTime
    dsr2        <- 1/(1+exp(-coefsArray[[6]][1,1] + coefsArray[[6]][1,2] * dat2S$avDate))
    psr2        <- dsr2 ^ par$hatchTime
    psr2        <- psr2[1]
    # print(psr2)
    dsr3        <- 1/(1+exp(-coefsArray[[6]][1,1] + coefsArray[[6]][1,2] * dat2S$avAge))
    psr3        <- dsr3 ^ par$hatchTime
    psr3        <- psr3[1]

    allInits    <- nestData$init
    numInit     <- sapply(dat2S$Date, function(x) sum(allInits==x))
    propInit    <- numInit/par$numNests
    propInitScl <- propInit/sum(propInit)

    if(debug>=4){
      # cat("\nlength of dsr2:\n", length(dsr2))
      cat("\n\t\t>-> inits & dates:\n")
      qvcalc::indentPrint(allInits)
      qvcalc::indentPrint(dat2S$Date)
      # cat("\nnum inits before date:\n")
      cat("\n\t\t>-> num inits on date:\n")
      qvcalc::indentPrint(numInit)
      cat("\n\t\t>-> proportion inits on date:\n")
      qvcalc::indentPrint(propInit)
      qvcalc::indentPrint(propInitScl)
    }
    dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugSummary)
    dsrList <- dsrList[-1]
    psrList <- lapply(dsrList, function(x) x^par$hatchTime)
    if(debug>=4){
      cat(sprintf("\n\t\t\t|> output of make_pred (dsrList:%s & psrList:%s):\n", length(dsrList), length(psrList)))
      qvcalc::indentPrint(dsrList)
      qvcalc::indentPrint(psrList)
    }
    psr     <- make_psr(psrList, propInitScl, db=config$debugLL) # nVal <- c(nVal, dsr1, psr1,psr[[1]], psr[[2]],psr[[3]],psr[[4]])
    if(debug>=2){
      cat("\n\t|> logistic exposure DSR & PSR (no covars):", dsr1, psr1)
      # cat("\n|> logistic exposure DSR & PSR (average date):", dsr2,psr2)
      cat("\n\t|> logistic exposure PSR (mods 2-5):", paste(psr,collapse=" ; "))
      # cat("\n\t|> logistic exposure PSR (av date; av age):", dsr2,psr2,dsr3,psr3)
    }

    # logexVal <- c(dsr1,psr1,dsr2,psr2,psr[[1]],psr[[2]],psr[[3]],psr[[4]]) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
    # logexVal        <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]],dsrT,psrT) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
    logexVal        <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]]) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
    # logexSupp       <- c(dsr2,psr2,dsr3,psr3)
    names(logexVal) <- lexp_name
    # names(logexSupp) <- lexp_supp


  #---- Add to DSR matrix: -----------------------------------------------------------------------------------------------
    # nVal <- c(nVal,logexVal,mcmcVal,markVal)
    # dVal <- c(dsrTrue,logexVal,logexSupp,mcmcVal,markVal)
    dVal <- c(dsrTrue,logexVal,mcmcVal,markVal)
    # nVal <- c(nVal,logexVal,markVal)
    # cat("\ndsrMat for this rep & par set:")
    # print(dsrMat[,r,i])
    # cat("\ndVal:")
    # print(dVal)
    dsrMat[,r,i] <- dVal
    nValMat[,r,i] <- nVal

    if (debug>=3) cat("\n\n\t|>|>dsr/psr Val:\n")
    if(debug>=3) qvcalc::indentPrint(dVal)
    if(debug>=4) qvcalc::indentPrint(dsrMat[,r,i])

    if (debug>=3) cat("\n\n\t|>|>nVal:\n")
    if(debug>=3) qvcalc::indentPrint(nVal)
    if(debug>=4) qvcalc::indentPrint(nValMat[,r,i])


  #---- Finish replicate: -----------------------------------------
    if(config$testing=="yes"){
      aDSR = nVal["aDSR"]
      leDSR = dsr1
      # valMat[,r,i]= c(nVal["aDSR"],nVal["leDSR"],nVal['leDSR']-nVal['aDSR'])
      #                       11            13             19              
      # valMat[,r,i]= c(nVal["aDSR"],nVal["leDSR"],nVal["mcmcDSR"],nVal['leDSR']-nVal['aDSR'],nVal["mcmcDSR"]-nVal["aDSR"])
      # pl <- pl + ggplot2::geom_density(data=initDF,ggplot2::aes(x=init),color="darkblue",alpha=0.9)
      # ggplot2::ggsave(plotFile, plot=pl, device="png", width=6,height=4,units="in")
      # dev.off() ## should save the plot opened at beginning of rep

      ## nVal is not named
      ##now it is 
      # aDSR = nVal[11]
      # trueDSR <- 
      # leDSR = nVal[13]
      # leDSR = nVal[15]
      # mcmcDSR = nVal[19]
      # mcmcDSR = nVal[21]
      mcmcDSR = llDSR
      # markDSR = RM_dsr
      markDSR = mdotDSR
      mayfDSR = nVal["mfDSR"]

      # valMat[,r,i]= c(aDSR,leDSR,mcmcDSR,markDSR,mayfDSR,leDSR-aDSR,mcmcDSR-aDSR,markDSR-aDSR,mayfDSR-aDSR)
      # vals         <- c(dsrT,leDSR,letop,mcmcDSR,markDSR,marktop,mayfDSR)
      # vals         <- c(dsrT,leDSR,mcmcDSR,markDSR,marktop,mayfDSR)
      vals         <- c(dsrT,aDSR,leDSR,mcmcDSR,markDSR,marktop,mayfDSR)
      diffs        <- c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,markDSR-dsrT,marktop-dsrT,mayfDSR-dsrT)
      valMat[,r,i] <- c(vals,diffs)
      # valMat[,r,i]= c(aDSR,leDSR,leDSR-aDSR)
      if(debug>=4)  cat("\nstore summary vals:\n")
      if(debug>=4)  qvcalc::indentPrint(valMat)
    }
    repID = repID + 1
  }

#---- Finish param set: -----------------------------------------
  # if (as.numeric(parID) %% 5 == 0){
  if (as.numeric(parID) %% 50 == 0){
    parStart = parID - 49 # parStart = as.numeric(parID) - 4
    fname <- sprintf("%s/nval_%sto%s.rds", outdir, parID-49,parID)
    saveMat <- nValMat[,,c(parStart:parID)] # print("incremental save:") print(fname)
    cat(sprintf("\n** incremental save from param set %s to %s (%s)", parStart,parID, fname))
    saveRDS(saveMat, fname) # print(saveMat)
  }
  parID = parID + 1
}

dsrvalname <- sprintf("%s/dsrval.rds", outdir)
saveRDS(dsrMat, dsrvalname)

nvalname <- sprintf("%s/nval.rds", outdir)
saveRDS(nValMat, nvalname)

if (debug>=2) cat("\nCoefficients:\n")
if (debug>=2) qvcalc::indentPrint(coefs)

if (debug>=2) cat("\nDSR Val:\n")
if (debug>=2) qvcalc::indentPrint(dsrMat)

if (debug>=2) cat("\nN Val:\n")
if (debug>=2) qvcalc::indentPrint(nValMat)

summ <- apply(valMat,c(1,3),mean,na.rm=TRUE) ## pass args to mean after function itself
# if(debug>=4)  qvcalc::indentPrint(valMat)
if(debug>=2) cat("\nSummary (mean for each param set):\n")
if(debug>=2)  qvcalc::indentPrint(summ)
