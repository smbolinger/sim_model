#---- LOAD FUNCTIONS & VARIABLES --------------------------------------------------------------

source("setup.R")
source("dsr_fun.R")

#---- LOOP THRU PARAM SETS ----------------------------------------------------------------------

parID = startParID

for(i in seq(startParID,length(pArrList))){

  #---- Make params, storms, surveys: --------------------------------------------------------------
  startTimePar <- Sys.time()
  rng <- np$random$default_rng(seed=rngSeed)
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1],debug=FALSE)},
                  error=function(e){
                  reticulate::py_last_error()
                  })

  cat(sprintf("\n\n.....%s......i=%03d.......seed=%d...........................................................................................................\n",format(startTimePar, "%H:%M:%S"),i,rngSeed))
  print(unlist(py_vars(par))) # print(class(par))
  cat(".....................................................................................................................................................\n")

  if(!vals$uniqueStorm){ 
    if(debug>=2) cat("\n\tstorm data, class",class(stormDat),":")
    if(debug>=2) qvcalc::indentPrint(stormDat)
    stormDays <- funs$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=vals$stormFromFile,db=TRUE)
    if(debug>=2) cat("\n\t>> storm days = ",stormDays)
    survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,complicate=TRUE,db=debug)},
                                     error=function(e){ 
                                       reticulate::py_last_error() # print(sys.calls()) # doesn't help if error in python
                                     }  )
  }

#---- LOOP THRU REPLICATES ------------------------------------------------------------------------------

  repID=0

  for(r in seq(nreps)){
    if(debug>=1) cat(sprintf("\n:::::::::::::::::::::::::::::: rep %s",i))
    cat(sprintf("-%s",r))
    if(debug>=1) cat(":::::::::::::::::::::::::::::::::::::::::::\n")

    if(vals$uniqueStorm){ # if(debug>=2) cat("\n\t>> overwriting storm days -")
      stormDays <- funs$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=vals$stormFromFile,db=TRUE)
      if(debug>=2) cat("\n\t>> storm days = ",stormDays)
      if(config$testing=="yes") stormDates <- c(stormDates, stormDays)
      survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,complicate=TRUE,db=debug)},
                                       error=function(e){ 
                                         reticulate::py_last_error() 
                                       }  )
    }
    skiptoNext <- FALSE ## whether or not to skip to next replicate

  #---- Full nest data: ----------------------------------------------------

    nweeks = round(par$brDays/7)-2 # nweeks = floor(par$brDays/7)

    # run the nest model and the observer model
    nestData1 <- withCallingHandlers({
      # make true nest histories: 
      nData <- nest$mk_histories(stormDays,initDat,par,rng,pyconfig,vals$initFromFile,nweeks)

      # make observation histories:
      obs$make_obs(nData,par,rng,obsVarNum,stormDays,survey,pyconfig,initDat,stormUnk,nweeks,vals$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      save_current(startParID, i, odir)
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }

    nVal <- obs$calc_nests(nestData1, par, rng,survey, obsVarNum, repID, parID,pyconfig,db=config$debugNests)
    names(nVal) <- nval_name
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)

    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if (config$testing=="yes") {
        all_hatch    <- sum(nestData1$fate==0, na.rm=TRUE)
        all_fld      <- sum(nestData1$fate==2, na.rm=TRUE)
      }
      if(config$obsSave){
        write(stormDays, file="out/storm_plot.txt", sep="\t", append=TRUE, ncolumns = 10)
        write(nestData1$init, file="out/init_plot.txt", sep="\t", append=TRUE, ncolumns = par$numNests)
      }
      if (saveNestData){
        nestDataMat[,,r,i] <- as.matrix(nestData1)
      }
    #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Nest data - discovered: ----------------------------------------------------
    nestData2 <- nestData1 |> filter(.data[[vals$obsVar]]>0) # remove undiscovered nests
    # obsData <- obs$make_obs(nestData,par,rng,obsVarNum,stormDays,survey,pyconfig,initDat,stormUnk,nweeks,sett$initFromFile,pandas=FALSE)
    ## why re-assign names?
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)

  #---- Nest data - analyzed: ----------------------------------------------------
    nestData <- nestData2 |> filter(afate!=7) 
    nestData <- nestData |> filter(end>i) ## remove nests found on end date (should be none?)

    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    if(config$testing=="yes"){

      print_ndata(nestData1, nestData2, nestData, debug=config$debugNests)
      num_disc      <- nrow(nestData)
      num_excl      <- sum(nestData$afate==7, na.rm=TRUE)
      num_misclass  <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
      num_misclass  <- num_misclass-num_excl
      prop_excl     <- num_excl/num_disc
      prop_misclass <- num_misclass/num_disc

    }
    #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rm(nestData2)

  #---- Calculate true DSR: ----------------------------------------------------
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] True DSR - make nest data [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)

      prDays_true  <- seq(1, max(nestData1$init)) # if(debug>=4) cat("\n\tdates for prediction: ", prDays, length(prDays))
      simplePSR <- nVal["hat"] / par$numNests

      # newDat <- expand.grid(Date=prDays, Age=seq(par$hatchTime))
      newDat <- data.frame(Date=prDays)
      nVal_true <- nrow(newDat)
      if(debug>=2) cat("\n>>> number of rows in newDat:", nVal_true)

      trueDSRlist <- mk_true_dsr(nestData1, mList_true, newDat, par, config)

      numInit     <- sapply(prDays, function(x) sum(nestData1$init==x))
      propInit    <- numInit/par$numNests
      propInitScl_true <- propInit/sum(propInit) ## make sure it sums to 1
      if(debug>=3){
        cat("\nsize of trueDSRlist:", length(trueDSRlist))
        cat("\nsize of trueDSRlist sublists:", paste(unlist(lapply(trueDSRlist,length)),collapse=" "))
        cat("\n& size of trueDSRmat:", dim(trueDSRmat[,c(1:nVal_true),r,i]))

        qvcalc::indentPrint(head(trueDSRmat[,c(1:nVal_true),r,i], 20))
        # qvcalc::indentPrint(head(trueDSRmat[c(1:nVal_true),,r,i], 20))
      }
      trueDSRmat[1,c(1:nVal_true),r,i] <- newDat$Date
      trueDSRmat[c(2:3),c(1:nVal_true),r,i] <- matrix(unlist(trueDSRlist), nrow=length(trueDSRlist), byrow=TRUE)
      if(debug>=3) qvcalc::indentPrint(head(trueDSRmat[,c(1:nVal_true),r,i], 20))
      if(config$obsSave){
        dsrplot_dat <- c(par$hatchTime, par$stormFrq, par$stormDur, par$pMortFl, par$probSurv, unlist(trueDSRlist[[2]]))
        # write(psrplot_dat, file="out/psr_plot.txt", sep="\t", append=TRUE, ncolumns=130)
        write(dsrplot_dat, file="out/dsr_plot.txt", sep="\t", append=TRUE, ncolumns=130)
        cat("\n\t>> saved true DSR to file for plotting")

        # propInitmat[,r,i] <- propInitScl     
      }

      dsrT = trueDSRlist[[1]][1]
      psrT = dsrT ^ par$hatchTime
      # psrT_date = make_psr(trueDSRlist[[2]],nestData1$init,prDays_true,par,config)
      out_true = make_weighted(mList_true[2],trueDSRlist[[2]],list(),nestData1$init,prDays,par,config,newDat)
      dsrT_date = out_true[[1]]
      psrT_date = out_true[[2]]
      rm(trueDSRlist)

      #-*~~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){

        if(debug>=3) cat(sprintf("\n\t|> apparent PSR: %s [num hatch]/%s [num total] = %s):", nVal["hat"],par$numNests,simplePSR))
        if(debug>=2) cat("\t|> true DSR, logit <date>:", dsrT_date)
        if(debug>=2) cat("\t|> true PSR, logit <date>:", psrT_date)
        if(debug>=3) cat("\t|> true DSR & PSR, logit <null>:",dsrT,psrT)
      }
      #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      trueDSR = ifelse(vals$dsrTrueVal=="date",dsrT_date, dsrT)

  #---- Logistic exposure: -----------------------------------------------------------------------------------------------

    if(config$logex){
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      dat2S <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, expoVal=0) 
      n_obs <- nrow(dat2S) # if (debug>=2) cat("\n\t\t\tnumber of obs=", n_obs)
      if(config$testing=="yes") obsLength <- c(obsLength, nrow(dat2S))
      if(config$testing=="yes") obsIntList <- c(obsIntList, max(dat2S$Exposure))
      modFit <- calc_logexp(mList, dat2S, config=config)
      if(any(modFit=="exception")){
        cat("all_dsr.R: exception -  go to next ~~")
        save_current(startParID, i, odir)
        skiptoNext <<- TRUE # need to use super-assignment
      }
    }

    ## if error, won't be able to get coefs, so go to next
    if(skiptoNext) { next }

    if(config$logex){
      coeff <- rep(NA, times=length(mList)) ## numeric vector of length x
      coeffSE <- rep(NA, times=length(mList)) ## numeric vector of length x
      vcOut <- rep(NA, times=length(mList)) ## numeric vector of length x

      coeff  <- withCallingHandlers({sapply(modFit, function(x) coef(summary(x))[,"Estimate"])},
        error = function(e){ "\t!! couldn't get coefs; error: e" })
      coeffSE  <- withCallingHandlers({sapply(modFit, function(x) coef(summary(x))[,"Std. Error"])},
        error = function(e){ "\t!! couldn't get coef SE; error: e" })
      vcOut <- withCallingHandlers({lapply(modFit, function(x) vcov(x))}, # get variance-covariance matrix
        error = function(e){ "\t!! couldn't get vcov matrix; error: e" })

      vcovMatMat[,r,i] <- unlist(vcOut)
      numInit     <- sapply(prDays, function(x) sum(nestData1$init==x))
      propInit    <- numInit/par$numNests
      propInitScl <- propInit/sum(propInit) ## make sure it sums to 1
      initPropMat[,r,i] <- propInitScl
      if(debug>=4){
        cat("\n\tcoefs & se:\n")
        qvcalc::indentPrint(coeff)
        qvcalc::indentPrint(unlist(coeffSE))
        cat("\n\t& vcov matrices:\n")
        qvcalc::indentPrint(vcOut)
      }
      if(config$coefSave=="all"){
        if(debug) cat("saving coefficients for rep to matrix")
        coefsMat[,r,i] <- c(unlist(coeff),
                            unlist(coeffSE),
                            mean(dat2S$Exposure,na.rm=TRUE),
                            mean(dat2S$Age,na.rm=TRUE),
                            # mean(dat2S$AvDate,na.rm=TRUE)) ## no idea why this doesn't work - oh, bc I'm dumb
                            mean(dat2S$avDate,na.rm=TRUE)) ## no idea why this doesn't work - oh, bc I'm dumb
                            # newDat$AvDate[1])
      }
      # if(config$survSave=="all"){
      if(config$predict){
        ## DATA FOR PREDICTIONS:
        ##NOTE: use all vals for date in all of them bc that's the var I want to plot
        newDat_age <- expand.grid(
                                  Date=mean(prDays),
                              Age = seq(par$hatchTime),
                               # avAge=mean(dat2S$Age,na.rm=TRUE),
                               avDate=mean(dat2S$avDate,na.rm=TRUE)) # byVal = 4
        newDat_date <- expand.grid(Date=prDays,
                              # Age = seq(par$hatchTime),
                               Age=mean(dat2S$Age,na.rm=TRUE),
                               avDate=mean(dat2S$avDate,na.rm=TRUE)) # byVal = 4
        newDat_all <- expand.grid( Date=prDays,
                              Age = seq(par$hatchTime),
                              # Date=prDays, ## move after age so that age cycles first
                               # avAge=mean(dat2S$Age,na.rm=TRUE),
                               avDate=mean(dat2S$avDate,na.rm=TRUE)) # byVal = 4
        DSR_out <- lapply(seq_along(modFit), function(x){
                            if(x==4){
                              newDat <- newDat_all
                            } else if(x==3){
                              newDat <- newDat_age
                            } else {
                              newDat <- newDat_date
                            }
                            if(debug>=3) cat("\n\t\t\t NEW DAT:")
                            if(debug>=3) qvcalc::indentPrint(head(newDat,30), indent=8)
                            make_preds(coeff[[x]],vcOut[[x]],mList[x],newDat,db=config$debugLogEx)
                            })
        if(debug>=5){
          cat("\n\t>> DSR_out:\n") # lists of DSR and SE values at all data combinations in newDat
          qvcalc::indentPrint(DSR_out)
          qvcalc::indentPrint(DSR_out[[1]])
          qvcalc::indentPrint(DSR_out[[1]][[2]])
        }
        DSRlist <- lapply(DSR_out, '[[', 1)
        seList <- lapply(DSR_out, '[[', 2)
        if(config$debug>=3){
          cat("\n\n\t\t\t>> PRINT DSRlist, seList:")
          lapply(seq_along(DSRlist), function(x){
                   dsr = DSRlist[[x]]
                   cat("\n",mList[x],":",c(head(dsr)), "...",c(tail(dsr), length(dsr)))
                            })
          lapply(seq_along(seList), function(x){
                   se = seList[[x]]
                   cat("\n",mList[x],":",c(head(se)),"...", c(tail(se), length(se))) 
                            })
        }


        out_list <- sapply(seq(length(DSRlist)), function(x){
                     make_weighted(mList[x],DSRlist[[x]], seList[[x]],
                                   # nestData$init, dat2S$Date,
                                   nestData$init, newDat$Date,
                                   par, config,newDat)
                                  })

        if(debug>=5) print(out_list)
        if(debug>=5) print(class(out_list))

        dsr_list <- unlist(out_list[1,])
        psr_list <- unlist(out_list[2,])
        se_list <- unlist(out_list[3,])
        cov_list <- c()
        logexVal <- unlist(tibble::lst(dsr_list,psr_list,se_list,cov_list))
        #-*~~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(config$testing=="yes"){
          # print(class(dsr_list))
          dsr1 <- dsr_list[1]
          # print(class(dsr1))
          psr1 <- psr_list[1]
          se1 <- se_list[1]
          if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] logistic exposure - output [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
          if(config$debug>=3){
            cat("\n\t\t>> logistic exposure DSR (avg weighted by inits per day): ",class(dsr_list), unlist(dsr_list), "\n")
            cat("\n\t\t>> logistic exposure PSR (avg weighted by inits per day): ",class(psr_list), unlist(psr_list), "\n")
            cat("\n\t|> logistic exposure DSR & PSR <null> & s.e.:", dsr1, psr1, se1,"\n")
          }
        }
      } else{
        logexVal <- c() ## return nothing if logex is true but surv save is false
      }
    } else{
      logexVal <- c() ## return nothing if logex is false
    }

  #---- MCMC model: ------------------------------------------------------------------------------------------------------

    if(config$mcmc){
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] MCMC model [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      obsSel <- nestData |> select('ID','init','end','fate','i','j','k','afate','totobs')
      jacPlSuf <- sprintf("%s-%s",i,r)
      withCallingHandlers({
        if (atype %in% c("control"))
          mlOut <- mlfun$PolyMort(obsSel,survey,par,rng,atype,pyconfig,suff=jacPlSuf,db=config$debugLL,scl_fac=0.20)
        else
          mlOut <- mlfun$PolyMort(obsSel,survey,par,rng,atype,pyconfig,suff=jacPlSuf,db=config$debugLL)
      },
      error=function(e){
        message("\nerror in MCMC model: ", e, "; try higher outLen") # print(sys.calls())
        reticulate::py_last_error()
        withCallingHandlers({
          if (atype=="control")
            mlOut <- mlfun$PolyMort(obsSel,survey,par,rng,atype,pyconfig,suff=jacPlSuf,db=config$debugLL,scl_fac=0.15,plt=TRUE)
          else
            mlOut <- mlfun$PolyMort(obsSel,survey,par,rng,atype,pyconfig,suff=jacPlSuf,db=config$debugLL,plt=TRUE)
        },
        error=function(e){
          skiptoNext <<- TRUE # need to use super-assignment
          message("\nerror in MCMC model: ", e, "; go to next") # print(sys.calls())
          save_current(startParID, i, odir)
          reticulate::py_last_error()
        })
      })
    }
    if(skiptoNext) { next }
    if(config$mcmc){
      mcmcDSR = mlOut[[1]][1]
      mcmcDSR_se = mlOut[[3]][1]
      mcmcPSR = mcmcDSR ^ par$hatchTime
      mcmcDPR = mlOut[[2]][1]
      mcmcDPR_se = mlOut[[3]][1]
      mcmc_ucl = mcmcDSR + mcmcDSR_se*1.96
      mcmc_lcl = mcmcDSR - mcmcDSR_se*1.96
      mcmcDSR_cov = as.numeric(trueDSR<mcmc_ucl & trueDSR>mcmc_lcl)
      # mcmcVal <-  unlist(tibble::lst())
      mcmcVal <- unlist(tibble::lst(mcmcDSR, mcmcPSR, mcmcDPR, mcmcDSR_se, mcmcDPR_se))
      # mcmcVal <- c(mcmcDSR, mcmcPSR, mcmcDPR, mcmcDSR_se, mcmcDPR_se, mcmcDSR_cov)
      # cat("\n\t\t>> all_dsr: output from matlab function:", mcmcVal)
      if(config$testing=="yes"){
        if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] MCMC model - output [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
        if(debug>=3) cat("\n")
        if(debug>=2) cat("\t|> MCMC DSR & PSR = ", mcmcVal)
      }
    } else {
      # mcmcVal = rep(-999, 5)
      mcmcVal <- c() 
    }

  #---- Mayfield: -----------------------------------------------------------------------------------------------
    if(TRUE){
      exposureNorm <- sum(nestData$j - nestData$i,na.rm=TRUE)
      exposureFinal <- sum(nestData$k - nestData$j,na.rm=TRUE) * 0.5
      exposure1 <- exposureNorm + exposureFinal
      numFail  <- sum(nestData$afate!=0,na.rm=TRUE)
      mayfDSR <- 1 - (numFail/exposure1)
      mayfPSR <- mayfDSR ^ par$hatchTime
      mayfVar <- ((exposure1-numFail)*numFail)/(exposure1**3)
      mayfSE <- sqrt(mayfVar)
    }

  #---- Add to DSR matrix: -----------------------------------------------------------------------------------------------
    mayfVal = unlist(tibble::lst(mayfDSR,mayfPSR,mayfVar,mayfSE,simplePSR))
    dsrTrue = unlist(tibble::lst(dsrT,psrT,dsrT_date,psrT_date))
    dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal)
    if(config$testing=="yes"){
      if(debug>=2) cat(sprintf("\t>> Mayfield DSR: 1 - (%s[numFail]/%s[exposure]) = %s", numFail, exposure1,mayfDSR))
      if(debug>=3) cat(sprintf("\t>> Mayfield standard error: %s", mayfSE))
      if(debug>=2) cat("\n\t** nVal:\n")
      if(debug>=2) qvcalc::indentPrint(nVal, indent=8) # if(debug>=2) cat("\n")
      if(debug>=2){
        cat("\n\t\tdVal & length(dVal) for this rep & par set:")
        qvcalc::indentPrint(length(dVal))
        qvcalc::indentPrint(dVal)
        # nVal <- c(nVal,logexVal,markVal)
      }
      if(debug>=2){
      # if(debug>=3 & config$survSave=="all"){
        cat("\n\t\tdsrMat & its dimensions for this rep & par set:")
        qvcalc::indentPrint(dim(dsrMat[,r,i]))
        qvcalc::indentPrint(dsrMat[,r,i])
        cat("\n\tdVal:")
        qvcalc::indentPrint(dVal)
      }
    }

    dsrMat[,r,i] <- dVal
    nValMat[,r,i] <- nVal

  #---- Finish replicate: -----------------------------------------
    if(config$testing=="yes"){
      valMat[,r,i] <- make_summary(dVal, nVal, truePSRcovar=vals$psrTrueVal,debug=debug)
    }

    repID = repID + 1
  }

#---- Finish param set: -----------------------------------------

  if(config$testing=="yes"){
    if (debug>=2){
      intTab <- table(obsIntList)
      cat("\t\t\t| all intervals: ")
      cat(sprintf("[%s]",paste(names(intTab),as.numeric(intTab),sep=": ")))
    }
  }

  ## save to file every 50 param sets
  if (as.numeric(parID) %% 50 == 0){
    parStart = parID - 49 # parStart = as.numeric(parID) - 4
    fname <- sprintf("%s/nval_%sto%s.rds", outdir, parID-49,parID)
    saveMat <- nValMat[,,c(parStart:parID)] # print("incremental save:") print(fname)
    # cat(sprintf("\n** incremental save from param set %s to %s (%s)", parStart,parID, fname))
    saveRDS(saveMat, fname) # print(saveMat)
    fname2 <- sprintf("%s/dsrval_%sto%s.rds", outdir, parID-49,parID)
    saveDSR <- dsrMat[,,c(parStart:parID)]
    saveRDS(saveDSR, fname2) # print(saveMat)
  }

  parID = parID + 1L
  rngSeed <- rngSeed + 1L ## add integer so not coerced to float

  endTimePar <- Sys.time()
  runTimePar <- format(as.POSIXct(as.numeric(endTimePar - startTimePar, units="secs"), 
                                               origin="1970-01-01", tz="UTC"),"%Hh %Mm %Ss")
  cat(sprintf("\n|>|> PAR SET %s RUN TIME: %s", i,runTimePar))
}
cat(sprintf("\n\nOUTPUT DIRECTORY: %s", odir))

dsrvalname <- sprintf("%s/dsrval%s%s.rds", odir,config$rngSeed,atype)
saveRDS(dsrMat, dsrvalname)

vcovname <- sprintf("%s/vcov%s%s.rds", odir,config$rngSeed,atype)
if(config$coefSave=="all") saveRDS(vcovMatMat, vcovname)

trueDSR_name <- sprintf("%s/trueDSR%s%s.rds", odir,config$rngSeed,atype)
if(config$coefSave=="all") saveRDS(trueDSRmat, trueDSR_name)

propinit_name <- sprintf("%s/initprop%s%s.rds", odir,config$rngSeed,atype)
if(config$coefSave=="all") saveRDS(initPropMat, propinit_name)

nvalname <- sprintf("%s/nval%s%s.rds", odir,config$rngSeed,atype)
if(config$ndatSave) saveRDS(nValMat, nvalname)

pDF   <- data.table::rbindlist(py_to_r(pArrList)) # print(pDF)
colnm <- names(py_to_r(pArrList))
print(colnm)
if(atype %in% c("range","control", "test2") & vary!="stormFrq") colnm <- c(colnm, vary)
parname <- sprintf("%s/par%s%s.rds", odir,config$rngSeed,atype)
saveRDS(pDF, parname)

if(config$coefSave=="all"){
  coefname <- sprintf("%s/coef%s%s.rds", odir,config$rngSeed,atype)
  saveRDS(coefsMat, coefname)
}

if(config$coefSave=="all"){
  coefname <- sprintf("%s/coef%s%s.rds", odir,config$rngSeed,atype)
  saveRDS(coefsMat, coefname)
}

if(config$testing=="yes"){
  if (saveNestData){
    ndataaname <- sprintf("%s/nest_data_%s%s_100reps.rds", odir,config$rngSeed,atype)
    saveRDS(nestDataMat, ndataaname)
  }
  cat(sprintf("\n|> STATIC PARAMS: prob surv=%s; decay rate=%s; disc prob=%s\n", par$probSurv, par$decayRate, par$discProb))
  cat("\nvary:", vary)
  cat("\nparam lists:\n")
  qvcalc::indentPrint(pDF)

  if (debug>=4) cat("\nCoefficients & standard error:\n")
  if (debug>=4 & config$coefSave=="all") qvcalc::indentPrint(coefsMat, indent=8)

  if (debug>=1) cat("\nDSR Val:\n")
  if (debug>=1) qvcalc::indentPrint(dsrMat, indent=8)

  if (debug>=1) cat("\nN Val:\n")
  if (debug>=1) qvcalc::indentPrint(nValMat, indent=8)

  if(debug>=4)  cat("\nvalMat:\n")
  if(debug>=4)  qvcalc::indentPrint(valMat)

  if(debug>=4)  cat("\nstorm dates:\n", stormDates)
  cat(sprintf("\tmax obs length: %s", max(obsLength)))
  cat(sprintf("\tmax interval: %s",max(obsIntList)))

  if(atype %in% c("test100","test500")) sdate_name <- sprintf("%s/storm_dates_%s.txt",odir,atype)
  if(atype %in% c("test100","test500")) write(stormDates,sdate_name)

  ## print summary regardless of debug settings (so even if atype=='range')
  summ <- apply(valMat,c(1,3),mean,na.rm=TRUE) ## pass args to mean after function itself
  cat("\nSummary (mean for each param set):\n")
  qvcalc::indentPrint(summ, indent=8)

  valMatList <- asplit(valMat, 3)

  if(atype %in% c("range" ,"nstest1" , "nstest2" , "testctrl", "snrange")){
    cat("\nPLOTS >> varying the levels of", vary, "\n")
    # db=TRUE
    db=FALSE
    print(prBoxPl("diff_mayf", valMatList, deb=db))
    print(prBoxPl("diff_lexp", valMatList, deb=db))
    print(prBoxPl("diff_mcmc", valMatList, deb=db))
    if(config$mcmcOld) print(prBoxPl("diff_mcmc_old", valMatList))
  }

}
endTime <- Sys.time()
runTime <- format(as.POSIXct(as.numeric(endTime - startTime, units="secs"), 
                             origin="1970-01-01", tz="UTC"),"%Hh %Mm %Ss")
cat(sprintf("\n\n|>|> TOTAL RUN TIME: %s\n", runTime))

