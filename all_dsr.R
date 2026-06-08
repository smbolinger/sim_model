
## attempting to streamline
## takes parts of logexp.R and datsim.py 
#
# startTime <- Sys.time()
# # arg <- unlist(strsplit(commandArgs(trailingOnly=TRUE), split=" "))
# arg <- unlist(strsplit(commandArgs(trailingOnly=FALSE), split=" "))
# print(arg)
# # file_arg <- grep("(?<=^--file=)[A-Za-z]*\,[A-Za-z]", arg, value = TRUE)
# # file_arg <- grep("(?<=file=)(\\w+\\.\\w+)", arg, perl=TRUE, value = TRUE)
# file_arg <- stringr::str_extract(arg, "(?<=file=)(\\w+\\.\\w+)")
# file_arg <- file_arg[!is.na(file_arg)]
# print(file_arg)
# print( class(file_arg))
# library(reticulate)
# # Sys.setenv(script_name="all_dsr.R")
# Sys.setenv(script_name=file_arg)
# py_run_file("init.py")
# library(MASS)
# library(brglm2)
# library(tidyr)
# suppressPackageStartupMessages(library(dplyr)) # load dplyr last so as not to mask select?
#
# # library(jsonlite)
# #NOTE could make a counter of all times at least one survey int == 0
# psrTrue = "date"

#---- LOAD FUNCTIONS & VARIABLES --------------------------------------------------------------
source("lexp_setup.R")
source("lexp_fun.R")
Sys.setenv(r_outdir=py_to_r(outdir))
# cat(sprintf("\n\tin R: outdir=%s & type=%s", py_to_r(outdir),class(py_to_r(outdir))))
# if(config$mark) library(RMark)
options(width=1000, digits=5, scipen=999)

#---- LOOP THRU PARAM SETS ----------------------------------------------------------------------
parID = 0
for(i in seq(length(pArrList))){

#---- Make params, storms, surveys: --------------------------------------------------------------
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  # {funs$mk_param_list(paramsArray[i-1], staticPar,debug=TRUE)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  # print(par) # if(debug) print(par$stormFrq) print(class(par))
  print(par)
  print(class(par))
  cat("\n\n...............................i=",i, ".........................................................................................................................................................\n")
  print(unlist(py_vars(par)))
  cat(".............................................................................................................................................................................................\n")
  
#---- LOOP THRU REPLICATES ------------------------------------------------------------------------------
  repID=0
  for(r in seq(nreps)){
    if(debug>=1) cat(sprintf("\n:::::::::::::::::::::::::::::: rep %s",i))
    cat(sprintf("-%s",r))
    if(debug>=1) cat(":::::::::::::::::::::::::::::::::::::::::::\n")

    if(TRUE){
      # if(debug>=2) cat("\n\t>> overwriting storm days -")
      stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=sett$stormFromFile,db=TRUE)
      if(debug>=2) cat("\n\t>> storm days = ",stormDays)
      survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,db=debug)},
      # survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,complicate=FALSE,db=debug)},
                                       error=function(e){ 
                                         reticulate::py_last_error() 
                                         # print(sys.calls()) # doesn't help if error in python
                                       }  )
    }

  #---- Full nest data: ----------------------------------------------------

    skiptoNext <- FALSE ## whether or not to skip to next replicate

    ## create the nest & observation data:
    nestData1 <- withCallingHandlers({
      nweeks = round(par$brDays/7)-2 # nweeks = floor(par$brDays/7)
      obs$make_obs(par,rng,obsVarNum,stormDays,survey,pyconfig,initDat,nweeks,sett$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }

    ## calculate some values from the nest data:
    nVal <- mod$calc_nests(nestData1, par, rng, obsVarNum, repID, parID,pyconfig,db=config$debugNests)
    names(nVal) <- nval_name

    ## filter and rename the nest data:
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)

    #-~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(config$testing=="yes"){
      if(debug>=3) cat("\n\t** nVal:\n")
      if(debug>=3) qvcalc::indentPrint(nVal, indent=8)
      # if(debug>=2) cat("\n")
      if(debug>=3) cat("\n\t[*] [*] [*] [*] [*]  NEST DATA [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      if(debug>=4) cat("\n\t\t** all nest data:\n") # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
      if(debug>=4 & debug<6) qvcalc::indentPrint(head(nestData1,30), indent=8) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)
      if(debug>=6) qvcalc::indentPrint(nestData1, indent=8) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)

      all_fld      <- sum(nestData1$fate==2, na.rm=TRUE)
      all_hatch    <- sum(nestData1$fate==0, na.rm=TRUE)
      all_longfin  <- sum(nestData1$fint>par$obsFreq, na.rm=TRUE)
      all_hatch_longfin <- sum(nestData1$fint>par$obsFreq, na.rm=TRUE)
      all_misclass <- sum(nestData1$fate!=nestData1$afate,na.rm=TRUE)
      all_unk      <- sum(nestData1$afate==7, na.rm=TRUE)
      apparent_all <- dsr$calc_dsr(nData=nestData1,nestType="all", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
      write(stormDays, file="out/storm_plot.txt", sep="\t", append=TRUE, ncolumns = 10)
      write(nestData1$init, file="out/init_plot.txt", sep="\t", append=TRUE, ncolumns = par$numNests)
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #---- Nest data - discovered: ----------------------------------------------------
    nestData <- nestData1 |> filter(.data[[obsVar]]>0) # remove undiscovered nests

    #-~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(TRUE){

      # num_disc <- nrow(nestData)
      # num_excl      <- sum(nestData$afate==7, na.rm=TRUE)
      # num_misclass  <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
      # num_misclass  <- num_misclass-num_excl
      # num_an        <- num_disc - num_excl
      #
      # mayfield_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="mayfield",
      #                               conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
      #
      # apparent_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="apparent",
      #                               conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
      #
      if(config$testing=="yes"){
        num_disc <- nrow(nestData)
        num_excl      <- sum(nestData$afate==7, na.rm=TRUE)
        num_misclass  <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
        num_misclass  <- num_misclass-num_excl
        num_an        <- num_disc - num_excl

        mayfield_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="mayfield",
                                      conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

        apparent_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="apparent",
                                      conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

        prop_excl     <- num_excl/num_disc
        prop_misclass <- num_misclass/num_disc
        disc_fld      <- sum(nestData$fate==2, na.rm=TRUE)
        disc_hatch    <- sum(nestData$fate==0, na.rm=TRUE)
        disc_longfin  <- sum(nestData$fint>par$obsFreq, na.rm=TRUE)
        if(debug>=3) cat(sprintf("\n\t\t>>> discovered nests (length=%s):\n", num_disc))
        if(debug>=3 & debug<6) qvcalc::indentPrint(head(nestData,25), indent=12)
        if(debug>=6) qvcalc::indentPrint(nestData)
        if(debug>=2) cat(sprintf("\t\t\tfor discovered nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_disc, apparent_disc))
      }
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Nest data - analyzed: ----------------------------------------------------
    nestData <- nestData |> filter(afate!=7) 
    nestData <- nestData |> filter(end>i) ## remove nests found on end date (should be none?)

    #-~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if(TRUE){
      # if(debug>=2) qvcalc::indentPrint(colSums(is.na(nestData)))
    #   mayfield_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="mayfield",
    #                                 # conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
    #                                 conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debug)
    #
    #   apparent_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="apparent",
    #                                 conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
    # }
    #
    if(config$testing=="yes"){
      mayfield_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="mayfield",
                                    # conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debug)

      apparent_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      an_fld <- sum(nestData$fate==2, na.rm=TRUE)
      an_hatch <- sum(nestData$fate==0, na.rm=TRUE)
      an_misclass <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
      an_longfin <- sum(nestData$fint>par$obsFreq, na.rm=TRUE)

      if(debug>=5){
        cat("\n\t\t>> NAs after excluding unknown fate nests:\n")
        print(colSums(is.na(nestData))) ## this is actually indented somewhat
        cat("\n")
      }

      if(debug>=3) cat(sprintf("\n\t\t>>> analyzed nests (length=%s ; num excluded=%s):\n",nrow(nestData), num_excl))
      if(debug>=3 & debug<6) qvcalc::indentPrint(head(nestData,25), indent=12)
      if(debug>=6) qvcalc::indentPrint(nestData, indent=8)
      if(debug>=3) cat(sprintf("\t\t\tfor analyzed nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_an, apparent_an))

      if(debug>=2){
        # allVal <- c("all_fld","all_hatch","all_longfin","all_unk","all_misclass")
        # allVal <- c(all_fld,all_hatch,all_longfin,all_misclass,all_unk)
        allVal <- c(all_fld,all_hatch,all_longfin,all_misclass,num_disc)
        allProp <- allVal/par$numNests
        discVal <- c(disc_fld,disc_hatch,disc_longfin,num_misclass,num_excl)
        discProp <- discVal/num_disc
        anVal <- c(an_fld,an_hatch,an_longfin,an_misclass)
        anProp <- anVal/num_an
        # cat(sprintf("\n\tall: flooded=%s, hatched=%s, long final=%s, unknown=%s, miclassified=%s ",allVal))
        cat(do.call(sprintf,c("\n\tall: flooded=%s, hatched=%s, long final=%s, miclassified=%s, discovered=%s ",as.list(allVal))))
        cat(do.call(sprintf,c("\t\t\t\t| prop flooded=%s, prop hatched=%s, prop long final=%s, prop miclassified=%s, prop discovered=%s ",as.list(allProp))))
        cat(do.call(sprintf,c("\n\tdiscovered: flooded=%s, hatched=%s, long final=%s, miclassified=%s, excluded=%s ",as.list(discVal))))
        cat(do.call(sprintf,c("\t | prop flooded=%.3f, prop hatched=%.3f, prop long final=%.3f, prop miclassified=%.3f, prop excluded=%.3f ",as.list(discProp))))
        cat(do.call(sprintf,c("\n\tanalyzed: flooded=%s, hatched=%s, long final=%s, misclassified=%s",as.list(anVal))))
        cat(do.call(sprintf,c("\t\t\t\t\t\t\t\t\t | prop flooded=%.3f, prop hatched=%.3f, prop long final=%.3f, prop miclassified=%.3f ",as.list(anProp))))
      }
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Calculate true DSR: ----------------------------------------------------
      # maxSurveyDay  <- max(nestData1$init) + par$hatchTime
      ## predict for all initiation dates, then scale by num nests initiated on each day:
      prDays  <- seq(1, max(nestData1$init)) # if(debug>=4) cat("\n\tdates for prediction: ", prDays, length(prDays))

      ## 1. basic calculation:
      simplePSR <- nVal["hat"] / par$numNests
      
      ## 2. create inputs for calculating true DSR using logistic exposure:
      ##    > needs to be all nests and all days (not just observed)
      modData <- mk_logex_data( nestData1, survey=survey, pyconfig=pyconfig, exposure=1 ) 
      mList_true <- c("Surv~1", "Surv~Date")                      ## models to fit
      newDat <- data.frame(Date=prDays)                           ## data for prediction

      ## 3. predict from glm - logit:

    if(TRUE){

      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] True DSR - logit [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)

      ret <- mk_true_dsr(nestData1, mList_true, newDat, par, config)
      dsrT = ret[1]
      psrT = ret[2]
      psrT_date = ret[3]

      #-~~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        if(debug>=3) cat(sprintf("\n\t|> apparent PSR: %s [num hatch]/%s [num total] = %s):", nVal["hat"],par$numNests,simplePSR))
        if(debug>=2) cat("\t|> true PSR, logit <date>:", ret[3])
        # write(psrOut[[2]], file="out/psr_plot.txt", sep="\t", append=TRUE, ncolumns=130)
        # psrPlot_true <- 
        if(debug>=3) cat("\t|> true DSR & PSR, logit <null>:", ret[c(1,2)])
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    } 

  #---- Mayfield: -----------------------------------------------------------------------------------------------
    if(TRUE){
      exposureNorm <- sum(nestData$j - nestData$i)
      exposureFinal <- sum(nestData$k - nestData$j) * 0.5
      exposure1 <- exposureNorm + exposureFinal
      numFail  <- sum(nestData$afate!=0)
      mayfDSR <- 1 - (numFail/exposure1)
      if(debug>=3) cat("\n")
      if(debug>=2) cat(sprintf("\t>> calculate Mayfield DSR: 1 - (%s[numFail]/%s[exposure]) = %s", numFail, exposure1,mayfDSR))
    }

  #---- Logistic exposure: -----------------------------------------------------------------------------------------------
    if(config$logex){
      dat2S <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, exposure=0) 

      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
        if(config$debugLogEx>=4) {
          cat("\n\n\t\t>>> dat2S:\n")
          qvcalc::indentPrint(head(dat2S, 30), indent=8)
        }
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      coefsArray <- calc_logexp(mList,dat2S,config=config)
      nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
      # # nestObs <- nestData |> dplyr::select(ID, init, i, j, k, afate, totobs) # print(head(nestObs))
      nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) # print(head(nestObs))
      numObs <- nestData[,"totobs"]
      if(any(coefsArray=="exception")){
        cat("  go to next ~~")
      #     # coefs[,r,i] <- coefsArray
        next
      }
      if(config$coefSave!="none") coefs[,r,i] = unlist(coefsArray)
    }

  #---- Logexp DSR & PSR: -----------------------------------------------------------------------------------------------
    if(config$logex){
      dsr1        <- 1/(1+exp(-coefsArray[[1]][1,1]))
      psr1        <- dsr1 ^ par$hatchTime
      ## covariate = average date 
      dsr2        <- 1/(1+exp(-coefsArray[[6]][1,1] + coefsArray[[6]][1,2] * dat2S$avDate))
      psr2        <- dsr2 ^ par$hatchTime
      psr2        <- psr2[1]
      # print(psr2)
      ## covariate = average date age of nest
      dsr3        <- 1/(1+exp(-coefsArray[[6]][1,1] + coefsArray[[6]][1,2] * dat2S$avAge))
      psr3        <- dsr3 ^ par$hatchTime
      psr3        <- psr3[1] ## shoud all be the same value

      dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
      dsrList <- dsrList[-1]
      allInits    <- nestData$init

      psr <- sapply(dsrList, function(x){
                       make_psr(x, allInits, dat2S$Date, par, config)
                                    })

      #-~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        # if(config$debugLogEx>=3) cat("\n\t\t>> psr (avg psrList weighted by nest initiation per day), excluding intercept-only model: ", unlist(psr), "\n")
        if(config$debugLogEx>=3) cat("\n\t\t>> logistic exposure PSR (avg weighted by inits per day), excl null model: ",class(psr), unlist(psr), "\n")
        # if(config$debugLogEx>=3) cat("\n\t\t>> psr22 (output of make_psr function), excluding intercept-only model: ", unlist(psr22), "\n")
        # if(config$debugLogEx>=3) qvcalc::indentPrint(psr)
        if(debug>=3){
          cat("\n\t|> logistic exposure DSR & PSR <null>:", dsr1, psr1)
          # cat("\n|> logistic exposure DSR & PSR (average date):", dsr2,psr2)
          # cat("\n\t|> logistic exposure PSR (mods 2-5):", paste(psr,collapse=" ; "))
          cat("\n\t|> logistic exposure PSR (mods 2-5):", paste(psr,collapse=" ; "))
          # cat("\n\t|> logistic exposure PSR (av date; av age):", dsr2,psr2,dsr3,psr3)
        }
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      logexVal <- unlist(tibble::lst(dsr1,psr1,psr))
      # logexVal <- unlist(tibble::lst(dsr1,psr1,psr[2:6]))
      # logexVal        <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]]) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
      # names(logexVal) <- lexp_name
    } else {
      logexVal <- c() }


  #---- MCMC model: ------------------------------------------------------------------------------------------------------

    if(config$mcmc){
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] MCMC model [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      lVal = withCallingHandlers({
        mod$rep_loop(par, rng, nestData, stormDays, survey, pyconfig) # to_r=TRUE
      },
      error=function(e){
        skiptoNext <<- TRUE # need to use super-assignment
        message("error in MCMC model: ", e) # print(sys.calls())
        reticulate::py_last_error()
      })
      llVal <- py_to_r(lVal$astype("float64")) # when it's a np ndarray, this dosn't work
      mcmcDSR <- py_to_r(llVal[[1]]) # list does convert, & needs to be 1-indexed?
      mcmcPSR <- llVal[[2]]
      mcmcDFR <- llVal[[3]]
      # mcmc1 <- c(mcmcDSR,mcmcPSR,mcmcDFR)
      mcmcVal <-  unlist(tibble::lst(mcmcDSR,mcmcPSR,mcmcDFR))

      #-~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        if(debug>=3) cat("\n")
        # if(debug>=3) cat("\n\t\t|>all MCMC model output:\n") # print(class(lVal))
        # if(debug>=3) qvcalc::indentPrint(lVal)

        # if(debug>=3) qvcalc::indentPrint(class(llVal))
        # if(debug>=3) qvcalc::indentPrint(llVal)
        # if(debug>=5){
        #   qvcalc::indentPrint(class(llVal))
        #   qvcalc::indentPrint(llVal)
        #   qvcalc::indentPrint(class(llDSR))
        #   qvcalc::indentPrint(llDSR)
        # }
        # if(debug>=3) cat("\n\t|> MCMC DSR & PSR = ", llDSR,llPSR)
        if(debug>=2) cat("\t|> MCMC DSR & PSR = ", mcmcVal)
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  }

  #---- Add to DSR matrix: -----------------------------------------------------------------------------------------------
    mayfVal = unlist(tibble::lst(mayfDSR))
    # if(FALSE){ ## if calculate true DSR is false (above) then this should be true
    #   dsrT = apparent_all
    #   psrT = dsrT ^ par$hatchTime
    #   if(config$mark) dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal,markVal)
    # }
    dsrTrue = unlist(tibble::lst(dsrT,psrT,psrT_date))
    # dVal <- c(dsrTrue,logexVal,mcmcVal,mayfDSR,mayfDSR_an)
    dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal)
    
    #-~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(config$testing=="yes"){
      if(debug>=4){
        cat("\n\t\tdVal & length(dVal) for this rep & par set:")
        qvcalc::indentPrint(length(dVal))
        qvcalc::indentPrint(dVal)
        # nVal <- c(nVal,logexVal,markVal)
      }
      if(debug>=5){
        cat("\n\t\tdsrMat & its dimensions for this rep & par set:")
        qvcalc::indentPrint(dim(dsrMat))
        qvcalc::indentPrint(dsrMat[,r,i])
        # cat("\ndVal:")
        # print(dVal)
      }
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dsrMat[,r,i] <- dVal
    nValMat[,r,i] <- nVal


  #---- Finish replicate: -----------------------------------------
    if(config$testing=="yes"){
      if(debug>=5) message("\n\n\t>> saving vals to matrix for summary")
      aDSR = nVal["aDSR"]
      aPSR = aDSR ^ par$hatchTime
      # dsrT = nVal["aDSR"]
      if(psrTrue=="date") {
        psrT = psrT_date
        if(config$debugDSR>=4) cat("\n\t\tpsrT = w/date covar:", psrT)
      }
      # mayfDSR = nVal["mfDSR"]
      mayfPSR = mayfDSR ^ par$hatchTime
      if(config$logex) {leDSR = dsr1 } else {leDSR = 0}
      if(config$logex) {lePSR = psr1 } else {lePSR = 0}

      if(debug>=2) cat(sprintf(
                               "\n\n\t<> <> PSR vals: true= %.5f, MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ",
                               psrT, mcmcPSR,  lePSR, mayfPSR)
      )

      if(debug>=2) cat(sprintf(
                               "\n\t<> <> <> <> <> <> <> <> diff from true: MCMC=%.5f, logEx=%.5f, Mayfield=%.5f \n",
                               mcmcPSR-psrT, lePSR-psrT, mayfPSR-psrT)
      )
      vals         <- c(psrT,psrT_date,aPSR,lePSR,mcmcPSR,mayfPSR)
      diffs        <- abs(c(psrT_date-psrT,aPSR-psrT,lePSR-psrT,mcmcPSR-psrT,mayfPSR-psrT))
      valMat[,r,i] <- c(par$stormFrq,par$pMortFl,par$obsFreq,par$discProb,par$decayRate,par$probSurv,
                        all_hatch,all_fld,num_disc,num_excl,prop_excl,prop_misclass,vals,diffs)
      # if(debug>=5)  cat("\nstore summary vals:\n")
      # if(debug>=5)  qvcalc::indentPrint(valMat, indent=8)
      # if(debug>=2) cat(sprintf("\nsummary of rep %s:", r))
      # if(debug>=2) cat(sprintf(" num nest=%s | obs int=%s | assigned DSR=%s", par$numNests,par$obsFreq,par$probSurv))
      # if(debug>=2)
    }

    repID = repID + 1
  }

#---- Finish param set: -----------------------------------------
  # if (as.numeric(parID) %% 5 == 0){
  if (as.numeric(parID) %% 50 == 0){
    parStart = parID - 49 # parStart = as.numeric(parID) - 4
    fname <- sprintf("%s/nval_%sto%s.rds", outdir, parID-49,parID)
    saveMat <- nValMat[,,c(parStart:parID)] # print("incremental save:") print(fname)
    # cat(sprintf("\n** incremental save from param set %s to %s (%s)", parStart,parID, fname))
    saveRDS(saveMat, fname) # print(saveMat)
  }
  parID = parID + 1
}

# dsrvalname <- sprintf("%s/dsrval.rds", outdir)
# cat("\n\nOUTPUT DIRECTORY:", odir)
cat(sprintf("\n\nOUTPUT DIRECTORY: %s", odir))
# print(odir)
dsrvalname <- sprintf("%s/dsrval%s%s.rds", odir,config$rngSeed,atype)
saveRDS(dsrMat, dsrvalname)

# nvalname <- sprintf("%s/nval.rds", outdir)
nvalname <- sprintf("%s/nval%s%s.rds", odir,config$rngSeed,atype)
saveRDS(nValMat, nvalname)

endTime <- Sys.time()
runTime <- format(as.POSIXct(as.numeric(endTime - startTime, units="secs"), 
                             origin="1970-01-01", tz="UTC"),"%Hh %Mm %Ss")
                             # origin="1970-01-01", tz="UTC"),"%H:%M:%S")

# runMin <- runTime/60
# runHour <- runTime/3600

# runTime <- case_when(runTime>60 ~ runTime/60, runTime>3600,runTime/3600)

cat(sprintf("\n\n|>|> TOTAL RUN TIME: %s\n", runTime))

if(config$testing=="yes"){
  # cat(sprintf("\n|> PARAMS: num nests=%s; prob surv=%s; decay rate=%s; disc prob=%s\n", par$numNests,par$probSurv, par$decayRate, par$discProb))
  cat(sprintf("\n|> STATIC PARAMS: prob surv=%s; decay rate=%s; disc prob=%s\n", par$probSurv, par$decayRate, par$discProb))
  cat("\nparam lists:\n")
  # if(debug>=3) qvcalc::indentPrint(lapply(py_to_r(pArrList), unlist))
  # plist <- lapply(py_to_r(pArrList), unlist)
  # plist <- py_to_r(pArrList)
  pDF   <- data.table::rbindlist(py_to_r(pArrList))
  # print(pDF)
  # pDF   <- pDF[,c(1,2,7:14)]
  colnm <- c("stormFate","numNests", "pMortFl", "MCtype", "propMC", "propUnk", "stormDur", "stormFrq", "obsFreq", "hatchTime")
  # if(atype %in% c("range","control","nostorm", "nstest1", "nstest2")) columns <- c(columns, vary)
  if(atype %in% c("range","control") & vary!="stormFrq") colnm <- c(colnm, vary)
  # if(atype %in% c("range","control")) colnm <- c(colnm, vary)
  # colnm <- colnm[colnm!=""]
  names(pDF) <- colnm
  qvcalc::indentPrint(pDF)

  if (debug>=4) cat("\nCoefficients:\n")
  if (debug>=4) qvcalc::indentPrint(coefs, indent=8)

  if (debug>=1) cat("\nDSR Val:\n")
  if (debug>=1) qvcalc::indentPrint(dsrMat, indent=8)

  if (debug>=1) cat("\nN Val:\n")
  if (debug>=1) qvcalc::indentPrint(nValMat, indent=8)

  if(debug>=4)  cat("\nvalMat:\n")
  if(debug>=4)  qvcalc::indentPrint(valMat)

  ## print summary regardless of debug settings (so even if atype=='range')
  summ <- apply(valMat,c(1,3),mean,na.rm=TRUE) ## pass args to mean after function itself
  cat("\nSummary (mean for each param set):\n")
  qvcalc::indentPrint(summ, indent=8)

  valMatList <- asplit(valMat, 3)
  # if(debug>=1)  cat(sprintf("\nvalMatList <%s>:\n", class(valMatList)))
  # if(debug>=1)  qvcalc::indentPrint(valMatList)
  # print(class(valMatList))

  prBoxPl <- function(val1, valMatList, deb=FALSE){
    # print(deb)
    # val1 <- "diff_mcmc"
    # cat(sprintf("\n>> difference from true DSR for %s by param %s:", val1, param))
    valStr <- stringr::str_extract(val1, "(?<=_)\\w+")
    # print(valStr)
    # cat(sprintf("\n>> difference between true DSR and %s DSR for each param set: \n\n", valStr))
    cat(sprintf("\n>> difference between true PSR and %s PSR for each param set: \n\n", valStr))
    # cat(sprintf("\n>> difference between apparent DSR and %s DSR for each param set: \n\n", valStr))
    boxPlList  <- lapply(valMatList, function(x) {
    # boxPlList  <- lapply(seq_along(valMatList), function(x) {
                           # print(x["diff_mcmc",])
                           # print(class(x["diff_mcmc",])) x["diff_mcmc",]
                           # x[val1,]
                           unname(x[val1,])
                           # mat = valMatList[[x]]
                           # print(mat)
                           # unname(mat[val1,]
                           # vname = paste0("parSet", x)
                           # assign(vname,unname(mat[val1,]))
          })
    names(boxPlList) <- paste0("parSet", seq(boxPlList))
    # if(deb) print(class(boxPlList))
    if(deb) cat(sprintf("\nboxPlList <%s>:\n", class(boxPlList)))
    if (deb) print(boxPlList) # txtplot::txtboxplot(boxPlList)
    # list2env(boxPlList, envir=environment()) ## unpack to local function evironmnet
    # if (deb) print(ls()) ## and then print the local environment

    # e <- list2env(boxPlList)
    # newList <- as.list(e)
    # if(deb) print(newList)
    # oldWidth <- getOption("width")
    # options(width=100)
    # do.call(txtplot::txtboxplot,boxPlList)

    ## show the function call itself:
    if (deb) cat("\nfunction call for txtboxplot:\n")
    callList <- list(as.name("txtplot::txtboxplot"), c(boxPlList, list(width=70)))
    # callList <- list(as.name("txtplot::txtboxplot"), c(newList, list(width=70)))
    if (deb) print(as.call(callList))

    ## need to pass the list vals as individual arguments:
    do.call(txtplot::txtboxplot, c(boxPlList, list(width=70, legend=FALSE)))
    # do.call(txtplot::txtboxplot, c(newList, list(width=70)))
    # do.call(callList)
    # txtplot::txtboxplot(boxPlList, width=70) ## can't just pass a list
    # options(width=oldWidth)
  }

  if(atype=="range" | atype=="nstest1" | atype=="nstest2" | atype=="test2"){
    cat("\nPLOTS >> varying the levels of", vary, "\n")
    # db=TRUE
    db=FALSE
    print(prBoxPl("diff_mayf", valMatList, deb=db))
    print(prBoxPl("diff_lexp", valMatList, deb=db))
    print(prBoxPl("diff_mcmc", valMatList, deb=db))
    if(config$mcmcOld) print(prBoxPl("diff_mcmc_old", valMatList))
  }

}

# boxpl <- apply(valMat, c(1,3), function(x){ 
                 # txtboxplot(data=as.data.frame(x, stringsAsFactors=T)) 
      # })
# print(boxpl)

# print(valMatList)
# print(dim(valMatList))
# valVecList <- asplit(valMat, 3)

# valMatList <- apply(asplit(valMat, 3), 1, asplit)
# prBoxPl <- function(val1, valMatList, param="", deb=FALSE){
  # pars <- c(par$numNests, par$stormFrq, par$pMortFl, par$hatchTime, par$obsFreq, par$decayRate, par$discProb)
  # print all except the one that varied for atrange
  # cat(sprintf("num nests=%s; num storms=%s; intensity=%s; hatch time=%s; obs int=%s; decay rate=%s; disc prob=%s",
  # par$numNests, par$stormFrq, par$pMortFl, par$hatchTime, par$obsFreq, par$decayRate, par$discProb))
  # cat(sprintf("num nests=%s; prob surv=%s; decay rate=%s; disc prob=%s", par$numNests,par$probSurv, par$decayRate, par$discProb))
