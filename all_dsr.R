
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
# if (debug>0) source("lexp_fun.R") else source("nodebug_lexp_fun.R")
source("lexp_fun.R")
Sys.setenv(r_outdir=py_to_r(outdir))
# cat(sprintf("\n\tin R: outdir=%s & type=%s", py_to_r(outdir),class(py_to_r(outdir))))
# if(config$mark) library(RMark)
options(width=1000, digits=5, scipen=999)

#---- LOOP THRU PARAM SETS ----------------------------------------------------------------------
# parID = 0
parID = startParID
# for(i in seq(length(pArrList))){
for(i in seq(startParID,length(pArrList))){
  # obsLength <- c()
  # obsIntList <- c()

#---- Make params, storms, surveys: --------------------------------------------------------------
  startTimePar <- Sys.time()
  rng <- np$random$default_rng(seed=rngSeed)
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  # {funs$mk_param_list(paramsArray[i-1], staticPar,debug=TRUE)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  cat(sprintf("\n\n.....%s......i=%03d.......seed=%d...........................................................................................................\n",format(startTimePar, "%H:%M:%S"),i,rngSeed))
  print(unlist(py_vars(par)))
  cat(".....................................................................................................................................................\n")

  # if(TRUE){
  #   # if(debug>=2) cat("\n\t>> overwriting storm days -")
  #   stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=sett$stormFromFile,db=TRUE)
  #   if(debug>=2) cat("\n\t>> storm days = ",stormDays)
  #   survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,db=debug)},
  #   # survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,complicate=FALSE,db=debug)},
  #                                    error=function(e){ 
  #                                      reticulate::py_last_error() 
  #                                      # print(sys.calls()) # doesn't help if error in python
  #                                    }  )
  # }

  
#---- LOOP THRU REPLICATES ------------------------------------------------------------------------------
  repID=0
  for(r in seq(nreps)){
    # if(r==6 & i==3){ debug <- 5 config$testing <- "yes" config$debug <- 5 config$debugObs <- 5 config$debugNests <- 5 config$debugLogEx <- 5 config$debugDSR <- 5 }
    if(debug>=1) cat(sprintf("\n:::::::::::::::::::::::::::::: rep %s",i))
    cat(sprintf("-%s",r))
    if(debug>=1) cat(":::::::::::::::::::::::::::::::::::::::::::\n")

    if(TRUE){
      # if(debug>=2) cat("\n\t>> overwriting storm days -")
      stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=sett$stormFromFile,db=TRUE)
      if(debug>=2) cat("\n\t>> storm days = ",stormDays)
      if(config$testing=="yes") stormDates <- c(stormDates, stormDays)
      survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,db=debug)},
      # survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, pyconfig,rng,complicate=FALSE,db=debug)},
                                       error=function(e){ 
                                         reticulate::py_last_error() 
                                         # print(sys.calls()) # doesn't help if error in python
                                       }  )
    }
    skiptoNext <- FALSE ## whether or not to skip to next replicate

  #---- Full nest data: ----------------------------------------------------

    ## create the nest & observation data:
    nestData1 <- withCallingHandlers({
      nweeks = round(par$brDays/7)-2 # nweeks = floor(par$brDays/7)
      obs$make_obs(par,rng,obsVarNum,stormDays,survey,pyconfig,initDat,stormUnk,nweeks,sett$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      save_current(startParID, i, odir)
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }
    if(debug>=4) cat("\n\t\t** all nest data:\n") # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
    if(debug>=4 & debug<6) qvcalc::indentPrint(head(nestData1,30), indent=8) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)
    # if(debug>=4) print(table(nestData1[,4]))
    # print(survey)

    ## calculate some values from the nest data:
    nVal <- mod$calc_nests(nestData1, par, rng,survey, obsVarNum, repID, parID,pyconfig,db=config$debugNests)
    # nVal <- obs$calc_nests(nestData1, par, rng,survey, obsVarNum, repID, parID,pyconfig,db=config$debugNests)
    names(nVal) <- nval_name

    ## filter and rename the nest data:
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)
    if(debug>=4) print(table(nestData1$fate))
    if(debug>=4) print(table(nestData1$init))

    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if(debug>0){
    #   if(debug>=3) cat("\n\t** nVal:\n")
    #   if(debug>=3) qvcalc::indentPrint(nVal, indent=8) # if(debug>=2) cat("\n")
    # }
    if(config$testing=="yes"){
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
    #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #---- Nest data - discovered: ----------------------------------------------------
    nestData <- nestData1 |> filter(.data[[obsVar]]>0) # remove undiscovered nests

    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Nest data - analyzed: ----------------------------------------------------
    nestData <- nestData |> filter(afate!=7) 
    nestData <- nestData |> filter(end>i) ## remove nests found on end date (should be none?)

    #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Calculate true DSR: ----------------------------------------------------
      ## predict for all initiation dates, then scale by num nests initiated on each day:
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] True DSR - make nest data [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      prDays_true  <- seq(1, max(nestData1$init)) # if(debug>=4) cat("\n\tdates for prediction: ", prDays, length(prDays))

      ## 1. basic calculation:
      ## all hatched / all nests
      simplePSR <- nVal["hat"] / par$numNests
      # simplePSR <- (1-nVal[""] / par$numNests
      
      ## 2. create inputs for calculating true DSR:
      ##    > needs to be all nests and all days (not just observed)
      # modData <- mk_logex_data( nestData1, survey=survey, pyconfig=pyconfig, expoVal=1 ) 
      mList_true <- c("Surv~1", "Surv~Date")                      ## models to fit
      newDat_true <- data.frame(Date=prDays_true)                           ## data for prediction
      newDat <- data.frame(Date=prDays)

      ## 3. predict from glm - logit:
      # trueDSRlist <- mk_true_dsr(nestData1, mList_true, newDat_true, par, config)
      trueDSRlist <- mk_true_dsr(nestData1, mList_true, newDat, par, config)
      # numFill <- preDays - length(trueDSRlist[[2]])
      # numFill <- preDays - max(nestData1$init)
      # cat("\nnumFill & max init date=",numFill,class(numFill),max(nestData1$init),class(max(nestData1$init)),"\n")
      # cat("\nnumFill & length of dsr list=",numFill,class(numFill),length(trueDSRlist[[2]]),class(length(trueDSRlist[[2]])),"\n")
      # trueDSR_date <- c(trueDSRlist[[2]], rep(-1,numFill)) ## date covar 
      # print(trueDSR_date)
      # trueDSRmat[,r,i] <- trueDSR_date
      # print(propInitScl)

      ## save the date-specific DSR vals to matrix & mayb to file?
      # if(config$coefSave=="yes"){
      #   numInit     <- sapply(prDays, function(x) sum(nestData1$init==x))
      #   propInit    <- numInit/par$numNests
      #   propInitScl <- propInit/sum(propInit) ## make sure it sums to 1
      #   trueDSRmat[1,,r,i] <- trueDSRlist[[2]]
      #   trueDSRmat[2,,r,i] <-  propInitScl    
      #
      #   # propInitmat[,r,i] <- propInitScl     
      # }

      # truePSRlist <- lapply(trueDSRlist, function(x) x ^ par$hatchTime)
      # ret2 <- make_psr_list(nestData1,survey,mList_true,par,pyconfig,expo=1)
      # print(ret2)
      dsrT = trueDSRlist[[1]][1]
      psrT = dsrT ^ par$hatchTime
      # psrT_date = make_psr(trueDSRlist[[2]],nestData1$init,prDays_true,par,config)
      out_true = make_weighted(trueDSRlist[[2]],list(),nestData1$init,prDays,par,config)
      dsrT_date = out_true[[1]]
      psrT_date = out_true[[2]]

      # dsrT = ret[1]
      # psrT = ret[2]
      # psrT_date = ret[3]
      # dsrT2 = ret2[[1]]
      # psrT2 = ret2[[2]]
      # psrT_date2 = ret2[[3]]

      # rm(trueDSR_date)
      rm(trueDSRlist)

      #-*~~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        # if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] True DSR - logit [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)

        if(debug>=3) cat(sprintf("\n\t|> apparent PSR: %s [num hatch]/%s [num total] = %s):", nVal["hat"],par$numNests,simplePSR))
        # if(debug>=2) cat("\t|> true PSR, logit <date>:", ret[3])
        if(debug>=2) cat("\t|> true DSR, logit <date>:", dsrT_date)
        if(debug>=2) cat("\t|> true PSR, logit <date>:", psrT_date)
        # if(debug>=2) cat("\t|> true PSR, logex <date>:", psrT2)
        # write(psrOut[[2]], file="out/psr_plot.txt", sep="\t", append=TRUE, ncolumns=130)
        # psrPlot_true <- 
        # if(debug>=3) cat("\t|> true DSR & PSR, logit <null>:", ret[c(1,2)])
        if(debug>=3) cat("\t|> true DSR & PSR, logit <null>:",dsrT,psrT)
        # if(debug>=3) cat("\t|> true DSR & PSR, logexp <null>:", psrT2, psrT_date2)
        if(config$coefSave=="yes"){
          if(debug>=3) cat("\nfill in matrix:\n" )
          if(debug>=3) print(trueDSRmat[,,r,i])
        }
      }
      #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #---- Logistic exposure: -----------------------------------------------------------------------------------------------

    if(config$logex){
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      dat2S <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, expoVal=0) 
      n_obs <- nrow(dat2S)
      if (debug>=2) cat("\n\t\t\tnumber of obs=", n_obs)
      # print(str(dat2S))
      if(config$testing=="yes") obsLength <- c(obsLength, nrow(dat2S))
      if(config$testing=="yes") obsIntList <- c(obsIntList, max(dat2S$Exposure))
      modFit <- calc_logexp(mList, dat2S, config=config)
      if(any(modFit=="exception")){
        cat("all_dsr.R: exception -  go to next ~~")
        save_current(startParID, i, odir)
        skiptoNext <<- TRUE # need to use super-assignment
          # coefs[,r,i] <- coefsArray
        # next
      }
    }
    if(skiptoNext) { next }
    if(config$logex){
      # coefDF  <- sapply(modFit, function(x) coef(summary(x)[,c("Estimate", "Std. Error")]))
      # print(coefDF) # coefList[[]]

      ## DATA FOR PREDICTIONS:
      ##NOTE: use all vals for date in all of them bc that's the var I want to plot
      newDat <- expand.grid(Date=prDays,
                             Age=mean(dat2S$Age,na.rm=TRUE),
                             avDate=mean(dat2S$avDate,na.rm=TRUE)) # byVal = 4

      # coeff  <- unlist(sapply(modFit, function(x) coef(summary(x))[,"Estimate"]))
      # coeffSE  <- unlist(sapply(modFit, function(x) coef(summary(x))[,"Std. Error"]))
      coeff  <- sapply(modFit, function(x) coef(summary(x))[,"Estimate"])
      coeffSE  <- unlist(sapply(modFit, function(x) coef(summary(x))[,"Std. Error"]))
      vcOut <- lapply(modFit, function(x) vcov(x)) # get variance-covariance matrix
      # ..exposure <- mean(dat2S$Exposure)
      # predNew <- lapply(modFit, function(x) predict(x, newdata=newDat,type="response",se.fit=TRUE))
      # predNew <- lapply(modFit, function(x) predict(x,type="response",se.fit=TRUE))
      # rm(..exposure)
      # if (debug >=3)  cat("\npredNew:")
      # if (debug >=3)  print(predNew)
      # seNew <- lapply(modFit, function(x) sqrt(diag(summary(x)$cov.unscaled)*summary(x)$dispersion))
      # if (debug >=3)  cat("\nseNew:")
      # if (debug >=3)  print(seNew)
      if(debug>=5){
        cat("\n\tcoefs & se:\n")
        # qvcalc::indentPrint(unlist(coeff))
        qvcalc::indentPrint(coeff)
        qvcalc::indentPrint(unlist(coeffSE))
        cat("\n\t& vcov matrices:\n")
        qvcalc::indentPrint(vcOut)
      }
      # print(class(coeff))
      # coefsMat[,r,i] <- c(coeff, coeffSE)
      # coefsMat[,r,i] <- c(coeff, coeffSE,mean(dat2S$Exposure))
      # print(newDat$AvDate[1])
      # print(mean(dat2S$AvDate))
      # print(mean(dat2S$AvDate, na.rm=T))
      coefsMat[,r,i] <- c(unlist(coeff),
                          unlist(coeffSE),
                          mean(dat2S$Exposure,na.rm=TRUE),
                          mean(dat2S$Age,na.rm=TRUE),
                          # mean(dat2S$AvDate,na.rm=TRUE)) ## no idea why this doesn't work - oh, bc I'm dumb
                          mean(dat2S$avDate,na.rm=TRUE)) ## no idea why this doesn't work - oh, bc I'm dumb
                          # newDat$AvDate[1])
      # qvcalc::indentPrint(coefsMat[,r,i])

      DSR_out <- lapply(seq_along(modFit), function(x){
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
      # print(DSRlist)
      # print(seList)
      # print(class(seList))


      # DSRout <- make_dsr_list(dat2S,prDat=newList,survey,mList,par,pyconfig,newList=TRUE, out=modFit)
      # cat("\nold DSR list function")
      # DSRout <- make_dsr_list(dat2S,prDat=dat2S,survey,mList,par,pyconfig,newList=FALSE, out=modFit)
      # # cat("\t>> extracting DSRlist and seList")
      # DSRlist <- lapply(DSRout, '[[', 1)
      # seList <- lapply(DSRout, '[[', 2)
      # # print(seList)
      # # seList <- DSRout[[2]]
      # if(any(DSRlist=="exception")){
      #   cat("all_dsr.R: exception - go to next ~~")
      #   save_current(startParID, i, odir)
      # #     # coefs[,r,i] <- coefsArray
      #   next
      # }
      # out_list <- sapply(seq(2,length(DSRlist)), function(x){
      out_list <- sapply(seq(length(DSRlist)), function(x){
      # out_list <- sapply(DSRlist, function(x){
                   # make_psr(x, nestData$init, dat2S$Date, par, config)
                    # newDat = newList[[x]]
                   make_weighted(DSRlist[[x]], seList[[x]],
                                 # nestData$init, dat2S$Date,
                                 nestData$init, newDat$Date,
                                 par, config)
                                })
      if(debug>=5) print(out_list)
      if(debug>=5) print(class(out_list))

      # dsr_list <- out_list[1,]
      # psr_list <- out_list[2,]
      # se_list <- out_list[3,]
      dsr_list <- unlist(out_list[1,])
      psr_list <- unlist(out_list[2,])
      se_list <- unlist(out_list[3,])

      # logexVal <- unlist(tibble::lst(dsr1,psr1,se1,dsr_list,psr_list,se_list))
      logexVal <- unlist(tibble::lst(dsr_list,psr_list,se_list))
      #-*~~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        # print(class(dsr_list))
        dsr1 <- dsr_list[1]
        # print(class(dsr1))
        psr1 <- psr_list[1]
        se1 <- se_list[1]
        if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] logistic exposure - output [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
        if(config$debug>=3){
          # cat("\n\t\t>> all_dsr.R: MAKING PSR LISTS\n")
          # cat("\n\t\t>> TEST DSR list 1: ",class(DSRlist), unlist(DSRlist), "\n")
          # cat("\n\t\t>> TEST DSR list 2: ",class(DSRlist2), unlist(DSRlist2), "\n")
          # cat("\npredList:\n")
          # print(predList)
          # dsr1 <- out_list[1,1]
          # psr1 <- out_list[2,1]
          # se1 <- out_list[3,1]
          cat("\n\t\t>> logistic exposure DSR (avg weighted by inits per day): ",class(dsr_list), unlist(dsr_list), "\n")
          cat("\n\t\t>> logistic exposure PSR (avg weighted by inits per day): ",class(psr_list), unlist(psr_list), "\n")
          cat("\n\t|> logistic exposure DSR & PSR <null> & s.e.:", dsr1, psr1, se1,"\n")
        }
        if(config$coefSave=="yes"){
          if(debug>=3) cat("\nfill in matrix:\n" )
          if(debug>=3) print(trueDSRmat[,,r,i])
        }
      }
      #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    } else {
      logexVal <- c() }

  #---- MCMC model: ------------------------------------------------------------------------------------------------------

    # if(TRUE){
    if(config$mcmc){
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] MCMC model [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      # cat("\n\t\t>> all_dsr: trying out matlab function")
      # cat("\nusing nestData & converting in matlab_func.py")
      # cat("\t>> all_dsr: make input data")
      # print(str(dat2S))
      # obsInp <- mlfun$mk_obs_mat(obsSel, survey,config)
      # cat("\t\t>> all_dsr: run the optimization")
      # cat("\nMCMC output:")
      # print(mlOut)
      # print(class(mlOut))
      ## can't send dat2S directly, because fates aren't coded correctly for the matrix model
      # obsSel <- dat2S |> select('Nest.ID','Surv','Exposure')
      # obsSel <- nestData |> select('ID','init','end','fate','i','j','k','afate','nobs')
      obsSel <- nestData |> select('ID','init','end','fate','i','j','k','afate','totobs')
      jacPlSuf <- sprintf("%s-%s",i,r)
      withCallingHandlers({
        # mlOut <- mlfun$PolyMort(obsSel,survey,par, rng,pyconfig,suff=jacPlSuf,db=config$debugLL)
        # if (par$stormFrq==0 | par$stormDur==0)
        if (atype=="control")
          mlOut <- mlfun$PolyMort(obsSel,survey,par,pyconfig,suff=jacPlSuf,db=config$debugLL,scl_fac=0.15)
        else
          mlOut <- mlfun$PolyMort(obsSel,survey,par,pyconfig,suff=jacPlSuf,db=config$debugLL)
      },
      error=function(e){
        message("\nerror in MCMC model: ", e, "; try higher outLen") # print(sys.calls())
        reticulate::py_last_error()
        withCallingHandlers({
          # mlOut <- mlfun$PolyMort(obsSel,survey,par, rng,pyconfig,plt=TRUE,db=3,suff=jacPlSuf)
          # mlOut <- mlfun$PolyMort(obsSel,survey,par,pyconfig,plt=TRUE,db=3,suff=jacPlSuf)
          # if (par$stormFrq==0 | par$stormDur==0)
          if (atype=="control")
            mlOut <- mlfun$PolyMort(obsSel,survey,par,pyconfig,suff=jacPlSuf,db=config$debugLL,scl_fac=0.15)
          else
            mlOut <- mlfun$PolyMort(obsSel,survey,par,pyconfig,suff=jacPlSuf,db=config$debugLL)
        },
        error=function(e){
          skiptoNext <<- TRUE # need to use super-assignment
          message("\nerror in MCMC model: ", e, "; go to next") # print(sys.calls())
          save_current(startParID, i, odir)
          reticulate::py_last_error()
        })
      })
      # if(skiptoNext) next
      # print(lVal)
      # llVal <- py_to_r(lVal$astype("float64")) # when it's a np ndarray, this dosn't work

      # mcmcVal <-  unlist(tibble::lst(mlOut[c(1,2,4,5)]))
    }
    if(skiptoNext) { next }
    if(config$mcmc){
      mcmcDSR = mlOut[[1]][1]
      mcmcDSR_se = mlOut[[3]][1]
      mcmcPSR = mcmcDSR ^ par$hatchTime
      mcmcDPR = mlOut[[2]][1]
      mcmcDPR_se = mlOut[[3]][1]
      # mcmcVal <-  unlist(tibble::lst())
      mcmcVal <- c(mcmcDSR, mcmcPSR, mcmcDPR, mcmcDSR_se, mcmcDPR_se)
      # cat("\n\t\t>> all_dsr: output from matlab function:", mcmcVal)
    } else {
      mcmcVal = rep(-999, 5)
    }
    # if(config$mcmc){
      #-*~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] MCMC model - output [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
        if(debug>=3) cat("\n")
        if(debug>=2) cat("\t|> MCMC DSR & PSR = ", mcmcVal)
      }
      #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # }

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
    # mayfVal = unlist(tibble::lst(mayfDSR))
    # mayfVal = unlist(tibble::lst(mayfDSR,mayfPSR,mayfVar,simplePSR))
    mayfVal = unlist(tibble::lst(mayfDSR,mayfPSR,mayfVar,mayfSE,simplePSR))
    dsrTrue = unlist(tibble::lst(dsrT,psrT,dsrT_date,psrT_date))
    # dsrTrue = unlist(tibble::lst(dsrT,psrT,psrT_date,dsrT2,psrT2,psrT_date2))
    # dVal <- c(dsrTrue,logexVal,mcmcVal,mayfDSR,mayfDSR_an)
    dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal)
    
    #-*~~~~debug~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(debug>0){
      if(debug>=2) cat("\n\t** nVal:\n")
      if(debug>=2) qvcalc::indentPrint(nVal, indent=8) # if(debug>=2) cat("\n")
      if(config$debug>=3){
        # cat("\n\t\t>> all_dsr.R: MAKING PSR LISTS\n")
        cat("\n\t>> logistic exposure DSR (avg weighted by inits per day): ",class(dsr_list), unlist(dsr_list), "\n")
        cat("\n\t>> logistic exposure PSR (avg weighted by inits per day): ",class(psr_list), unlist(psr_list), "\n")
      }
      if(debug>=3) cat("\n")
      if(debug>=2) cat("\t|> MCMC DSR, DFR, & PSR (&se) = ", mcmcVal)
      if(debug>=2) cat("\n\t\t|> logistic exposure DSR & PSR <null>:", dsr1, psr1)
      if(debug>=3) cat("\n")
      if(debug>=2) cat(sprintf("\t>> calculate Mayfield DSR: 1 - (%s[numFail]/%s[exposure]) = %s", numFail, exposure1,mayfDSR))
      if(debug>=3) cat(sprintf("\t>> Mayfield standard error: %s", mayfSE))
    }
    if(config$testing=="yes"){
      if(debug>=4){
        cat("\n\t\tdVal & length(dVal) for this rep & par set:")
        qvcalc::indentPrint(length(dVal))
        qvcalc::indentPrint(dVal)
        # nVal <- c(nVal,logexVal,markVal)
      }
      if(debug>=3){
        cat("\n\t\tdsrMat & its dimensions for this rep & par set:")
        qvcalc::indentPrint(dim(dsrMat[,r,i]))
        qvcalc::indentPrint(dsrMat[,r,i])
        cat("\n\tdVal:")
        qvcalc::indentPrint(dVal)
      }
    }
    #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
      } else {
        psrT = aPSR
        if(config$debugDSR>=4) cat("\n\t\tpsrT (aPSR):", psrT)
      }
      # mayfDSR = nVal["mfDSR"]
      mayfPSR = mayfDSR ^ par$hatchTime
      if(config$logex) {leDSR = dsr1 } else {leDSR = 0}
      if(config$logex) {lePSR = psr1 } else {lePSR = 0}
      # leDSR_2 <- predList[[1]]$fit[1]
      # lePSR_2 <- leDSR_2 ^ par$hatchTime
      mcmcPSR = dsrMat['mcmcPSR',r,i]


      if(debug>=2) cat(sprintf(
                               "\n\n\t<> <> PSR vals: true= %.5f, MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ",
                               psrT, mcmcPSR,  lePSR, mayfPSR)
      )

      if(debug>=2) cat(sprintf(
                               "\n\t<> <> <> <> <> <> <> <> diff from true: MCMC=%.5f, logEx=%.5f, Mayfield=%.5f \n",
                               mcmcPSR-psrT, lePSR-psrT, mayfPSR-psrT)
      )
      vals         <- c(psrT,psrT_date,aPSR,lePSR,mcmcPSR,mayfPSR)
      # vals         <- c(psrT,psrT_date,aPSR,lePSR,lePSR_2,mcmcPSR,mayfPSR)
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
  
  endTimePar <- Sys.time()
  # save_current(startParID, i, odir)
  if(config$testing=="yes"){
    runTimePar <- format(as.POSIXct(as.numeric(endTimePar - startTimePar, units="secs"), 
                                                 origin="1970-01-01", tz="UTC"),"%Hh %Mm %Ss")
    cat(sprintf("\n|>|> PAR SET %s RUN TIME: %s", i,runTimePar))
    # cat(sprintf("\tmax obs length (%s nests): %s", par$numNests, max(obsLength)))
    # cat(sprintf("\tmax interval (obsInt=%s): %s", par$obsFreq,max(obsIntList)))
    # cat(sprintf("\tall intervals: %s", paste(obsIntList,collapse=" ")))
    intTab <- table(obsIntList)
    cat("\tall intervals: ")
    cat(sprintf("[%s]",paste(names(intTab),as.numeric(intTab),sep=": ")))
    # cat(paste(paste0("[",names(intTab)),as.numeric(intTab),sep=": "))
    # cat(sprintf("\tall intervals: "))
    # qvcalc::indentPrint(table(obsIntList))
    # save_current(startParID, i, odir)
  }

  if (as.numeric(parID) %% 50 == 0){
    parStart = parID - 49 # parStart = as.numeric(parID) - 4
    fname <- sprintf("%s/nval_%sto%s.rds", outdir, parID-49,parID)
    saveMat <- nValMat[,,c(parStart:parID)] # print("incremental save:") print(fname)
    # cat(sprintf("\n** incremental save from param set %s to %s (%s)", parStart,parID, fname))
    saveRDS(saveMat, fname) # print(saveMat)
  }
  parID = parID + 1L
  rngSeed <- rngSeed + 1L ## add integer so not coerced to float
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

pDF   <- data.table::rbindlist(py_to_r(pArrList)) # print(pDF)
# pDF   <- pDF[,c(1,2,7:14)]
# colnm <- c("stormFate","numNests", "pMortFl", "MCtype", "propMC", "propUnk", "stormDur", "stormFrq", "obsFreq", "hatchTime")
colnm <- names(py_to_r(pArrList))
print(colnm)
if(atype %in% c("range","control", "test2") & vary!="stormFrq") colnm <- c(colnm, vary)
# names(pDF) <- colnm
parname <- sprintf("%s/par%s%s.rds", odir,config$rngSeed,atype)
saveRDS(pDF, parname)

coefname <- sprintf("%s/coef%s%s.rds", odir,config$rngSeed,atype)
saveRDS(coefsMat, coefname)

                             # origin="1970-01-01", tz="UTC"),"%H:%M:%S")
# runMin <- runTime/60 runHour <- runTime/3600 runTime <- case_when(runTime>60 ~ runTime/60, runTime>3600,runTime/3600)

endTime <- Sys.time()
runTime <- format(as.POSIXct(as.numeric(endTime - startTime, units="secs"), 
                             origin="1970-01-01", tz="UTC"),"%Hh %Mm %Ss")
cat(sprintf("\n\n|>|> TOTAL RUN TIME: %s\n", runTime))



if(config$testing=="yes"){
  # cat(sprintf("\n|> PARAMS: num nests=%s; prob surv=%s; decay rate=%s; disc prob=%s\n", par$numNests,par$probSurv, par$decayRate, par$discProb))
  cat(sprintf("\n|> STATIC PARAMS: prob surv=%s; decay rate=%s; disc prob=%s\n", par$probSurv, par$decayRate, par$discProb))
  cat("\nvary:", vary)
  cat("\nparam lists:\n")
  # if(debug>=3) qvcalc::indentPrint(lapply(py_to_r(pArrList), unlist))
  # plist <- lapply(py_to_r(pArrList), unlist)
  # plist <- py_to_r(pArrList)
  # if(atype %in% c("range","control")) colnm <- c(colnm, vary)
  # colnm <- colnm[colnm!=""]
  # print(colnm)
  qvcalc::indentPrint(pDF)

  if (debug>=4) cat("\nCoefficients & standard error:\n")
  # if (debug>=4) qvcalc::indentPrint(coefs, indent=8)
  if (debug>=4) qvcalc::indentPrint(coefsMat, indent=8)

  if (debug>=1) cat("\nDSR Val:\n")
  if (debug>=1) qvcalc::indentPrint(dsrMat, indent=8)

  if (debug>=1) cat("\nN Val:\n")
  if (debug>=1) qvcalc::indentPrint(nValMat, indent=8)

  if(debug>=4)  cat("\nvalMat:\n")
  if(debug>=4)  qvcalc::indentPrint(valMat)

  if(debug>=4)  cat("\nstorm dates:\n", stormDates)
  cat(sprintf("\tmax obs length: %s", max(obsLength)))
  cat(sprintf("\tmax interval: %s",max(obsIntList)))
# nvalname <- sprintf("%s/nval%s%s.rds", odir,config$rngSeed,atype)
  # if(atype %in% c("test100","test500")) ol_name <- sprintf("%s/obs_length_%s.txt",odir,atype)
  # if(atype %in% c("test100","test500")) write(obsLength,ol_name) ## have one saved in 202060701
  if(atype %in% c("test100","test500")) sdate_name <- sprintf("%s/storm_dates_%s.txt",odir,atype)
  if(atype %in% c("test100","test500")) write(stormDates,sdate_name)

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

  # if(atype=="range" | atype=="nstest1" | atype=="nstest2" | atype=="test2"){
  # if(atype %in% c("range" ,"nstest1" , "nstest2" , "test2", "snrange")){
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
