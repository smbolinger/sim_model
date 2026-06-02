
## attempting to streamline
## takes parts of logexp.R and datsim.py 

startTime <- Sys.time()
library(reticulate)
Sys.setenv(script_name="logexp.R")
py_run_file("init.py")
library(MASS)
suppressPackageStartupMessages(library(dplyr)) # load dplyr last so as not to mask select?
library(brglm2)
library(dplyr)
library(tidyr)
# library(jsonlite)
#NOTE could make a counter of all times at least one survey int == 0
psrTrue = "date"

#---- LOAD FUNCTIONS & VARIABLES --------------------------------------------------------------
source("lexp_fun.R")
source("lexp_setup.R")
Sys.setenv(r_outdir=py_to_r(outdir))
cat(sprintf("\n\tin R: outdir=%s & type=%s", py_to_r(outdir),class(py_to_r(outdir))))
# if(config$mark) library(RMark)
options(width=1000, digits=5, scipen=999)

#---- LOOP THRU PARAM SETS ----------------------------------------------------------------------
parID = 0
for(i in seq(length(pArrList))){

#---- Make params, storms, surveys: --------------------------------------------------------------
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  # print(par) # if(debug) print(par$stormFrq) print(class(par))
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
      stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=sett$stormFromFile,db=debug)
      if(debug>=2) cat("\n\t>> storm days = ",stormDays)
      survey    <- withCallingHandlers({obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=pyconfig,db=debug)},
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
      obs$make_obs(par,rng,stormDays,survey,pyconfig,initDat,nweeks,sett$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }

    ## calculate some values from the nest data:
    nVal <- mod$calc_nests(nestData1, par, rng, repID, parID,pyconfig,db=config$debugNests)
    names(nVal) <- nval_name
    ## filter and rename the nest data:
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if(TRUE){
    if(config$testing=="yes"){
      if(debug>=2) cat("\n\t** nVal:\n")
      if(debug>=2) qvcalc::indentPrint(nVal, indent=8)
      # if(debug>=2) cat("\n")
      if(debug>=3) cat("\n\t[*] [*] [*] [*] [*]  NEST DATA [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      if(debug>=4) cat("\n\t\t** all nest data:\n") # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
      if(debug>=4 & debug<6) qvcalc::indentPrint(head(nestData1,30), indent=8) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)
      if(debug>=6) qvcalc::indentPrint(nestData1, indent=8) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)

      all_fld      <- sum(nestData1$fate==2, na.rm=TRUE)
      all_hatch    <- sum(nestData1$fate==0, na.rm=TRUE)
      all_longfin  <- sum(nestData1$fint>par$obsFreq, na.rm=TRUE)
      all_misclass <- sum(nestData1$fate!=nestData1$afate,na.rm=TRUE)
      all_unk      <- sum(nestData1$afate==7, na.rm=TRUE)
      apparent_all <- dsr$calc_dsr(nData=nestData1,nestType="all", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)
      write(stormDays, file="out/storm_plot.txt", sep="\t", append=TRUE, ncolumns = 10)
      write(nestData1$init, file="out/init_plot.txt", sep="\t", append=TRUE, ncolumns = par$numNests)
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #---- Nest data - discovered: ----------------------------------------------------
    # disc     <- sum(nestData1$totobs < 1, na.rm=TRUE)
    nestData <- nestData1 |> filter(totobs>0) # remove undiscovered nests
    num_disc <- nrow(nestData)
    num_excl      <- sum(nestData$afate==7, na.rm=TRUE)
    num_misclass  <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
    num_misclass  <- num_misclass-num_excl

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(TRUE){
      disc_fld      <- sum(nestData$fate==2, na.rm=TRUE)
      disc_hatch    <- sum(nestData$fate==0, na.rm=TRUE)
      disc_longfin  <- sum(nestData$fint>par$obsFreq, na.rm=TRUE)
      num_an        <- num_disc - num_excl
      prop_excl     <- num_excl/num_disc
      prop_misclass <- num_misclass/num_disc

      mayfield_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="mayfield",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      apparent_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      if(debug>=3) cat(sprintf("\n\t\t>>> discovered nests (length=%s):\n", num_disc))
      if(debug>=3 & debug<6) qvcalc::indentPrint(head(nestData,25), indent=12)
      if(debug>=6) qvcalc::indentPrint(nestData)
      if(debug>=2) cat(sprintf("\t\t\tfor discovered nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_disc, apparent_disc))
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Nest data - analyzed: ----------------------------------------------------
    # nestData <- nestData |> filter(afate!=7) |> na.omit() # remove unknown fate nests
    nestData <- nestData |> filter(afate!=7) 
    # print(nrow(nestData))
    nestData <- nestData |> filter(end>i) ## remove nests found on end date (should be none?)
    # print(nrow(nestData))

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(TRUE){
      if(debug>=2){
        cat("\n\t\t>> NAs after excluding unknown fate nests:\n")
        print(colSums(is.na(nestData))) ## this is actually indented somewhat
        cat("\n")
      }

      # if(debug>=2) qvcalc::indentPrint(colSums(is.na(nestData)))
      an_fld <- sum(nestData$fate==2, na.rm=TRUE)
      an_hatch <- sum(nestData$fate==0, na.rm=TRUE)
      an_misclass <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
      an_longfin <- sum(nestData$fint>par$obsFreq, na.rm=TRUE)

      mayfield_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="mayfield",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      apparent_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="apparent",
                                    conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

      if(debug>=3) cat(sprintf("\n\t\t>>> analyzed nests (length=%s ; num excluded=%s):\n",nrow(nestData), num_excl))
      if(debug>=3 & debug<6) qvcalc::indentPrint(head(nestData,25), indent=12)
      if(debug>=6) qvcalc::indentPrint(nestData, indent=8)
      if(debug>=2) cat(sprintf("\t\t\tfor analyzed nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_an, apparent_an))

      if(debug>=3){
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
    if(TRUE){
      prDays  <- seq(1, max(nestData1$init))
      # if(debug>=4) cat("\n\tdates for prediction: ", prDays, length(prDays))

      ## 1. basic calculation:
      simplePSR <- nVal["hat"] / par$numNests
      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] True DSR - logistic exposure [*] [*] [*] [*] [*] [*] [*] \n\n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      
      ## 2. create inputs for calculating true DSR using logistic exposure:
      ##    > needs to be all nests and all days (not just observed)
      modData <- mk_logex_data( nestData1, survey=survey, pyconfig=pyconfig, exposure=1 ) 
      mList_true <- c("Surv~1", "Surv~Date")                      ## models to fit
      # mList_true <- c("Survival~1", "Survival~Date")                      ## doesn't make sense?
      coef_out <- calc_logexp(mList_true, modData, config=config) ## get coefficients
      newDat <- data.frame(Date=prDays)                           ## data for prediction
      # psrList_true <- mk_true_dsr(nestData1, "status ~ Date", newDat, par, config)
      # psr_true <- mk_true_dsr(nestData1, "status ~ Date", newDat, par, config)
      # ret <- mk_true_dsr(nestData1, "status ~ Date", newDat, par, config)

      ## 3. predict from glm:
      # cat("\nmake predictions - old")
      dsrTrueList <- make_pred(coef_out, nmod=2, mods=mList_true, newDat=newDat,hTime=par$hatchTime, db=config$debugLogEx)
      if(debug>=7) print(dsrTrueList)
      dsrTrueList <- dsrTrueList[[2]]

      # dsrT <-  1/(1+exp(-coef_out[[2]][1,1] + coef_out[[2]][1,2] * modData$avAge))
      allInits    <- nestData1$init 
      numInit     <- sapply(newDat$Date, function(x) sum(allInits==x))
      propInit    <- numInit/par$numNests
      propInitScl <- propInit/sum(propInit)
      # psr <- lapply(psrList, function(x) sum(x*propInitScl))
      # dsrT <- sum(propInitScl * dsrList[[2]])
      # if(debug>=3) cat("\n\tdsrT [date]:", dsrT)
      # psrT <- dsrT ^ par$hatchTime
      # if(debug>=3) cat("\n\tpsrT [date]:", psrT)
      # psrT <- dsrList[[2]] ^ par$hatchTime

      if(debug>=5) cat(sprintf("\n\t\tdsrTrueList (length=%s):\n\t\t", length(dsrTrueList)), unlist(dsrTrueList))
      psrTrueList <- lapply(dsrTrueList, function(x) x ^ par$hatchTime)
      if(debug>=5) cat(sprintf("\n\t\tpsrTrueList (length=%s):\n\t\t",length(psrTrueList)), unlist(psrTrueList))
      # psrTrueList <- lapply(psrTrueList, function(x) sum(propInitScl * x))
      ## psrTrueList is no longer a list of 2 lists
      psrT_date <- sum(propInitScl * unlist(psrTrueList))
      if(debug>=5) cat(sprintf("\n\t\tpropInitScl (length=%s):\n\t\t", length(propInitScl)), unlist(propInitScl))
      if(debug>=5) cat(sprintf("\n\t\tpsrTrueList, scaled (length=%s):\n\t\t",length(psrTrueList)), unlist(psrTrueList)) # dsrT <-  1/(1+exp(-coef_out[[1]][1,1]))
      dsrT <-  1/(1+exp(-coef_out[[1]][1,1]))
      psrT <- dsrT ^ par$hatchTime
      # if(debug>=3) cat("\n\t\ttrue DSR & PSR, null <logexp>:", c(dsrT, psrT))
      # psrT_date <- psrTrueList[[2]]

      if(debug>=3) cat("\n\n\t[*] [*] [*] [*] [*] True DSR - logit [*] [*] [*] [*] [*] [*] [*] \n\n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)

      ret <- mk_true_dsr(nestData1, mList_true, newDat, par, config)
      if(debug>=5) cat("\n\t\t\t>> return values from binomial DSR function:",ret)
      # if(debug>=3) cat("\n\t\ttrue PSR, date <logexp> ( psrTrueList[[2]] ):", psrT_date)
      # dsrT = dsrList[[1]]
      # if(debug>=3) cat("\n\tpsrT, scaled:", unlist(psrT))
      # dsrTnocov <- sum(propInit * dsrList[[1]])
      # psrTnocov <- dsrTnocov ^ par$hatchTime
      # dsrTrue <- c(dsrT,psrT)

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(config$testing=="yes"){
        # psrPlot_true <- 
        if(debug>=4) cat("\n\t\t\t>-> coef_out:\n")
        if(debug>=4) qvcalc::indentPrint(coef_out, indent=8)
        if(debug>=2) cat(sprintf("\n\t|>|> basic calculation of PSR: %s [num hatched] / %s [num total] = %s):", nVal["hat"],par$numNests,simplePSR))
        if(debug>=2) cat("\n\t|> true DSR & PSR, logistic exposure (no covars):", dsrT, psrT)
        if(debug>=2) cat("\n\t|> true PSR, logistic exposure (date covar):", psrT_date)
        if(debug>=3) cat("\n\t|> true DSR & PSR, logit <no covars>:", ret[c(1,2)])
        if(debug>=3) cat("\n\t|> true PSR, logit <date covar>:", ret[3])
        if(debug>=5){
          cat("\n\t\t\t>-> true DSR vals:\n")
          qvcalc::indentPrint(dsrTrueList[[2]], indent=8)
          cat("\n\t\t\t\t>-> inits & dates:\n")
          qvcalc::indentPrint(allInits, indent=8)
          qvcalc::indentPrint(newDat$Date, indent=8)
          # cat("\nnum inits before date:\n")
          cat("\n\t\t\t\t>-> num inits on date:\n")
          qvcalc::indentPrint(numInit, indent=8)
          cat(sprintf("\n\t\t\t\t>-> proportion inits on date (sum=%s):\n", sum(propInit)))
          qvcalc::indentPrint(propInit, indent=8)
        }
      }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    } 

  #---- Logistic exposure: -----------------------------------------------------------------------------------------------
    if(config$logex){
      # if(debug>=3) cat("\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      if(debug>=2) cat("\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n\n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      dat2S <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, exposure=0) 
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(debug>=2) {
        cat("\n\t\t>>> dat2S:\n")
        qvcalc::indentPrint(head(dat2S, 30), indent=8)
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      coefsArray <- calc_logexp(mList,dat2S,config=config)
      # if(debug>=2) cat("\n<*><*> Logistic exposure <*><*><*><*>\n")
      # excpt <- FALSE
      nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
      # # nestObs <- nestData |> dplyr::select(ID, init, i, j, k, afate, totobs) # print(head(nestObs))
      nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) # print(head(nestObs))
      numObs <- nestData[,"totobs"]
      # dat2S = modData
      if(any(coefsArray=="exception")){
        cat("  go to next ~~")
      #     # coefs[,r,i] <- coefsArray
        next
      }
      if(config$coefSave!="none") coefs[,r,i] = unlist(coefsArray)
      # coefsArray = modOut
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

      allInits    <- nestData$init
      numInit     <- sapply(dat2S$Date, function(x) sum(allInits==x))
      propInit    <- numInit/par$numNests
      propInitScl <- propInit/sum(propInit)

      dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
      dsrList <- dsrList[-1]
      ## NOTE make_pred already starts at model number 2, so why exclude 1st output??
      ## NOTE because it is empty bc index starts at 2 for list as well
      psrList <- lapply(dsrList, function(x) x^par$hatchTime)
      psr <- lapply(psrList, function(x) sum(x*propInitScl))

      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(TRUE){
        if(debug>=4){
        #   # cat("\nlength of dsr2:\n", length(dsr2))
          cat("\n\t\t\t\t>-> inits & dates:\n")
          qvcalc::indentPrint(allInits, indent=8)
          qvcalc::indentPrint(dat2S$Date, indent=8)
          # cat("\nnum inits before date:\n")
          cat("\n\t\t\t\t>-> num inits on date:\n")
          qvcalc::indentPrint(numInit, indent=8)
          cat("\n\t\t\t\t>-> proportion inits on date:\n")
          qvcalc::indentPrint(propInit, indent=8)
          qvcalc::indentPrint(propInitScl, indent=8)
        }
        if(config$debugLogEx>=4){
          cat(sprintf("\n\t\t\t|> output of make_pred (dsrList:%s & psrList:%s):\n", length(dsrList), length(psrList)))
          qvcalc::indentPrint(dsrList, indent=8)
          qvcalc::indentPrint(psrList, indent=8)
        }
        if(config$debugLogEx>=1) cat("\n\t\t>> psr (avg psrList weighted by nest initiation per day), excluding intercept-only model: ", unlist(psr), "\n")
        # if(config$debugLogEx>=3) qvcalc::indentPrint(psr)
        if(debug>=3){
          cat("\n\t|> logistic exposure DSR & PSR (no covars):", dsr1, psr1)
          # cat("\n|> logistic exposure DSR & PSR (average date):", dsr2,psr2)
          cat("\n\t|> logistic exposure PSR (mods 2-5):", paste(psr,collapse=" ; "))
          # cat("\n\t|> logistic exposure PSR (av date; av age):", dsr2,psr2,dsr3,psr3)
        }
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      logexVal        <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]]) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
      names(logexVal) <- lexp_name
    } else {
      logexVal <- c() }


  #---- MCMC model: ------------------------------------------------------------------------------------------------------

    if(config$mcmc){
      if(debug>=2) cat("\n\t[*] [*] [*] [*] [*] MCMC model [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      lVal = withCallingHandlers({
        mod$rep_loop(par, rng, nestData, stormDays, survey, pyconfig) # to_r=TRUE
      },
      error=function(e){
        skiptoNext <<- TRUE # need to use super-assignment
        message("error in MCMC model: ", e) # print(sys.calls())
        reticulate::py_last_error()
      })
      # if(debug>=3) cat("\n\t\t|>all MCMC model output:\n") # print(class(lVal))
      # if(debug>=3) qvcalc::indentPrint(lVal)

      llVal <- py_to_r(lVal$astype("float64")) # when it's a np ndarray, this dosn't work
      # if(debug>=3) qvcalc::indentPrint(class(llVal))
      # if(debug>=3) qvcalc::indentPrint(llVal)
      # llDSR <- as.numeric(llVal[0]) llPSR <- as.numeric(llVal[1]) llDFR <- as.numeric(llVal[2])

      # llDSR <- py_to_r(llVal[[1]]) # list does convert, & needs to be 1-indexed?
      # llPSR <- llVal[[2]]
      # llDFR <- llVal[[3]]
      mcmcDSR <- py_to_r(llVal[[1]]) # list does convert, & needs to be 1-indexed?
      mcmcPSR <- llVal[[2]]
      mcmcDFR <- llVal[[3]]
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # if(debug>=5){
      #   qvcalc::indentPrint(class(llVal))
      #   qvcalc::indentPrint(llVal)
      #   qvcalc::indentPrint(class(llDSR))
      #   qvcalc::indentPrint(llDSR)
      # }
      # if(debug>=3) cat("\n\t|> MCMC DSR & PSR = ", llDSR,llPSR)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      mcmc1 <- c(mcmcDSR,mcmcPSR,mcmcDFR)
      # mcmcVal <- c(llDSR,llPSR,llDFR)
      # if(config$debugLL>=2) {
      #   # llArg =
      #   printFun$printLL()
      # }
  }

  #---- Add to DSR matrix: -----------------------------------------------------------------------------------------------
    # nVal <- c(nVal,logexVal,mcmcVal,markVal)
    # dVal <- c(dsrTrue,logexVal,logexSupp,mcmcVal,markVal)
    # mayfDSR = nVal["mfDSR"]
    # dsrT = nVal["aDSR"]
    mayfDSR = mayfield_an
    if(FALSE){ ## if calculate true DSR is false (above) then this should be true
      dsrT = apparent_all
      psrT = dsrT ^ par$hatchTime
    }
    # dsrTrue = c(dsrT,psrT)
    dsrTrue = c(dsrT,psrT,psrT_date)
    if(config$mcmcOld){ mcmcVal <- c(mcmc1, lVal_py ) } else { mcmcVal = mcmc1 }
    if(config$mcmcOld){ mayfVal <- c(mayfDSR, mayfDSR_an) } else { mayfVal = mayfDSR }
    if(config$mark) dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal,markVal)
    # dVal <- c(dsrTrue,logexVal,mcmcVal,mayfDSR,mayfDSR_an)
    dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(debug>=4){
      cat("\n\t\tdVal & length(dVal) for this rep & par set:")
      qvcalc::indentPrint(length(dVal))
      qvcalc::indentPrint(dVal)
      # nVal <- c(nVal,logexVal,markVal)
      cat("\n\t\tdsrMat & its dimensions for this rep & par set:")
      qvcalc::indentPrint(dim(dsrMat))
      qvcalc::indentPrint(dsrMat[,r,i])
      # cat("\ndVal:")
      # print(dVal)
    }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dsrMat[,r,i] <- dVal
    nValMat[,r,i] <- nVal

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if (debug>=3) cat("\n\n\t|>|>dsr/psr Val:\n")
    # if(debug>=3) qvcalc::indentPrint(dVal)
    # if(debug>=5) qvcalc::indentPrint(dsrMat[,r,i])
    #
    # if (debug>=4) cat("\n\n\t|>|>nVal:\n")
    # if(debug>=4) qvcalc::indentPrint(nVal)
    # if(debug>=5) qvcalc::indentPrint(nValMat[,r,i])
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #---- Finish replicate: -----------------------------------------
    if(config$testing=="yes"){
      if(debug>=5) message("\n\n\t>> saving vals to matrix for summary")
      aDSR = nVal["aDSR"]
      aPSR = aDSR ^ par$hatchTime
      # dsrT = nVal["aDSR"]
      # leDSR = dsr1
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
      if(config$mcmcOld) mcmcDSR_old = lVal_py[1]
      if(config$mark) markDSR = mdotDSR
      # print(mcmcDSR_old)
      # mcmcDSR = llDSR
      # markDSR = RM_dsr
      if(psrTrue=="date") {
        psrT = psrT_date
        if(config$debugDSR>=4) cat("\n\t\tpsrT = w/date covar:", psrT)
      }
      mayfDSR = nVal["mfDSR"]
      mayfPSR = mayfDSR ^ par$hatchTime
      if(config$logex) {leDSR = dsr1 } else {leDSR = 0}
      if(config$logex) {lePSR = psr1 } else {lePSR = 0}

      # valMat[,r,i]= c(aDSR,leDSR,mcmcDSR,markDSR,mayfDSR,leDSR-aDSR,mcmcDSR-aDSR,markDSR-aDSR,mayfDSR-aDSR)
      # vals         <- c(dsrT,leDSR,letop,mcmcDSR,markDSR,marktop,mayfDSR)
      # vals         <- c(dsrT,leDSR,mcmcDSR,markDSR,marktop,mayfDSR)
      if(config$mcmcOld){
        if(debug>=2) cat(sprintf("\n\t<> <> DSR vals: assigned=%s, true= %.5f, MCMC=%.5f, MCMC old=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ", par$probSurv, dsrT,mcmcDSR, mcmcDSR_old, leDSR, mayfDSR))
        if(debug>=2) cat(sprintf("\n\t<> <> <> <> <> <> <> <> diff from true: MCMC=%.5f, MCMC old=%.5f, logEx=%.5f, Mayfield=%.5f \n",mcmcDSR-dsrT,mcmcDSR_old-dsrT, leDSR-dsrT, mayfDSR-dsrT))
      } else {
        if(debug>=2) cat(sprintf("\n\t<> <> DSR vals: assigned=%s, true= %.5f, MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ", par$probSurv, dsrT,mcmcDSR,  leDSR, mayfDSR))
        # if(debug>=2) cat(sprintf("\n\t<> <> <> <> diff from apparent: MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> <> <> \n",mcmcDSR-dsrT, leDSR-dsrT, mayfDSR-dsrT))
        if(debug>=2) cat(sprintf("\n\t<> <> <> <> <> <> <> <> diff from true: MCMC=%.5f, logEx=%.5f, Mayfield=%.5f \n",mcmcDSR-dsrT, leDSR-dsrT, mayfDSR-dsrT))
      }
      # if(debug>=2) cat(sprintf("\n\t<> <> <> <> diff from true: MCMC=%s, logEx=%s, Mayfield=%s <> <> <> <> \n",mcmcDSR-dsrT, leDSR-dsrT, mayfDSR-dsrT))
      if(config$mcmcOld){
        vals         <- c(dsrT,aDSR,lDSR,mcmcDSR,mcmcDSR_old,mayfDSR)
        # vals         <- c(psrT,aPSR,lDSR,mcmcDSR,mcmcDSR_old,mayfDSR)
        # diffs        <- c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,mcmcDSR_old-dsrT,mayfDSR-dsrT)
        diffs        <- abs(c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,mcmcDSR_old-dsrT,mayfDSR-dsrT))
        # diffs        <- c(dsrT-aDSR,leDSR-aDSR,mcmcDSR-aDSR,mcmcDSR_old-aDSR,mayfDSR-aDSR)
      } else { 
        # vals         <- c(dsrT,aDSR,leDSR,mcmcDSR,mayfDSR)
        # diffs        <- abs(c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,mayfDSR-dsrT))
        vals         <- c(psrT,psrT_date,aPSR,lePSR,mcmcPSR,mayfPSR)
        diffs        <- abs(c(aPSR-psrT,psrT_date-psrT,lePSR-psrT,mcmcPSR-psrT,mayfPSR-psrT))
        # diffs        <- abs(c(dsrT-aDSR,leDSR-aDSR,mcmcDSR-aDSR,mayfDSR-aDSR))
      }
      # excl=
      # if(debug>=4)  cat("\nstore summary vals here:\n")
      # if(debug>=4)  qvcalc::indentPrint(valMat, indent=8)
      valMat[,r,i] <- c(par$stormFrq,par$pMortFl,par$obsFreq,par$discProb,par$decayRate,par$probSurv,
                        num_disc,num_excl,prop_excl,prop_misclass,vals,diffs)
      # valMat[,r,i] <- c(par$stormFrq,par$pMortFl,par$obsFreq,par$discProb,par$decayRate,par$probSurv,num_disc,num_excl,vals,diffs)
      # valMat[,r,i]= c(aDSR,leDSR,leDSR-aDSR)
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
  cat(sprintf("\n|> PARAMS: prob surv=%s; decay rate=%s; disc prob=%s\n", par$probSurv, par$decayRate, par$discProb))
  if (debug>=2) cat("\nCoefficients:\n")
  if (debug>=2) qvcalc::indentPrint(coefs, indent=8)

  if (debug>=1) cat("\nDSR Val:\n")
  if (debug>=1) qvcalc::indentPrint(dsrMat, indent=8)

  if (debug>=1) cat("\nN Val:\n")
  if (debug>=1) qvcalc::indentPrint(nValMat, indent=8)

  summ <- apply(valMat,c(1,3),mean,na.rm=TRUE) ## pass args to mean after function itself
# if(debug>=4)  qvcalc::indentPrint(valMat)
  if(debug>=1) cat("\nSummary (mean for each param set):\n")
  if(debug>=1)  qvcalc::indentPrint(summ, indent=8)

  valMatList <- asplit(valMat, 3)
  print(class(valMatList))
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

  prBoxPl <- function(val1, valMatList, deb=FALSE){
    # val1 <- "diff_mcmc"
    # cat(sprintf("\n>> difference from true DSR for %s by param %s:", val1, param))
    valStr <- stringr::str_extract(val1, "(?<=_)\\w+")
    # print(valStr)
    # cat(sprintf("\n>> difference between true DSR and %s DSR for each param set: \n\n", valStr))
    cat(sprintf("\n>> difference between apparent DSR and %s DSR for each param set: \n\n", valStr))
    boxPlList  <- lapply(valMatList, function(x) {
    # boxPlList  <- lapply(seq_along(valMatList), function(x) {
                           # print(x["diff_mcmc",]) print(class(x["diff_mcmc",])) x["diff_mcmc",]
                           # x[val1,]
                           unname(x[val1,])
                           # mat = valMatList[[x]]
                           # print(mat)
                           # unname(mat[val1,]
                           # vname = paste0("parSet", x)
                           # assign(vname,unname(mat[val1,]))
          })
    if(deb) print(class(boxPlList))
    names(boxPlList) <- paste0("parSet", seq(boxPlList))
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

# print(prBoxPl("diff_mayf", valMatList, deb=T))
  if(atype=="range" | atype=="nstest"){
    print(prBoxPl("diff_mayf", valMatList))
    print(prBoxPl("diff_lexp", valMatList))
    print(prBoxPl("diff_mcmc", valMatList))
    if(config$mcmcOld) print(prBoxPl("diff_mcmc_old", valMatList))
  }

}

# boxpl <- apply(valMat, c(1,3), function(x){ 
                 # txtboxplot(data=as.data.frame(x, stringsAsFactors=T)) 
      # })
# print(boxpl)
