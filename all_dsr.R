
## attempting to streamline
## takes parts of logexp.R and datsim.py 

startTime <- Sys.time()
library(reticulate)
Sys.setenv(script_name="logexp.R")
py_run_file("init.py")
library(MASS)
suppressPackageStartupMessages(library(dplyr)) # load dplyr last so as not to mask select?
library(brglm2)
# library(jsonlite)
#NOTE could make a counter of all times at least one survey int == 0

#---- LOAD FUNCTIONS & VARIABLES --------------------------------------------------------------
source("lexp_fun.R")
source("lexp_setup.R")
Sys.setenv(r_outdir=py_to_r(outdir))
cat(sprintf("\n\tin R: outdir=%s & type=%s", py_to_r(outdir),class(py_to_r(outdir))))
# if(config$mark) library(RMark)
options(width=1000, digits=5, scipen=999)

parID = 0
for(i in seq(length(pArrList))){
  # if(debug) cat("\n\n...............................i=",i, ".........................................................................................................................................................\n")
  cat("\n\n...............................i=",i, ".........................................................................................................................................................\n")

#---- Make params, storms, surveys: --------------------------------------------------------------
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  # print(par) # if(debug) print(par$stormFrq)
  print(class(par))
  print(unlist(py_vars(par)))
  cat(".............................................................................................................................................................................................\n")
  # stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng, stFromFile=sett$stormFromFile)

  stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=sett$stormFromFile,db=debug)
  survey    <- withCallingHandlers(
                                   # {obs$mk_surveys(stormDays,par$obsFreq,par$brDays,conf=pyconfig,db=debug,complicate=F)},
                                   {obs$mk_surveys(stormDays,par$obsFreq,par$brDays,conf=pyconfig,db=debug,complicate=T)},
                                   error=function(e){ 
                                     reticulate::py_last_error() 
                                     # print(sys.calls()) # doesn't help if error in python
                                   }  )
  
  # mod$main(fnUnique=config$fnUnique,
  #      parLists=pLists,
  #      stormdays=stormDays,
  #      msg=config$msg,
  #      testing=config$testing)

#----------------------------------------------------
  repID=0
  for(r in seq(nreps)){
    if(debug>=1) cat(sprintf("\n:::::::::::::::::::::::::::::: rep %s",i))
    cat(sprintf("-%s",r))
    if(debug>=1) cat(":::::::::::::::::::::::::::::::::::::::::::\n")

    if(TRUE){
      # if(debug>=2) cat("\n\t>> overwriting storm days for each rep\n")
      if(debug>=2) cat("\n\t>> overwriting storm days -")
      stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng,stormDat, stFromFile=sett$stormFromFile,db=debug)
      if(debug>=2) cat(stormDays)
      survey    <- withCallingHandlers(
                                       {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=pyconfig,db=debug)},
                                       error=function(e){ 
                                         reticulate::py_last_error() 
                                         # print(sys.calls()) # doesn't help if error in python
                                       }  )
    }


  #---- Full nest data: ----------------------------------------------------
    skiptoNext <- FALSE
    nestData1 <- withCallingHandlers({
      nweeks = round(par$brDays/7)-2
      # nweeks = floor(par$brDays/7)
      # nweeks = par$brDays//7
      obs$make_obs(par,rng,stormDays,survey,pyconfig,initDat,nweeks,sett$initFromFile,pandas=FALSE)
    },
    error=function(e){
      skiptoNext <<- TRUE # need to use super-assignment
      message("error in nest data: ", e, "; go to next replicate. (turn on print(sys.calls) for more from R)") # print(sys.calls())
      reticulate::py_last_error()
    })
    if(skiptoNext) { next }
    # if (config$debugNests>=3){
    #   cat("\n\t|>creating nVals")
    #   cat("(par,rep,flood,hatch,disc,excl,unkn,misclass\n")
    #   cat("\t\t\tavFint,avK,true DSR, true PSR, mayfield, apparent)\n")
    # }
    nVal <- mod$calc_nests(nestData1, par, rng, repID, parID,pyconfig,db=config$debugNests)
    colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
    nestData1 <- nestData1 |> as.data.frame(row.names=NULL) |> setNames(colnames)
    names(nVal) <- nval_name

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(debug>=2) cat("\n\t** nVal:\n")
    if(debug>=2) qvcalc::indentPrint(nVal, indent=8)
    if(debug>=2) cat("\n")
    if(debug>=3) cat("\n\t[*] [*] [*] [*] [*]  NEST DATA [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
    if(debug>=4) cat("\n\t\t** all nest data:\n") # if (config$debugNests>=3) qvcalc::indentPrint(nestData1)
    if(debug>=4) qvcalc::indentPrint(nestData1, indent=8) # if (config$testing=="yes") lines(density(nestData1$init),col="green",)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    all_fld <- sum(nestData1$fate==2, na.rm=TRUE)
    all_hatch <- sum(nestData1$fate==0, na.rm=TRUE)
    all_longfin <- sum(nestData1$fint>par$obsFreq, na.rm=TRUE)
    all_misclass <- sum(nestData1$fate!=nestData1$afate,na.rm=TRUE)
    all_unk <- sum(nestData1$afate==7, na.rm=TRUE)
    # an_misclass <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
    apparent_all <- dsr$calc_dsr(nData=nestData1,nestType="all", calcType="apparent",
                                  conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)


  #---- Nest data - discovered: ----------------------------------------------------
    # disc     <- sum(nestData1$totobs < 1, na.rm=TRUE)
    nestData <- nestData1 |> filter(totobs>0) # remove undiscovered nests
    num_disc <- nrow(nestData)
    # if (config$debugNests>=3) cat("\n\t** discovered nests (1 to 15):\n")
    # if (config$debugNests>=3) qvcalc::indentPrint(head(nestData,15))
    # if(debug>=3) cat(sprintf("\n\t** discovered nests (length=%s):\n", nrow(nestData)))
    disc_fld <- sum(nestData$fate==2, na.rm=TRUE)
    disc_hatch <- sum(nestData$fate==0, na.rm=TRUE)
    disc_longfin <- sum(nestData$fint>par$obsFreq, na.rm=TRUE)
    num_misclass <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
    
    # if(debug>=3) cat(sprintf("\n\tdiscovered nests-final interval length:"))
    # if(debug>=3) print(nestData$fint)

    num_excl <- sum(nestData$afate==7, na.rm=TRUE)
    num_an <- num_disc - num_excl
    prop_excl <- num_excl/num_disc
    prop_misclass <- num_misclass/num_disc

    mayfield_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="mayfield",
                                  conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

    apparent_disc <- dsr$calc_dsr(nData=nestData,nestType="discovered", calcType="apparent",
                                  conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(debug>=3) cat(sprintf("\n\t\t>>> discovered nests (length=%s):\n", num_disc))
    if(debug>=3 & debug<4) qvcalc::indentPrint(head(nestData,25), indent=8)
    if(debug>=4) qvcalc::indentPrint(nestData)
    if(debug>=2) cat(sprintf("\t\t\tfor discovered nests: Mayfield DSR=%s ; apparent DSR=%s\n", mayfield_disc, apparent_disc))
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Nest data - analyzed: ----------------------------------------------------
    nestData <- nestData |> filter(afate!=7) |> na.omit() # remove unknown fate nests
    an_fld <- sum(nestData$fate==2, na.rm=TRUE)
    an_hatch <- sum(nestData$fate==0, na.rm=TRUE)
    an_misclass <- sum(nestData$fate!=nestData$afate,na.rm=TRUE)
    an_longfin <- sum(nestData$fint>par$obsFreq, na.rm=TRUE)

    mayfield_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="mayfield",
                                  conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

    apparent_an <- dsr$calc_dsr(nData=nestData,nestType="analysis", calcType="apparent",
                                  conf=config,incTime=par$hatchTime,psurv=par$probSurv,debug=config$debugDSR)

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(debug>=3) cat(sprintf("\n\t\t>>> analyzed nests (length=%s ; num excluded=%s):\n",nrow(nestData), num_excl))
    if(debug>=3 & debug<4) qvcalc::indentPrint(head(nestData,25), indent=8)
    if(debug>=4) qvcalc::indentPrint(nestData, indent=8)
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
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #---- Program MARK: ----------------------------------------------------
    if(config$mark){
      # if (debug>=2) cat("\n<*><*><*> Run RMark <*><*><*><*><*>\n")
      library(RMark)
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

  #---- Calculate DSR from datsim.py: ----------------------------------------------------
    if(config$mcmcOld){
      par1 <- dc$asdict(par)
      config1 <- dc$asdict(pyconfig)
      # saveDat <- list("par"=par1, "stormDays"=stormDays, "survey"=survey, "nestData1"=nestData1)
      # NOTE also can't save ndarrays as json
      # saveDat <- list("config"=config1,"par"=par1, "stormDays"=stormDays, "survey"=survey)
      saveDat <- list("outdir"=outdir,"parID"=i,"repID"=r,"config"=config1,"par"=par1, "stormDays"=stormDays, "survey"=survey)
      # cat("pickling")
      # print(saveDat)
      # needs to be saved where it won't accidentally be read by another instance of script:
      pklFile <- sprintf("%s/data.pkl",outdir)
      ndFile <- sprintf("%s/nestData1.csv", outdir)
      # py_save_object(saveDat, "analysis/data.pkl") ## can directly use the pickle module w/functions
      py_save_object(saveDat, pklFile) ## can directly use the pickle module w/functions
      # with open("analysis/data.pkl", "wb") as file:

      # write.csv(nestData1, "analysis/nestData1.csv", sep=",",row.names=F)
      # write.csv(nestData1, ndFile, sep=",",row.names=F)
      write.csv(nestData1, ndFile, row.names=F)
      # Sys.sleep(10)
      # write_json(py$saveDat,"analysis/data.json")
      # datStr <- json$dumps(saveDat)
      # writeLines(datStr,"analysis/data.json")
      # write_json(saveDat,"analysis/data.json") ## it's got python objects, so can't use this function w/o editing
      # pyfile <- paste0(homeDir, "/datsim2.py")
      # sysCmd <- sprintf("poetry run python3 %s %s",pyfile,outdir)
      # system(sysCmd)
      # source_python("datsim2.py")
      ## objects aren't automatically in environment:
      py_run_file("datsim2.py")
      # lVal_py <- py_to_r(py$lVal_py) ## I have no idea what class "environment" is
      lVal_py <- py$lVal_py 
      mayfDSR_an <- py$mayfDSR_an
      # print(result_from_python)
      # if (debug>=3) cat(sprintf("\n\t>>->> python output: %s length: %s", paste(lVal_py, collapse=" ; "), length(lVal_py),class(lVal_py)))
      # mcmc2 <- lVal_py
      # mcmc2 <- c(lDSR,lPSR,lDFR) ## vals from datsim2.py
      # print(lVal)
      # names(save
      # mcmcVal  <- c(mcmc1, mcmc2)
    }


  #---- Calculate true DSR: ----------------------------------------------------
    if(TRUE){
      prDays  <- seq(1, max(nestData1$init))
      # if(debug>=4) cat("\n\tdates for prediction: ", prDays, length(prDays))
      truePSR <- nVal["hat"] / par$numNests
      # if(debug>=2) cat(sprintf("\n\t|>|> true PSR: %s [num hatched] / %s [num total] = %s):", nVal["hat"],par$numNests,truePSR))
      # if(debug>=3) cat(sprintf("\n\toooo|> true PSR: %s [num hatched] / %s [num total] = %s):", nVal["hat"],par$numNests,truePSR))

      ## needs to be all nests and all days (not just observed)
      # coef_out <- calc_logexp(mList, nestData, survey=survey, config=config, exposure=1, debug=config$debug) 
      modData <- mk_logex_data( nestData1, survey=survey, pyconfig=pyconfig, exposure=1 ) 
      # coef_out <- calc_logexp(mList, nestData1, survey=survey, config=config, exposure=1, debug=config$debug) 
      # coef_out <- calc_logexp(mList, modData, config=pyconfig) 
      mList_true <- c("Surv~1", "Surv~Date")
      # coef_out <- calc_logexp(mList, modData, config=config) 
      coef_out <- calc_logexp(mList_true, modData, config=config) 
      newDat <- data.frame(Date=prDays)
      dsrList <- make_pred(coef_out, nmod=2, mods=mList_true, newDat=newDat,hTime=par$hatchTime, db=config$debugLogEx)

      # dsrT <-  1/(1+exp(-coef_out[[1]][1,1]))
      # dsrT <-  1/(1+exp(-coef_out[[2]][1,1] + coef_out[[2]][1,2] * modData$avAge))
      # propInit <- 
      allInits    <- nestData1$init
      numInit     <- sapply(newDat$Date, function(x) sum(allInits==x))
      propInit    <- numInit/par$numNests
      dsrT <- sum(propInit * dsrList[[2]])
      psrT <- dsrT ^ par$hatchTime
      # dsrTrue <- c(dsrT,psrT)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(debug>=4) cat("\n\t\t\t>-> coef_out:\n")
      if(debug>=4) qvcalc::indentPrint(coef_out, indent=8)
      if(debug>=2) cat("\n\t|> true DSR & PSR: logistic exposure (no covars):", dsrT, psrT, "\n")
      if(debug>=5){
        cat("\n\t\t\t>-> DSR vals:\n")
        qvcalc::indentPrint(dsrList[[2]], indent=8)
        cat("\n\t\t\t\t>-> inits & dates:\n")
        qvcalc::indentPrint(allInits, indent=8)
        qvcalc::indentPrint(newDat$Date, indent=8)
        # cat("\nnum inits before date:\n")
        cat("\n\t\t\t\t>-> num inits on date:\n")
        qvcalc::indentPrint(numInit, indent=8)
        cat(sprintf("\n\t\t\t\t>-> proportion inits on date (sum=%s):\n", sum(propInit)))
        qvcalc::indentPrint(propInit, indent=8)
      }
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    } 

  #---- Logistic exposure: -----------------------------------------------------------------------------------------------
    if(config$logex){
      # if(debug>=3) cat("\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      if(debug>=2) cat("\n\t[*] [*] [*] [*] [*] logistic exposure [*] [*] [*] [*] [*] [*] [*] \n") # if (config$debugNests] =3) qvcalc::indentPrint(nestData1)
      modData <- mk_logex_data( nestData, survey=survey, pyconfig=pyconfig, exposure=0) 
      modOut <- calc_logexp(mList,modData,config=config)
      # if(debug>=2) cat("\n<*><*> Logistic exposure <*><*><*><*>\n")
      # excpt <- FALSE
      nNest <- nrow(nestData) # cat("\nnumber of nests:", nNest)
      # # nestObs <- nestData |> dplyr::select(ID, init, i, j, k, afate, totobs) # print(head(nestObs))
      nestObs <- nestData |> dplyr::select(ID, init,end,fate, i, j, k, afate) # print(head(nestObs))
      numObs <- nestData[,"totobs"]
      dat2S = modData
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if(debug>=2) {
        cat("\n\t\t>>> dat2S:\n")
        qvcalc::indentPrint(head(modData, 30), indent=8)
      }
      #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
      # if(debug>=4) cat("\n\tmodOut:")
      # if(debug>=4) qvcalc::indentPrint(modOut)
      if(config$coefSave!="none") coefs[,r,i] = unlist(modOut)
      coefsArray = modOut
      # if (debug>=4) cat("\n\t<> coefficients:\n")
      # # if (debug>=4) qvcalc::indentPrint(coefs[,r,i])
      # if (debug>=4) qvcalc::indentPrint(coefsArray)
    }

  #---- Logexp DSR & PSR: -----------------------------------------------------------------------------------------------
    if(config$logex){
      dsr1        <- 1/(1+exp(-coefsArray[[1]][1,1]))
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

    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # mList2 <-
      dsrList <- make_pred(coefsArray, nmod, mList, newDat=dat2S,hTime=par$hatchTime, db=config$debugLogEx)
      dsrList <- dsrList[-1]
      ## NOTE make_pred already starts at model number 2, so why exclude 1st output??
      ## NOTE because it is empty bc index starts at 2 for list as well
      psrList <- lapply(dsrList, function(x) x^par$hatchTime)
      # if(debug>=5){
      if(config$debugLogEx>=4){
        cat(sprintf("\n\t\t\t|> output of make_pred (dsrList:%s & psrList:%s):\n", length(dsrList), length(psrList)))
        qvcalc::indentPrint(dsrList, indent=8)
        qvcalc::indentPrint(psrList, indent=8)
      }
      # psr     <- make_psr(psrList, propInitScl, db=config$debugLogEx) # nVal <- c(nVal, dsr1, psr1,psr[[1]], psr[[2]],psr[[3]],psr[[4]])
      psr <- lapply(psrList, function(x) sum(x*propInitScl))
      # if(config$debugLogEx>=1) cat(sprintf("\n\t\t>> psr (avg psrList weighted by nest initiation per day), excluding intercept-only model: %s\n ", unlist(psr)))
      if(config$debugLogEx>=1) cat("\n\t\t>> psr (avg psrList weighted by nest initiation per day), excluding intercept-only model: ", unlist(psr), "\n")
      # if(config$debugLogEx>=3) qvcalc::indentPrint(psr)
      # if(debug>=3){
      #   cat("\n\t|> logistic exposure DSR & PSR (no covars):", dsr1, psr1)
      #   # cat("\n|> logistic exposure DSR & PSR (average date):", dsr2,psr2)
      #   cat("\n\t|> logistic exposure PSR (mods 2-5):", paste(psr,collapse=" ; "))
      #   # cat("\n\t|> logistic exposure PSR (av date; av age):", dsr2,psr2,dsr3,psr3)
      # }

      # logexVal <- c(dsr1,psr1,dsr2,psr2,psr[[1]],psr[[2]],psr[[3]],psr[[4]]) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
      # logexVal        <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]],dsrT,psrT) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
      logexVal        <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]]) # logexVal <- c(dsr1,psr1,psr[[1]],psr[[2]],psr[[3]],psr[[4]])
      # logexSupp       <- c(dsr2,psr2,dsr3,psr3)
      names(logexVal) <- lexp_name
      # names(logexSupp) <- lexp_supp
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

      llDSR <- py_to_r(llVal[[1]]) # list does convert, & needs to be 1-indexed?
      llPSR <- llVal[[2]]
      llDFR <- llVal[[3]]
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # if(debug>=5){
      #   qvcalc::indentPrint(class(llVal))
      #   qvcalc::indentPrint(llVal)
      #   qvcalc::indentPrint(class(llDSR))
      #   qvcalc::indentPrint(llDSR)
      # }
      # if(debug>=3) cat("\n\t|> MCMC DSR & PSR = ", llDSR,llPSR)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      mcmc1 <- c(llDSR,llPSR,llDFR)
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
    dsrTrue = c(dsrT,psrT)
    if(config$mcmcOld){ mcmcVal <- c(mcmc1, lVal_py ) } else { mcmcVal = mcmc1 }
    if(config$mcmcOld){ mayfVal <- c(mayfDSR, mayfDSR_an) } else { mayfVal = mayfDSR }
    if(config$mark) dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal,markVal)
    # dVal <- c(dsrTrue,logexVal,mcmcVal,mayfDSR,mayfDSR_an)
    dVal <- c(dsrTrue,logexVal,mcmcVal,mayfVal)
    #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if(debug>=4){
    #   cat("\ndVal & length(dVal) for this rep & par set:")
    #   print(length(dVal))
    #   print(dVal)
    #   # nVal <- c(nVal,logexVal,markVal)
    #   cat("\ndsrMat & its dimensions for this rep & par set:")
    #   print(dim(dsrMat))
    #   print(dsrMat[,r,i])
    #   # cat("\ndVal:")
    #   # print(dVal)
    # }
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
      # print(mcmcDSR_old)
      mcmcDSR = llDSR
      # markDSR = RM_dsr
      if(config$mark) markDSR = mdotDSR
      mayfDSR = nVal["mfDSR"]
      if(config$logex) {leDSR = dsr1 } else {leDSR = 0}

      # valMat[,r,i]= c(aDSR,leDSR,mcmcDSR,markDSR,mayfDSR,leDSR-aDSR,mcmcDSR-aDSR,markDSR-aDSR,mayfDSR-aDSR)
      # vals         <- c(dsrT,leDSR,letop,mcmcDSR,markDSR,marktop,mayfDSR)
      # vals         <- c(dsrT,leDSR,mcmcDSR,markDSR,marktop,mayfDSR)
      if(config$mcmcOld){
        if(debug>=2) cat(sprintf("\n\t<> <> DSR vals: assigned=%s, true= %.5f, MCMC=%.5f, MCMC old=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ", par$probSurv, dsrT,mcmcDSR, mcmcDSR_old, leDSR, mayfDSR))
        if(debug>=2) cat(sprintf("\n\t<> <> <> <> diff from apparent: MCMC=%.5f, MCMC old=%.5f, logEx=%.5f, Mayfield=%.5f <> <> <> <> \n",mcmcDSR-dsrT,mcmcDSR_old-dsrT, leDSR-dsrT, mayfDSR-dsrT))
      } else {
        if(debug>=2) cat(sprintf("\n\t<> <> DSR vals: assigned=%s, true= %.5f, MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> ", par$probSurv, dsrT,mcmcDSR,  leDSR, mayfDSR))
        if(debug>=2) cat(sprintf("\n\t<> <> <> <> diff from apparent: MCMC=%.5f, logEx=%.5f, Mayfield=%.5f <> <> <> <> \n",mcmcDSR-dsrT, leDSR-dsrT, mayfDSR-dsrT))
      }
      # if(debug>=2) cat(sprintf("\n\t<> <> <> <> diff from true: MCMC=%s, logEx=%s, Mayfield=%s <> <> <> <> \n",mcmcDSR-dsrT, leDSR-dsrT, mayfDSR-dsrT))
      if(config$mcmcOld){
        vals         <- c(dsrT,aDSR,leDSR,mcmcDSR,mcmcDSR_old,mayfDSR)
        # diffs        <- c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,mcmcDSR_old-dsrT,mayfDSR-dsrT)
        diffs        <- abs(c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,mcmcDSR_old-dsrT,mayfDSR-dsrT))
        # diffs        <- c(dsrT-aDSR,leDSR-aDSR,mcmcDSR-aDSR,mcmcDSR_old-aDSR,mayfDSR-aDSR)
      } else { 
        vals         <- c(dsrT,aDSR,leDSR,mcmcDSR,mayfDSR)
        diffs        <- abs(c(aDSR-dsrT,leDSR-dsrT,mcmcDSR-dsrT,mayfDSR-dsrT))
        # diffs        <- abs(c(dsrT-aDSR,leDSR-aDSR,mcmcDSR-aDSR,mayfDSR-aDSR))
      }
      # excl=
      valMat[,r,i] <- c(par$stormFrq,par$pMortFl,par$obsFreq,par$discProb,par$decayRate,par$probSurv,num_disc,num_excl,prop_excl,prop_misclass,vals,diffs)
      # valMat[,r,i] <- c(par$stormFrq,par$pMortFl,par$obsFreq,par$discProb,par$decayRate,par$probSurv,num_disc,num_excl,vals,diffs)
      # valMat[,r,i]= c(aDSR,leDSR,leDSR-aDSR)
      if(debug>=5)  cat("\nstore summary vals:\n")
      if(debug>=5)  qvcalc::indentPrint(valMat, indent=8)
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

cat(sprintf("\n|>|> TOTAL RUN TIME: %s\n", runTime))

if(config$testing=="yes"){
  cat(sprintf("\n|> PARAMS: num nests=%s; prob surv=%s; decay rate=%s; disc prob=%s\n", par$numNests,par$probSurv, par$decayRate, par$discProb))
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
