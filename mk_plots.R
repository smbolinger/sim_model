

library(ggplot2)
library(reticulate)
library(dplyr) # load dplyr last so as not to mask select?
source("lexp_fun.R")
source("lexp_setup.R")

parID = 0 
for(i in seq(length(pArrList))){

#---- Make params, storms, surveys: --------------------------------------------------------------
  par <- tryCatch(
                  {funs$mk_param_list(paramsArray[i-1], staticPar)},
                  error=function(e){
                  reticulate::py_last_error()
                  })
  cat("\n\tPARAMS:\n")
  print(par) # if(debug) print(par$stormFrq)
  stormData <- list()
  # stormTable <- data.table::rbindlist(stormdata_list,idcol=T)
  # stormdata_list <- lapply(seq(config$nreps), function(x){
  for(x in seq(config$nreps)){
                    # stormData[[x]] <- nest$stormGen(par$stormFrq, par$stormDur,config,rng)
                    stormData[[x]] <- data.frame(ID=x,day=nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng))
                    # as.data.frame() |>
                    # setNames(colnames) |>
                    # mutate(repID=x)
  }
                  # })
  # print(stormData)
  cat("\n\tstormData=\n")
  qvcalc::indentPrint(stormData)
  # stormDF <- data.frame(ID=names(unlist(stormData)), days=unlist(stormData))
  # stormDF <- as.data.frame(sapply())
  # stormDat <- sapply()
  stormDF <- data.table::rbindlist(stormData)
  cat("\n\tstormDF=\n")
  qvcalc::indentPrint(stormDF)
  ## storm days & survey days don't actually affect initiation dates, but need them to run functions:
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur,pyconfig,rng) # pl <- ggplot2::ggplot()
  survey    <- withCallingHandlers(
                                   {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=pyconfig)},
                                   error=function(e){ 
                                   reticulate::py_last_error() # print(sys.calls()) # doesn't help if error in python
                                   }  )

  if(FALSE){
    initDF <- data.frame(day=as.numeric(names(initDateList)),prop=unlist(initDateList,use.names=F))
    initDF$init <- initDF$prop * par$numNests
    cat("\n\tINIT DF:\n")
    qvcalc::indentPrint(initDF)
  }
  # plotFile <- sprintf("%s/figs/%s_inits_density.png",outdir,parID)
  cat("\n\tMAKING PLOT OF INITIATION DATES\n")
  # pl <- ggplot2::ggplot()
  nweeks = round(par$brDays/7)-1
  colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
  # ndata <- lapply(seq(50),
  # ndata_list <- lapply(seq(50), function(x){
  cat(sprintf("\n\t|> py config = %s <type=%s>", pyconfig, class(pyconfig)))
  print(pyconfig)
  ndata_list <- lapply(seq(config$nreps), function(x){
           # ndata <- obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE)
           # ndata <- obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE)
           obs$make_obs(par,rng,stormDays,survey,pyconfig,nweeks,sett$initFromFile,pandas=FALSE) |>
             as.data.frame() |>
             setNames(colnames) |>
             mutate(repID=x)
            # if (config$debugNests>=4) cat("\nall nest data:\n")
            # if (config$debugNests>=4) qvcalc::indentPrint(ndata)
           # pl <- pl + ggplot2::geom_density(data=nestData,ggplot2::aes(x=!!enquo(init)),color="darkturquoise",alpha=0.5) ## force evaluation 
           # pl <- pl + ggplot2::geom_density(data=ndata,aes(x=init),color="darkturquoise",alpha=0.5) ## force evaluation
         })
  if(debug>=3) print(str(ndata_list))
  allData <- data.table::rbindlist(ndata_list,idcol=T)
  print(class(allData))
  print(head(allData))
  # fname <- sprintf("out/%s/plot_data_%s.rds",config$rngSeed,parID)
  fname <- sprintf("figs/plot_data_%s.rds",parID)
  saveRDS(allData, fname)
  fname2 <- sprintf("figs/plot_stormdata_%s.rds",parID)
  saveRDS(stormDF, fname2)
  # colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
  # names(allData) <- colnames
  # allData <- allData |> as.data.frame() |> setNames(colnames)
  # print(names(allData))
  # pl <- ggplot(allData,aes(x=init,group=.id))
  pl <- ggplot(allData,aes(x=init,group=repID))
  pl <- pl + geom_density(color="darkturquoise",alpha=0.03)
  # pl <- pl + labs(y=
  # pl <- pl + ggplot2::geom_density(data=initDF,ggplot2::aes(x=init),color="darkblue",alpha=0.9) alpha doesn't seem to work correctly
  plotFile <- sprintf("figs/%s_inits_density.png",parID)
  plotFile2 <- sprintf("figs/%s_storms_density.png",parID)
  ggplot2::ggsave(plotFile, plot=pl, device="png", width=6,height=4,units="in")
  cat("\n\tMAKING PLOT OF STORM DAYS\n")
  pl2 <- ggplot(stormDF, aes(x=day, group=ID)) 
  pl2 <- pl2 + geom_density(color="darkslateblue",alpha=0.03)
  ggplot2::ggsave(plotFile, plot=pl2, device="png", width=6,height=4,units="in")

  parID = parID + 1
}
