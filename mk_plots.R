

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
  print(par) # if(debug) print(par$stormFrq)
  stormDays <- nest$stormGen(par$stormFrq, par$stormDur)
  survey    <- withCallingHandlers(
                                   {obs$mk_surveys(stormDays, par$obsFreq, par$brDays, conf=config)},
                                   error=function(e){ 
                                     reticulate::py_last_error() 
                                     # print(sys.calls()) # doesn't help if error in python
                                   }  )

  initDF <- data.frame(day=as.numeric(names(initDateList)),prop=unlist(initDateList,use.names=F))
  initDF$init <- initDF$prop * par$numNests
  cat("\n\tINIT DF:\n")
  qvcalc::indentPrint(initDF)
  # plotFile <- sprintf("%s/figs/%s_inits_density.png",outdir,parID)
  plotFile <- sprintf("figs/%s_inits_density.png",parID)
  cat("\n\tMAKING PLOT\n")
  pl <- ggplot2::ggplot()
  nweeks = round(par$brDays/7)-1
  colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
  # ndata <- lapply(seq(50),
  # ndata_list <- lapply(seq(50), function(x){
  ndata_list <- lapply(seq(config$nreps), function(x){
           # ndata <- obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE)
           # ndata <- obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE)
           obs$make_obs(par,stormDays,survey,config,nweeks,sett$initFromFile,pandas=FALSE) |>
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
  fname <- sprintf("out/plot_data_5s.rds",parID)
  saveRDS(allData, fname)
  # colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs')
  # names(allData) <- colnames
  # allData <- allData |> as.data.frame() |> setNames(colnames)
  # print(names(allData))
  # pl <- ggplot(allData,aes(x=init,group=.id))
  pl <- ggplot(allData,aes(x=init,group=repID))
  pl <- pl + geom_density(color="darkturquoise",alpha=0.03)
  
  # pl <- pl + ggplot2::geom_density(data=initDF,ggplot2::aes(x=init),color="darkblue",alpha=0.9)
  ##alpha doesn't seem to work correctly
  ggplot2::ggsave(plotFile, plot=pl, device="png", width=6,height=4,units="in")
  # pl <- 

  parID = parID + 1
}
