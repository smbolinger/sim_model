if (config$predict){
  dimss <- c(3,preDays,nmod,length(pArrList)) # print(dimss)
  dnames <- list(c("est", "lcl", "ucl"), seq(preDays),mNames,seq(length(pArrList))) # print(dnames)
  pred <- array(NA, dim=dimss, dimnames=dnames )
  if(debug>=6) print(dimnames(pred))
}
## NOTE: this only includes the first 5 models
# coef_list <- c("est","lcl","ucl")
coef_list <- c("est","se")
# mod_names <- c("_dot","_date_int","_date_b1","_datesq_int","_datesq_b1","_datesq_b2",
#                "age_int","age_b1","agedate_int","agedate_b1","agedate_b2")
mod_names <- c("_null","_date_int","_date_b1", "_age_int","_age_b1",
               "_agedate_int","_agedate_b1","_agedate_b2","_avdate_int","_avdate_b1")
# coef_names <- do.call(paste0, expand.grid(coef_list, mod_names))
coef_names <- unlist(lapply(coef_list, function(x) paste0(x,mod_names)))
coef_names <- c(coef_names, "avg_expos","avg_age","avg_avdate")
# if(debug>=0) print(seq(nreps))
# if(debug>=0) print(class(nreps))

## rows should come first (will keep column names when splitting matrix later on)
coefsMat <- array(NA,
               dim=c(nreps, length(coef_names), length(pArrList)),
               dimnames=list(seq(nreps),coef_names, seq(length(pArrList))))
# if(debug>=0) qvcalc::indentPrint(dimnames(coefs))
if(debug>=0) cat("\n\tcoef names:", coef_names)
# if(debug>=0) qvcalc::indentPrint(coef_names)
coefList <- list()

modNames <- list(c("null"),
                 c("int","date"),
                 c("int","age"),
                 c("int","age","date"),
                 c("int","avdate"))
vcovNames <- unlist(sapply(modNames, function(x) outer(x, x, paste, sep="-")))
# vcovNames <- sapply(modNames, function(x) unlist(outer(x, x, paste, sep="-")))
cat("\n vcov names, length=", length(vcovNames))
print(vcovNames)

# vcovNames <- sapply(modNames, function(x){ paste(expand.grid(modNames[[x]]))

               # })

## rows should come first (will keep column names when splitting matrix later on)
vcovMatMat <- array(NA,
               dim=c(nreps, length(vcovNames), length(pArrList)),
               dimnames=list(seq(nreps),vcovNames, seq(length(pArrList))))

print(dimnames(vcovMatMat))

cat("\n initPropMat, dimnames:")

## rows should come first (will keep column names when splitting matrix later on)
initPropMat <- array(NA,
               dim=c(nreps, preDays, length(pArrList)),
               dimnames=list(seq(nreps),prDays, seq(length(pArrList))))

print(dimnames(initPropMat))

# lexp_name <- c("leDSR1","lePSR1","leDSRdate","lePSRdate","lePSR2","lePSR3","lePSR4","lePSR5")
if(config$logex){
  # lexp_name <- c("leDSR1","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","lePSR6")
  # lexp_name <- c("leDSR1","lePSR1","leDSR2","leDSR3","leDSR4","leDSR5","leDSR6","lePSR2","lePSR3","lePSR4","lePSR5","lePSR6")
#   lexp_name_p <- c("leDSR1p","leDSR2p","leDSR3p","leDSR4p","leDSR5p","lePSR1p","lePSR2p","lePSR3p","lePSR4p","lePSR5p")
#   lexp_se_p <- c("seDSR1p","seDSR2p","seDSR3p","seDSR4p","seDSR5p")
#   lexp_name <- c("leDSR1","lePSR1","seDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR2","lePSR3","lePSR4","lePSR5")
#   lexp_se <- c("seDSR2","seDSR3","seDSR4","seDSR5")
# # lexp_supp <- c("leDSRava","lePSRava","leDSRavd","lePSRavd","leDSRavad","lePSRavad")
#   lexp_supp <- c("leDSRavd","lePSRavd","leDSRava","lePSRava")
#   lexp_all <- c(lexp_name,lexp_se,lexp_name_p,lexp_se_p)
#   lexpMat <- array(NA,
#                   dim=c(length(lexp_all), nreps,length(pArrList)),
#                   dimnames=list(lexp_all,seq(nreps), seq(length(pArrList))))
#   cat("\n\tlexp matrix dim:", dim(lexpMat))
#   cat("\t\t& lexp val names:", lexp_all)
  # lexp_name <- c("leDSR1","lePSR1","seDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR2","lePSR3","lePSR4","lePSR5")
  lexp_dsr <- c("leDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
  lexp_se <- c("seDSR1","seDSR2","seDSR3","seDSR4","seDSR5")
  # lexp_cov <- c("covDSR1","covDSR2","covDSR3","covDSR4","covDSR5")
  # lexp_name <- c(lexp_dsr, lexp_se. lexp_cov)
  lexp_name <- c(lexp_dsr, lexp_se)
# lexp_supp <- c("leDSRava","lePSRava","leDSRavd","lePSRavd","leDSRavad","lePSRavad")
  lexp_supp <- c("leDSRavd","lePSRavd","leDSRava","lePSRava")
# truedsr_name <- c("tDSR","tPSR","tDSRdate","tPSRdate","tDSRage","tPSRage")
} else {
  lexp_name <- c()
  lexp_supp <- c()
}
if(config$survSave!="all") lexp_name <- c()

truedsr_name <- c("tDSR","tPSR","tDSR_date","tPSR_date")
# truedsr_name <- c("tDSR","tPSR","tPSR_date","tDSR2","tPSR2","tPSR_date2")
# if(config$mcmcOld){
#   mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDFR","mcmcDSR_old","mcmcPSR_old","mcmcDFR_old")
#   mayfdsr_name <- c("mayfDSR","mayfDSR_old")
# } else {
if(config$mcmc){
  mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcDSR_se","mcmcDPR_se")
  # mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcDSR_se","mcmcDPR_se","mcmcDSR_cov")
} else {
  mcmc_name <- c()
}
# mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcSE")
mayfdsr_name <- c("mayfDSR","mayfPSR","mayfVar","mayfSE","simplePSR")
  # mayfdsr_name <- c("mayfDSR")
# }
mark_name <- c()
if(config$mark) mark_name <- c("markDSR","markPSR","markDSRdate","markPSRdate","markDSRdsAge","markPSRdsAge","marktopDSR","marktopmod")

# dsr_name  <- c(truedsr_name,lexp_name,lexp_supp, mcmc_name, mark_name)

# dsr_name  <- c("parID","repID",truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
# dsr_name  <- c(truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
# dsr_name  <- c(truedsr_name,lexp_name,lexp_se, mcmc_name,mayfdsr_name, mark_name)
dsr_name  <- c(truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)
# print(dsr_name)
# print(length(dsr_name))

## rows should come first (will keep column names when splitting matrix later on)
dsrMat <- array(NA,
                # dim=c(length(dsr_name), nreps,length(pArrList)),
                # dimnames=list(dsr_name,seq(nreps), seq(length(pArrList))))
                dim=c( nreps,length(dsr_name),length(pArrList)),
                dimnames=list(seq(nreps),dsr_name, seq(length(pArrList))))
cat("\n\tDSR matrix dim:", dim(dsrMat))
cat("\t\t& DSR val names:", dsr_name)

# print(dim(dsrMat))
# print(dimnames(dsrMat))
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mfDSR","appDSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
                # 1       2     3       4     5       6     7     8     9       10      11    12      13    14        
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc",
# nval_name <- c("parID","repID","fld", "hat", "sNest", "dsc", "excl","unk","mc",
               # "avfint","avk","maxi","aDSR","aPSR","mfDSR","appDSR")
nval_name <- c("parID","repID","fld", "hat","fld_dsc","hat_dsc","fld_an",
               "hat_an", "sNest", "dsc", "excl","unk","mc","mc2", "avfint",
               "avk","maxi","lint","aDSR","aPSR","mfDSR","appDSR")
cat("\n\tnVal names:", nval_name)
# dsr_name <- c("leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","mcmcDSR","mcmcPSR","mcmcDFR","markDSR","markPSR")
# allval_name <- c(nval_name, dsr_name,mcmc_name,mark_name)

## rows should come first (will keep column names when splitting matrix later on)
nValMat <- array(NA,
                 # dim=c(length(nval_name), nreps,length(pArrList)),
                 # dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))
                 dim=c( nreps,length(nval_name),length(pArrList)),
                 dimnames=list(seq(nreps),nval_name, seq(length(pArrList))))
# allval_name <- c(nval_name, dsr_name,mark_name)
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5","rmDSR","rmPSR")
# nval_name <- c("parID","repID","fld", "hat", "dsc", "excl","unk","mc","avfint","avk","aDSR","aPSR","mDSR","mPSR","leDSR","lePSR1")
# nValMat <- array(NA, dim=c(length(nval_name), nreps,length(pArrList)), dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))

# pred2 <- array(NA, dim=c(2, preDays, nparsets), dimnames=list(c("m2","m3"),seq(preDays), seq(nparsets)) ) # print(pred2)
# par_names <- c("storm_fate", "num_nests", "flsurv_given", "MCtype", "propMC", "propUnk",
               # "storm_dur", "storm_freq", "obs_int", "hatch_time", 
# )

# vnames <- c("trueDSR","discDSR","anDSR","lexpDSR", "diff1","diff2","diff3")
## for summary at end of test:
# vnames <- c("trueDSR","lexpDSR","mcmcDSR","markDSR","mayfDSR", "diff_lexp","diff_mcmc","diff_mark","diff_mayf")
# vnames <- c("true","lexp","lexp_top","mcmc","mark","mark_top","mayf")
# vnames <- c("true","app","lexp","mcmc","mark","mark_top","mayf")
if(config$mcmcOld){
  vnames <- c("true","app","lexp","mcmc","mcmc_old","mayf")
} else {
  # vnames <- c("true","app","lexp","mcmc","mayf")
  # vnames <- c("true","true2","app","lexp","lexp-pr","mcmc","mayf")
  # vnames <- c("true","true2","app","lexp","mcmc","mayf")
  vnames <- c("trueDate","lexp","mcmc","mayf","trueNull","app")
  # vnames <- c("mayf","trueNull","app")
}
# if(config$logex) vnames <- c("lexp",vnames)
# if(config$mcmc) vnames <- c("mcmc",vnames)
# vnames <- c("trueDate",vnames)
# vnames2 <- paste0("dsr_",vnames)
vnames2 <- paste0("psr_",vnames)
diffnames <- paste0("diff_",vnames[-1])
allnames <- c("number_storms","storm_mortality","obs_interval","discovery_probability","evidence_decay_rate",
              # "dsr_given","number_discovered","number_excluded",vnames2,diffnames)
              # "dsr_given","dsr_true_d","survey_days","total_number_hatched", "total_number_flooded",
              "dsr_given","dsr_true_d","total_number_hatched", "total_number_flooded",
              "number_discovered","number_excluded", "proportion_excluded","proportion_misclassified",
              vnames2,diffnames)
# print(allnames)
# valMat <- array(NA, dim=c(length(vnames), nreps,nparsets), dimnames=list(vnames,seq(nreps), seq(nparsets)))
valMat <- array(NA, dim=c(length(allnames), nreps,nparsets), dimnames=list(allnames,seq(nreps), seq(nparsets)))
# if(debug>=3) print(valMat)
psrPlot <- array(NA, dim=c(nreps, preDays))
psrPlot_true <- array(NA, dim=c(nreps, preDays))

stormDates <- c()
obsLength <- c()
obsIntList <- c()

# trueDSRmat <- array(NA, dim=c(150,nreps,nparsets))

# save propInit in same matrix? should be same length
trueDSRmat <- array(NA, dim=c(2,preDays,nreps,nparsets))
# propInitmat <- array(NA, dim=c(preDays,nreps,nparsets))
# dateDSRmat <- array(NA, dim=c(preDays,nreps,nparsets))
modDSRmat <- array(NA, dim=c(nmod+1,preDays,nreps,nparsets))
# print(dim(modDSRmat))
# print(dim(trueDSRmat))
ncol <- length(colnames)
cat("ncol=", ncol)
# nDataMat <- array(NA, dim=c(250,length(colnames), nreps, nparsets)) 
nestDataMat <- array(NA, dim=c(100,length(colnames), nreps, nparsets), dimnames=list(seq(100), colnames, seq(nreps), seq(nparsets))) 
cat("\nnest data matrix dimensions:")
print(dim(nestDataMat))
fateMat <- array(NA, dim=c(250,nreps,nparsets), dimnames=list(seq(250), seq(nreps), seq(nparsets)))
a_fateMat <- array(NA, dim=c(250,nreps,nparsets), dimnames=list(seq(250), seq(nreps), seq(nparsets)))

Sys.setenv(printset=FALSE) 

