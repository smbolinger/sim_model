
##--------------- CREATE ARRAYS TO STORE DATA: ----------------------------------------------
coef_list <- c("est","se")
mod_names <- c("_null","_date_int","_date_b1", "_age_int","_age_b1",
               "_agedate_int","_agedate_b1","_agedate_b2","_avdate_int","_avdate_b1")
coef_names <- unlist(lapply(coef_list, function(x) paste0(x,mod_names)))
coef_names <- c(coef_names, "avg_expos","avg_age","avg_avdate")

coefsMat <- array(NA,
               dim=c(length(coef_names), nreps, length(pArrList)),
               dimnames=list(coef_names,seq(nreps), seq(length(pArrList))))

if(debug>=0) cat("\n\t\tcoef names:", coef_names)
coefList <- list()

modNames <- list(c("null"),
                 c("int","date"),
                 c("int","age"),
                 c("int","age","date"),
                 c("int","avdate"))

vcovNames <- unlist(sapply(modNames, function(x) outer(x, x, paste, sep="-")))
# vcovNames <- sapply(modNames, function(x) unlist(outer(x, x, paste, sep="-")))
# cat("\n vcov names, length=", length(vcovNames))
# print(vcovNames)

vcovMatMat <- array(NA,
               dim=c(length(vcovNames), nreps, length(pArrList)),
               dimnames=list(vcovNames,seq(nreps), seq(length(pArrList))))

# print(dimnames(vcovMatMat))


# cat("\n initPropMat, dimnames:")
initPropMat <- array(NA,
               dim=c(vals$preDays, nreps, length(pArrList)),
               dimnames=list(prDays,seq(nreps), seq(length(pArrList))))

# print(dimnames(initPropMat))

mayfdsr_name <- c("mayfDSR","mayfPSR","mayfVar","mayfSE","simplePSR")
# truedsr_name <- c("tDSR","tPSR","tDSR_date","tPSR_date","tDSR_dateage","tPSR_dateage")
truedsr_name <- c("tDSR","tPSR","tDSR_date","tPSR_date")
mark_name <- c()
lexp_name <- c()
lexp_supp <- c()
mcmc_name <- c()
if(config$mark) mark_name <- c("markDSR","markPSR","markDSRdate","markPSRdate","markDSRdsAge","markPSRdsAge","marktopDSR","marktopmod")

if(config$logex){
  lexp_dsr <- c("leDSR1","leDSR2","leDSR3","leDSR4","leDSR5","lePSR1","lePSR2","lePSR3","lePSR4","lePSR5")
  lexp_se <- c("seDSR1","seDSR2","seDSR3","seDSR4","seDSR5")
  # lexp_cov <- c("covDSR1","covDSR2","covDSR3","covDSR4","covDSR5")
  # lexp_name <- c(lexp_dsr, lexp_se. lexp_cov)
  lexp_name <- c(lexp_dsr, lexp_se)
# lexp_supp <- c("leDSRava","lePSRava","leDSRavd","lePSRavd","leDSRavad","lePSRavad")
  lexp_supp <- c("leDSRavd","lePSRavd","leDSRava","lePSRava")
# truedsr_name <- c("tDSR","tPSR","tDSRdate","tPSRdate","tDSRage","tPSRage")
# } else {
}
# if(config$survSave!="all") lexp_name <- c()
if(!config$predict) lexp_name <- c()

if(config$mcmc){
  mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcDSR_se","mcmcDPR_se")
  # mcmc_name <- c("mcmcDSR","mcmcPSR","mcmcDPR","mcmcDSR_se","mcmcDPR_se","mcmcDSR_cov")
# } else {
}

dsr_name  <- c(truedsr_name,lexp_name, mcmc_name,mayfdsr_name, mark_name)

dsrMat <- array(NA,
                dim=c(length(dsr_name), nreps,length(pArrList)),
                dimnames=list(dsr_name,seq(nreps), seq(length(pArrList))))
cat("\n\t\tDSR matrix dim:", dim(dsrMat))
cat("\t\t& DSR val names:", dsr_name)

nval_name <- c("parID","repID","fld", "hat","fld_dsc","hat_dsc","fld_an",
               "hat_an", "sNest", "dsc", "excl","unk","mc","mc2", "avfint",
               "avk","maxi","lint","aDSR","aPSR","mfDSR","appDSR")
cat("\n\t\tnVal names:", nval_name)

nValMat <- array(NA,
                 dim=c(length(nval_name), nreps,length(pArrList)),
                 dimnames=list(nval_name,seq(nreps), seq(length(pArrList))))

if(config$mcmcOld){
  vnames <- c("true","app","lexp","mcmc","mcmc_old","mayf")
} else {
  vnames <- c("trueDate","lexp","mcmc","mayf","trueNull","app")
}
vnames2 <- paste0("psr_",vnames)
diffnames <- paste0("diff_",vnames[-1])
allnames <- c("number_storms","storm_mortality","obs_interval","discovery_probability","evidence_decay_rate",
              # "dsr_given","number_discovered","number_excluded",vnames2,diffnames)
              # "dsr_given","dsr_true_d","survey_days","total_number_hatched", "total_number_flooded",
              "dsr_given","dsr_true_d","total_number_hatched", "total_number_flooded",
              "number_discovered","number_excluded", "proportion_excluded","proportion_misclassified",
              vnames2,diffnames)

valMat <- array(NA, dim=c(length(allnames), nreps,nparsets), dimnames=list(allnames,seq(nreps), seq(nparsets)))
# if(debug>=3) print(valMat)
psrPlot <- array(NA, dim=c(nreps, vals$preDays))
psrPlot_true <- array(NA, dim=c(nreps, vals$preDays))

stormDates <- c()
obsLength <- c()
obsIntList <- c()

# trueDSRmat <- array(NA, dim=c(5,vals$preDays*28,nreps,nparsets),
trueDSRmat <- array(NA, dim=c(nmod_true+1,vals$preDays*28,nreps,nparsets),
                    # dimnames=list( c("date","age","true", "true.date", "true.dateage"),seq(vals$preDays*28), seq(nreps), seq(nparsets))
                    dimnames=list( c("date","true", "true.date"),seq(vals$preDays*28), seq(nreps), seq(nparsets))
)
# cat("\ntrueDSRmat dimnames:") 
# qvcalc::indentPrint(dimnames(trueDSRmat))
cat("\n\t\t>> max number of rows for trueDSR:", vals$preDays*c(16,20,28))
cat("\t>> predict for specific dates/ages instead?")
# propInitmat <- array(NA, dim=c(preDays,nreps,nparsets))
# dateDSRmat <- array(NA, dim=c(preDays,nreps,nparsets))
modDSRmat <- array(NA, dim=c(nmod+1,vals$preDays,nreps,nparsets))
# print(dim(modDSRmat))
# print(dim(trueDSRmat))
ncol <- length(colnames)
cat("\tncol=", ncol)
# nDataMat <- array(NA, dim=c(250,length(colnames), nreps, nparsets)) 
nestDataMat <- array(NA, dim=c(100,length(colnames), nreps, nparsets), dimnames=list(seq(100), colnames, seq(nreps), seq(nparsets))) 
cat("\n\t\tnest data matrix dimensions:")
qvcalc::indentPrint(dim(nestDataMat))
fateMat <- array(NA, dim=c(250,nreps,nparsets), dimnames=list(seq(250), seq(nreps), seq(nparsets)))
a_fateMat <- array(NA, dim=c(250,nreps,nparsets), dimnames=list(seq(250), seq(nreps), seq(nparsets)))
