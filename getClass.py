#!/usr/local/bin/python

import numpy as np
from dataclasses import dataclass
import statsmodels.api as sm
# from statsmodels.genmod.families import links
from statsmodels.genmod.families.links import Logit,Link

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Params: # most importantly, Pylance recognizes the attributes, unlike 
              # dict keys
    stormFate:bool
    # stormFate: int
    stormUnk:  bool
    # stormFail:bool
    # stormUnk:bool
    pMortFl:  np.float32
    MCtype:   str
    # MCtype:  int 
    propMC:   np.float32
    propUnk:  np.float32
    stormFrq: int
    stormDur: int
    decayRate:np.float32
    obsFreq:  int
    hatchTime:int
    numNests: int
    probSurv: np.float32
    discProb: np.float32
    brDays:   int
    # whichLike:int
    # SprobSurv:np.float32
    # fateCues: np.float32
    # useSMat:  bool
    # pWrong:   np.float32 
    # wType:    int   # type of incorrect fate value: 0, 2, 7

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class StrConfig: 
    likeDir:     str
    stormInit:   str

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Config: 
    """
    use different debug var bc these will print for every time optimizer runs
    """
    # rng:         Generator
    # args:        list[str]
    hTime:      int
    mark:       bool
    mcmc:       bool
    mcmcOld:    bool
    logex:      bool
    rngSeed:      int
    optimizer:   str
    optimFunc:   str
    optimGlob:   bool ## global optimizer?
    nreps:       int
    stormFate:   int
    # sFateType:   int
    numNests:   int
    mcType:     int
    rangeVar:    int
    saveNData:   bool
    testing:     str
    predict:    bool
    mayfStart:    bool
    # debug:       bool
    # debugLL:     bool
    # debugNests:  bool
    # debugFlood:  bool
    # debugObs:    bool
    # debugM:      bool
    # debugSummary: bool
    startParID:  int
    debug:       int
    debugLL:     int
    debugNests:  int
    debugFlood:  int
    debugObs:    int
    debugM:      int
    debugSummary: int
    debugLogEx: int
    debugDSR: int

    predSave:   str
    coefSave:   str
    obsSave:    bool
    plotPred:   bool
    # useWSL:      bool
    # likeDir:     str
    # stormInit:   str
    numOut:      int
    fnUnique:    bool
    useWin:      bool
    other:       str
    msg:         str
    # testing:     bool
    # likeFile:    str
    # colNames:    str

# NOTE still need to add exposure as an offset, I guess. 
# NOTE    --> get type-checking error when trying to make inverse link
# class ExpLink(links.Link):
# class ExpLink(Link):
class ExpLink(Logit):
  # from Shaffer: a = 1/expos; 
  #               fwdlink link = log((_mean_**a)/(1-_mean_$*a)); 
  #               invlink ilink = (exp(_xbeta_)/(l+exp(_xbeta_  )))**expos;

  # can't pass exposure as a list
  #can't easily do predictions if passed as array unless you have 
  # the same number of values to predict to as values in array

  #can it be a function based on the dates?

  def __init__(self, exposure):
    self.exposure=exposure
  # def ___call___(self, a, mu):

  # def __call__(self, mu):
  def __call__(self, p):
    # return np.log((mu**a)/(1-mu**a))
    return np.log(p**(1.0/self.exposure)/(1.0-p**(1.0/self.exposure)))
  # def inverse(self, eta):

  # def inverse(self, lin_pred):
  def inverse(self, z):
    # return 1.0/(1.0 + np.exp(eta))**self.exposure
    return 1.0/(1.0 + np.exp(z))**self.exposure

  def derivative(self, p):
    # Derivative is complex, often handled by numerical approximation
    # in custom links. Simplified approach:
    return 1.0 / (p * (1.0 - p) * (1.0/self.exposure))
#   def inverse(self, xbeta, expos):
#     return (np.exp(xbeta) / (1 + np.exp(xbeta))) ** expos
## Usage within GLM
## exposure_time should be a numpy array of interval lengths
## endog (y) is binary (0/1)
## exog (X) includes intercept and covariates
## code:
  #> link = LogitExposure(exposure=exposure_time)
  #> model = sm.GLM(endog_y, exog_X, family=sm.families.Binomial(link=link))
  #> results = model.fit()
# this is all from an LLM


