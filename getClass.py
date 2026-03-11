#!/usr/local/bin/python

import numpy as np
from dataclasses import dataclass

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Params: # most importantly, Pylance recognizes the attributes, unlike 
              # dict keys
    numNests: int
    stormDur: int
    stormFrq: int
    obsFreq:  int
    hatchTime:int
    brDays:   int
    whichLike:int
    probSurv: np.float32
    SprobSurv:np.float32
    pMortFl  :np.float32
    discProb: np.float32
    # fateCues: np.float32
    stormFate:bool
    useSMat:  bool
    pWrong:   np.float32 
    wType:    int   # type of incorrect fate value: 0, 2, 7
@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Config: 
    """
    use different debug var bc these will print for every time optimizer runs
    """
    # rng:         Generator
    # args:        list[str]
    nreps:       int
    debug:       bool
    debugLL:     bool
    debugNests:  bool
    debugFlood:  bool
    debugObs:    bool
    debugM:      bool
    debugSummary: bool
    # useWSL:      bool
    useWin:      bool
    # testing:     bool
    testing:     str
    fnUnique:    bool
    # likeFile:    str
    likeDir:     str
    stormInit:   str
    # colNames:    str
    numOut:      int
    rngSeed:      int

