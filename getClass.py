#!/usr/local/bin/python

import numpy as np
from dataclasses import dataclass

@dataclass # type-secure (can't accidentally pass wrong type) & can be immutable
class Params: # most importantly, Pylance recognizes the attributes, unlike 
              # dict keys
    numNests: int
    probSurv: np.float32
    pMortFl  :np.float32
    stormFrq: int
    stormDur: int
    obsFreq:  int
    discProb: np.float32
    hatchTime:int
    stormFate:bool
    brDays:   int
    whichLike:int
    decayRate:np.float32
    SprobSurv:np.float32
    # fateCues: np.float32
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
    rngSeed:      int
    optimizer:   str
    nreps:       int
    stormFate:   int
    saveNData:   bool
    testing:     str
    mayfStart:    bool
    debug:       bool
    debugLL:     bool
    debugNests:  bool
    debugFlood:  bool
    debugObs:    bool
    debugM:      bool
    debugSummary: bool
    # useWSL:      bool
    likeDir:     str
    stormInit:   str
    numOut:      int
    fnUnique:    bool
    useWin:      bool
    msg:         str
    # testing:     bool
    # likeFile:    str
    # colNames:    str

