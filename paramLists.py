
import numpy as np

# NOTE make sure params are in same order!!! makes loading for analysis much easier!

staticPar = {
            'discProb': 0.8,
            'probSurv' : 0.98,
            'brDays': 180,
            # 'decayRate': 0.12,
            'decayRate': 0.1,
            # 'decayRate': 0.08,
            'stormUnk': False
            # 'useSMat': False #not used
             }

parLists = {
            'stormFate': [False,True],
            # 'stormFate': [False],
            # 'stormFate': [True],
            # 'numNests' : [ 250,500],
            # 'numNests' : [ 300],
            'numNests' : [ 250],
            # 'pMortFl'  : [0.9, 0.75, 0.6], # flood/storm severity
            'pMortFl'  : [0.9, 0.7, 0.5], # flood/storm severity6
            'MCtype'   : ["none"],
            'propMC'    : [0.0],
            'propUnk'   : [0.0],
            'stormDur' : [1, 2],
            'stormFrq' : [1,2,3,4],
            'obsFreq'  : [3, 5, 7],
            'hatchTime': [16, 20, 28]
            } 

plTest  = {
           # 'stormFate': [False],
           'stormFate': [True,False],
           # 'stormFate': [True],
           # 'numNests'  : [250],
           'numNests'  : [100],
           # 'numNests'  : [500],
           # 'numNests'  : [250,500],
           # 'pMortFl'   : [0.6, 0.9],
           'pMortFl'   : [0.75],
           'MCtype'   : ["none"],
           'propMC'    : [0.0],
           'propUnk'   : [0.0],
           'stormDur'  : [1],
           # 'stormDur'  : [2],
           # 'stormFrq'  : [1,2,3,4,5],
           # 'stormFrq'  : [1,2,3,4],
           # 'stormFrq'  : [4,3,2,1],
           # 'stormFrq'  : [0],
           # 'stormFrq'  : [0,2,4],
           # 'stormFrq'  : [0,2],
           'stormFrq'  : [2],
           # 'stormFrq'  : [2,4],
           # 'stormFrq'  : [4],
           # 'obsFreq'   : [3,7],
           # 'obsFreq'   : [3,5,7],
           'obsFreq'   : [7],
           'hatchTime' : [20],
           # 'obsFreq'   : [1,3],
           # 'stormFate': [False],
            # 'decayRate': [ 0.08],
           # 'hatchTime' : [16,20,28] ,
           # 'hatchTime' : [16,28] ,
           # 'pWrong'    : [0],
           # 'wType'     : [7],
           }
            # 'useSMat'  : [True, False]
            # }

# plTest2 = {'numNests'  : [10,30], ## control values
plControl = {
            'stormFate': [False],
            'probSurv':  [0.966],
            'numNests' : [250,500],
            # 'probSurv' : list(np.linspace(0.88,0.99, 12)), 
            # 'brDays': [180],
            # 'discProb': [0.8],
            # 'decayRate': [0.08],
            # 'decayRate': [0.04],
            'pMortFl'  : [0], # flood/storm severity
            'MCtype'   : ["none"],
            'propMC'    : [0.0],
            'propUnk'   : [0.0],
            'stormDur' : [0],
            'stormFrq' : [0],
            'obsFreq'  : [3,5,7],
            # 'obsFreq'  : [1,3,5],
            'hatchTime': [20],
            # 'hatchTime': [16,20,28],
            # 'decayRate': [ 0.08],
            # 'pWrong':    [0],
            # 'wType': [7] ,
            # 'pWrong':    [0.05, 0.1, 0.2, 0.3],
            }

# plTestRange = {'numNests'  : [100], ## test a wide range of param values
plTestRange = {
               'stormFate': [True,False],
               # 'stormFate': [False],
               # 'stormFate': [True],
               'numNests'  : [250], ## test a wide range of param values
               # 'probSurv'  : [ 0.98],
               # 'brDays': [180],
               # 'discProb': [0.8],
               # # 'discProb': [0.9],
               # # 'decayRate': [0.04],
               # # 'decayRate': [0.08],
               # 'decayRate': [0.1],
               # 'decayRate': [0.12],
               # 'decayRate':  list(np.linspace(0.02,0.2, 19)), 
               'pMortFl'   : [0.75],
               'MCtype'   : ["none"],
               'propMC'    : [0.0],
               'propUnk'   : [0.0],
               'stormDur'  : [2],
               # 'stormFrq'  : list(np.arange(7).astype(int)), 
               # 'stormFrq'  : [0,1,2,3,4,5],
               # 'stormFrq'  : [0,2],
               'stormFrq'  : [2],
               # 'obsFreq'   : [1,2,3,4,5,6,7],
               'obsFreq'   : [5],
               'hatchTime' : [20],
               # 'stormFate': [False],
               # 'stormFate': [True],
               # 'pWrong'    : [0],
               # 'wType'     : [7],
               # 'decayRate': [ 0.08],
    }

plTest2 = {
            'stormFate': [True,False],
            'numNests'  : [500], ## control values
            # 'probSurv' : [0.98],
            # 'brDays': [180],#NOTE if changing brDays, need no storms
            # 'discProb': [1],
            # 'decayRate': [0.0],
            # 'probSurv' : [0.93],
            # 'probSurv' : list(np.arange(0.85,stop=1.00,step=0.1)),
            # 'probSurv' : list(np.linspace(0.85,0.99, 15)), #+> more precise for floats
            'pMortFl'  : [0], # flood/storm severity
            'MCtype'   : ["none"],
            'propMC'    : [0.0],
            'propUnk'   : [0.0],
            'stormDur' : [1],
            'stormFrq' : [2,4],
            # 'obsFreq'  : [1,3],
            # 'obsFreq'  : [3],
            'obsFreq'  : [3,5,7],
            'hatchTime': [20],
            # 'decayRate': [0.08],
            # 'pWrong':    [0.05, 0.1, 0.2, 0.3],
            # 'pWrong':    [0.1, 0.2, 0.3, 0.4],
            # 'pWrong': [0],
            # 'wType': [7],
           }
plCtlTest = {
            'stormFate': [False],
            'probSurv':  [0.966],
            'numNests'  : [250], ## control values
            # 'probSurv' : [0.98],
            # 'brDays': [180],#NOTE if changing brDays, need no storms
            # 'discProb': [1],
            # 'decayRate': [0.0],
            # 'probSurv' : [0.93],
            # 'probSurv' : list(np.arange(0.85,stop=1.00,step=0.1)),
            # 'probSurv' : list(np.linspace(0.85,0.99, 15)), #+> more precise for floats
            'pMortFl'  : [0], # flood/storm severity
            'MCtype'   : ["none"],
            'propMC'    : [0.0],
            'propUnk'   : [0.0],
            'stormDur' : [0],
            'stormFrq' : [0],
            # 'obsFreq'  : [1,3],
            'obsFreq'  : [3],
            # 'obsFreq'  : [3,5,7],
            'hatchTime': [20],
            # 'decayRate': [0.08],
            # 'pWrong':    [0.05, 0.1, 0.2, 0.3],
            # 'pWrong':    [0.1, 0.2, 0.3, 0.4],
            # 'pWrong': [0],
            # 'wType': [7],
           }
plNSTest = {
            'decayRate': [0.0], # this should overwrite the static value?
            'probSurv':  [0.966], ## 0.50 of nests should hatch
            'stormFate': [False],
            'numNests' : [250],
            # 'probSurv' : [0.98],
            # 'brDays': [180],
            # 'discProb': [0.8],
            # 'decayRate': [0.0],
            'pMortFl'  : [0], # flood/storm severity
            'MCtype'   : ["hatch2fail","fail2hatch"],
            # 'MCtype'   : ["hatch2fail"],
            # 'MCtype'   : ["hf","unk"],
            # 'propMC'    : [0.05,0.1],
            'propMC'    : [0.0],
            # 'propMC'    : list(np.linspace(0.0,0.4,9)),
            'propUnk'   : [0.0],
            # 'propUnk'   : list(np.linspace(0.0,0.4,9)),
            'stormDur' : [0],
            'stormFrq' : [0],
            'obsFreq'  : [3],
            'hatchTime': [20],
            # 'decayRate': [0.08],
            # 'pWrong':    [0.05, 0.1, 0.2, 0.3],
            # 'pWrong':    [0.0],
            # 'wType': [ 7] ,
            # 'hatchTime': [16, 20, 28],
            }

plNoStorm = {
            'decayRate': [0.0], # this should overwrite the static value?
            'probSurv':  [0.966],
            'stormFate': [False],
            'numNests' : [250,500],
            # 'probSurv' : [0.98],
            # 'brDays': [180],
            # 'discProb': [0.8],
            # 'decayRate': [0.0],
            'pMortFl'  : [0], # flood/storm severity
            # 'MCtype'   : ["none"],
            'MCtype'   : ["hatch2fail","fail2hatch"],
            'propMC'    : [0.0],
            # 'propMC'    : list(np.linspace(0.0,0.5,11)),
            'propUnk'   : [0.0],
            # 'propUnk'   :  list(np.linspace(0.0,0.5,11)),
            'stormDur' : [0],
            'stormFrq' : [0],
            'obsFreq'  : [3],
            'hatchTime': [20],
            # 'pWrong':    [0.05, 0.1, 0.2, 0.3],
            # 'pWrong':    [0],
            # 'wType': [7] ,
            # 'hatchTime': [16, 20, 28],
            }

# parLists2 = {'numNests' : [250, 500],
plTestFlood  = {'numNests'  : [100],
# plTest  = {'numNests'  : [30],
             #   'probSurv'  : [0.96],
             #  'brDays': [180],
             # 'discProb': [0.8],
             # 'decayRate': [0.08],
           'pMortFl'   : [0.9, 0.6],
            'MCtype'   : ["none"],
           'propMC'    : [0.0],
           'propUnk'   : [0.0],
        #    'stormDur'  : [1, 3],
        #    'stormFrq'  : [1, 3],
        #    'obsFreq'   : [3, 5],
           'stormDur'  : [2],
           'stormFrq'  : [1, 4],
           'obsFreq'   : [3, 7],
           'stormFate': [False,True],
           # 'pWrong'    : [0],
           # 'wType'     : [7],
           'hatchTime' : [16, 28],
        #    'hatchTime' : [20, 28] }
            # 'useSMat'  : [True, False]
            }

plDefault = {
             'stormFate': [False],
            'numNests'  : [50],
            # plTest  = {'numNests'  : [30],
            # 'probSurv':  [0.96],
            # 'brDays': [180],
            # 'discProb': [0.8],
            # 'decayRate': [0.08],
            'pMortFl':   [0.75],
            'MCtype'   : ["none"],
            'propMC'    : [0.0],
            'propUnk'   : [0.0],
            'stormDur':  [2],
            'stormFrq'  : [1, 3],
            # 'stormFrq':  [2],
            # 'obsFreq'   : [3, 7],
            'obsFreq':   [3],
            'hatchTime': [20],
            # 'pWrong':    [0],
            # 'wType':     [7] 
            }
            
# plTest  = {'numNests'  : [30],
##+> for testing ndMatrix (small num nests & br days; high disc prob
##+>                         & low decay rate, short hatch time)
plSmall = {
             'stormFate':  [False],
             'numNests':   [100], # 
             'decayRate': [0],
             # 'probSurv':   [0.96],
             #    # 'brDays': [180],
             # 'brDays':     [60],
             # 'discProb':   [0.8],
             # # 'decayRate':  [0.04],
             # 'decayRate': [0.08],
             'pMortFl':    [0.75],
             'MCtype':     ["none"],
             'propMC':     [0.0],
             'propUnk':    [0.0],
             'stormDur':   [1],
             # 'stormFrq':   [1, 3],
             'stormFrq':  [0,1],
             'obsFreq':    [3, 5],
             # 'obsFreq':    [2,4],
             'hatchTime':  [20],
             # 'pWrong':     [0],
             # 'wType':      [7]
             }

plSupp = {
           'stormFate': [False,True],
            'numNests'  : [250,500], # +> for adding supplemental param sets
           # 'probSurv':  [0.91],
           # 'probSurv':  [0.94,0.96],
           #  # 'probSurv' : list(np.linspace(0.88,0.99, 12)), 
           #    'brDays': [180],
           #   'discProb': [0.8],
           #   'decayRate': [0.08],
           'pMortFl':   [0.9,0.75,0.6],
            'MCtype'   : ["none"],
           'propMC'    : [0.0],
           'propUnk'   : [0.0],
           'stormDur':  [1,2],
        #    'stormFrq'  : [1, 3],
           'stormFrq':  [1,2,3,4],
        #    'obsFreq'   : [3, 7],
           'obsFreq':   [3,4,5],
           # 'pWrong':    [0],
           # 'wType':     [-1],
           'hatchTime': [16,20,28],
          }

plSubset = {
           'stormFate': [False,True],
            'numNests' : [ 250],
            # 'probSurv' : [0.94,0.96,0.98],
            # 'probSurv' : [0.98],
            # # 'probSurv' : [0.96],
            #   'brDays': [180],
            #  'discProb': [0.8],
            #  'decayRate': [0.08],
            # 'probSurv' : [0.92, 0.97],
            # 'pMortFl'  : [0.9, 0.75, 0.6], # flood/storm severity
            'pMortFl'  : [0.9, 0.6], # flood/storm severity
            'MCtype'   : ["none"],
           'propMC'    : [0.0],
           'propUnk'   : [0.0],
            # 'pMortFl'  : [0.75], # flood/storm severity
            'stormDur' : [1, 2],
            # 'stormDur' : [3],
            # 'stormDur' : [1,2,3],
            'stormFrq' : [1,4],
            # 'stormFrq' : [1,2,3,4,5],
            # 'stormFrq' : [1,2,3,4],
            # 'obsFreq'  : [3, 5, 7],
            'obsFreq'  : [3],
            # 'obsFreq'  : [2, 4, 6],
            # 'obsFreq'  : [3, 4, 5],
            # 'obsFreq'  : [3,5],
           # 'stormFate': [False],
            # 'pWrong':    [0],
            # # 'wType':     [2,7],
            # 'wType':     [-1],
            'hatchTime': [20]
            # 'hatchTime': [16, 20, 28]
            } 

plDebug = {'numNests'  : [50],
# plTest  = {'numNests'  : [30],
           # 'probSurv':  [0.96],
           #    'brDays': [180],
           #   'discProb': [0.8],
           #   'decayRate': [0.08],
           'pMortFl':   [0.75],
            'MCtype'   : ["none"],
           'propMC'    : [0.0],
           'propUnk'   : [0.0],
           'stormDur':  [1],
        #    'stormFrq'  : [1, 3],
           'stormFrq':  [0],
        #    'obsFreq'   : [3, 7],
           'obsFreq':   [3],
           'stormFate': [False],
           'hatchTime': [16, 28],
           # 'pWrong':    [0.2],
           # 'wType':     [7]
           }
            
#endregion--------------------------------------------------------------------
#   FUNCTIONS
# -----------------------------------------------------------------------------
# Some are very small and specific (e.g. logistic function); others are 
# quite involved.
# -----------------------------------------------------------------------------


