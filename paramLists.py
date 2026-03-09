
staticPar = {'brDays': 180,
             'SprobSurv': 0.2, # never actually used
             'discProb': 0.7,
             'whichLike': 1,
            #  'stormFate': True,
             'useSMat': False }

parLists = {'numNests' : [250, 500],
            # 'probSurv' : [0.95, 0.97],
            'probSurv' : [0.96],
            'pMortFl'  : [0.9, 0.75, 0.6], # flood/storm severity
            'stormDur' : [1, 2],
            'stormFrq' : [1, 2, 3, 4],
            'obsFreq'  : [3, 5, 7],
           'stormFate': [False,True],
            'hatchTime': [16, 20, 28], 
            'pWrong':    [0],
            'wType':     [-1] }

parLists2 = {'numNests' : [250, 500],
            'probSurv' : [0.96],
            'pMortFl'  : [0], # flood/storm severity
            'stormDur' : [0],
            'stormFrq' : [0],
            'obsFreq'  : [3],
           'stormFate': [False],
            'hatchTime': [16, 20, 28],
            'pWrong':    [0.05, 0.1, 0.2, 0.3],
            'wType': [2, 7] }

plTest  = {'numNests'  : [100],
# plTest  = {'numNests'  : [50],
           'probSurv'  : [0.96],
           'pMortFl'   : [0.75],
        #    'stormDur'  : [1],
           'stormDur'  : [2],
        #    'stormFrq'  : [2],
           'stormFrq'  : [1,2],
        #    'obsFreq'   : [3],
           'obsFreq'   : [3, 7],
           'pWrong'    : [0.1],
           'wType'     : [7],
           'stormFate': [False,True],
           'hatchTime' : [20, 28] 
           }
        #    'hatchTime' : [20],
            # 'useSMat'  : [True, False]
            # }

plTest2 = {'numNests'  : [100],
            'probSurv' : [0.96],
            'pMortFl'  : [0], # flood/storm severity
            'stormDur' : [0],
            'stormFrq' : [0],
            'obsFreq'  : [3],
            'stormFate': [False],
            'hatchTime': [16, 20, 28],
            # 'pWrong':    [0.05, 0.1, 0.2, 0.3],
            'pWrong':    [0.1, 0.2, 0.3, 0.4],
            'wType': [2, 7] }

plTestFlood  = {'numNests'  : [100],
# plTest  = {'numNests'  : [30],
               'probSurv'  : [0.96],
           'pMortFl'   : [0.9, 0.6],
        #    'stormDur'  : [1, 3],
           'stormDur'  : [2],
        #    'stormFrq'  : [1, 3],
           'stormFrq'  : [1, 4],
        #    'obsFreq'   : [3, 5],
           'obsFreq'   : [3, 7],
           'stormFate': [False,True],
        #    'hatchTime' : [20, 28] }
           'hatchTime' : [16, 28],
            # 'useSMat'  : [True, False]
            }

plDebug = {'numNests'  : [50],
# plTest  = {'numNests'  : [30],
           'probSurv':  [0.96],
           'pMortFl':   [0.75],
           'stormDur':  [1],
        #    'stormFrq'  : [1, 3],
           'stormFrq':  [0],
        #    'obsFreq'   : [3, 7],
           'obsFreq':   [3],
           'stormFate': [False],
           'hatchTime': [16, 28],
           'pWrong':    [0.2],
           'wType':     [7] }
            
#endregion--------------------------------------------------------------------
#   FUNCTIONS
# -----------------------------------------------------------------------------
# Some are very small and specific (e.g. logistic function); others are 
# quite involved.
# -----------------------------------------------------------------------------
