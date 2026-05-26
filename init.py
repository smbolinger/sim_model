
import numpy as np
import os
from datetime import datetime
from helpers import print

scriptName = os.environ.get('script_name')
dtime = datetime.today().strftime('%d %b %Y @ %H:%M')
# atype = os.environ.get('atypeR')
# nWeeks = 2
# initFromFile = True
# stormFromFile = True

# print("\n\n<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
print("\n\n+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
print(" + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
# print(f"\n <> <> <> <> <> <> <> <> datsim.py - {dtime} <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>")
print(f"\n <> <> <> <> <> <> <> <> {scriptName} - {dtime} <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>\n")
print("+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ")
print(" + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + \n")

# config = choose_config(atype)
# pLists = choose_parlist(atype, config)
# odir  = mk_outdir(now_short, con=config)
# paramsArray = mk_param_list_list(parL=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False)
# pArrList = mk_param_list_list(parL=pLists, fdir=odir, suf=f"{config.rngSeed}{atype}", debug=False, listRet=True)
# if config.debug>=2:
#   print("|>|>param sets:")
#   print(pArrList)
# rng = np.random.default_rng(seed=config.rngSeed)
# print_settings(config,atype,initFromFile,paramsArray)
