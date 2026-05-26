
import time
from rsettings import config, pLists, rng,atype,now_short,staticPar
from datsim import main

startTime = time.perf_counter()

# print("\n!!! NOT RUNNING MAIN MODEL")
# main(fnUnique=config.fnUnique,rng=rng,atype=atype, nowStr=now_short,parLists=pLists, msg=config.msg, testing=config.testing)
main(atype,rng,config,pLists,pStatic=staticPar,nowStr=now_short)

endTime = time.perf_counter()

elapsed_time = endTime - startTime 

if elapsed_time < 60:
    print(f"Runtime: {elapsed_time:.2f} seconds") #
elif elapsed_time < 3600:
    minutes = elapsed_time / 60
    print(f"Runtime: {minutes:.2f} minutes") #
elif elapsed_time > 3600:
    hours = elapsed_time / 3600
    print(f"Runtime: {hours:.2f} hours") #


