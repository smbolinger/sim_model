
# preceding line is just above each if statement, in a comment
# all the indentations should match what they were, too

# +> OBSERVER.PY -------------------------------------------------------------
# +> mk_surveys:

  #*# storms      = np.split(stormDays, splits)
  if conf.debug>=2:
    print(f"\t\t-> {stormDays=}")
    print("\t\t>-->survey days before alteration:")
    arrPrint(surveyDays)
    # d = {ind: v for ind,v in enumerate(surveyDays)}
    # print(f"{d}")
    print(f"\t\t{storms=}")
  # +> inside for loop:

    #*# sDiff   =  surveyDays[stormPos] - lastDay
    if conf.debug>=2:
      print(f"\t|>{s=}", end=" ")
      print(f"\t|>{surveyDays[stormPos]=}", end=" ")
      # print(f"\t|>{(lastDay)=} ; {sDiff=}", end=" ")
      print(f"\t|> add {int(sDiff)} to survey days >= {(int(lastDay))} ")
      # print(f"\t|> ")

  #*# surveyDays  = surveyDays[np.isin(surveyDays, stormDays) == False]
  if conf.debug>=2:
    print("\n\t\t>-->survey days after alteration:") # NOTE: need \n bc of prev
    arrPrint(surveyDays)


  #*# surveyInts  = np.array([0] + [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
  if conf.debug: 
    print(f"\t\t>-> all survey days, minus storms (len {len(surveyDays)}):")
    # indPrint(surveyDays) 
    arrPrint(surveyDays) 

# -----------------------------------------------------------------------------
# +> mk_per:

  #*# nestPeriod = np.transpose(nestPeriod) # +> an array of start,end pairs 
  if con.debugNests>=5:
    print( f"\t\t\t>> start & end of nest period:\n")
    arrPrint(nestPeriod)

# -----------------------------------------------------------------------------
# +> assign_fate:

  # fateCuesPresent   = expDecay(n0=1, k=0.1, t=intFinal)
  # if cn.debugObs>=3: 
  #   timevals = np.arange(0,10)
  #   probvals = expDecay(n0=1, k=0.1, t=timevals)

    # print("\n\trange of prob vals for plot:",np.min(probvals), np.max(probvals))
    # fig = plt.plot(probvals,timevals) # maybe can't assign a plotext plot to an object?
    # plt.clt()
    # plt.clear_color()
    # plt.clf()
    # plt.plot(probvals, timevals)
    # # fig.savefig("figs/expDecay.png")
    # plt.show() # syntax for plotext is almost identical to matplotlib syntax
    #
  if cn.debugObs>=4:
    print("\t\t\t|> probability of fate cues:")
    arrPrint(np.round(fateCuesPresent,3))
  if cn.debugObs>=5:
    print("\t\t\t|>& final int (for comparison):")
    arrPrint(intFinal)

  # fateProb = rng.uniform(low=0, high=1, size=numNests)
  if cn.debugObs >=4:
    print("\t\t\t|>random probs for fate:")
    arrPrint(np.round(fateProb,3))

  # assignedFate[fateProb < fateCuesPresent] = trueFate[fateProb < fateCuesPresent] 
  if cn.debugObs>=2:
    print("\t\t\t>-> true fates (all nests, not just discovered):")
    arrPrint(trueFate)
    print("\t\t\t>-> assigned fates before (all nests, not just discovered):")
    arrPrint(assignedFate)

  # if stormFate: assignedFate[intFinal > obsFr] = 2
  if cn.debugObs>=2:
    print(
        f"\t\t>=>=> true fate counts (proportion):"
        f"\tH:{sum(trueFate==0)}({sum(assignedFate==0)/numNests})|"
        f"D:{sum(trueFate==1)}({sum(assignedFate==1)/numNests})|"
        f"F:{sum(trueFate==2)}({sum(assignedFate==2)/numNests})"
        )
    print("\t\t\t>-> assigned fates after incorrect fates assigned(all nests):")
    arrPrint(assignedFate)
    
  if cn.debugObs>=4: 
    print("\t\t\t>-> compare random probs to fateCuesPresent:\n") 
    print("\t {[np.round(fateProb[f],3):np.round(fateCuesPresent[f],3) for f in 1:numNests]}")

    print(f"\t>-> or to pWrong: {pWrong} with fill value: {assignVal}")
  if cn.debugFlood>=4:
    print("\t\t\t>-> nests with storm in final interval:", np.where(intFinal>obsFr))
    print("\t\t\t>-> storm fate == True?", stormFate)

# -----------------------------------------------------------------------------
# +> svy_position:

  # position = np.searchsorted(surveyDays, initiation) 
  if cn.debugObs>=5:
    print("\t\t>> initiation dates:\n")
    arrPrint(initiation)
    print("\t\t>>>> position of initiation date in survey day list:\n") 
    arrPrint( position)
    print("\t\t>> end dates:\n")
    arrPrint(nestEnd)

  # surveyDays = dict(zip(np.arange(len(surveyDays)), surveyDays))
  if cn.debugObs>=5:
    print("\t\t>>>> position of end date in survey day list:\n", position2, len(position2)) 
    print("\t\t>> survey days with index number:\n", surveyDays)

# -----------------------------------------------------------------------------
# +> observer:

  #at beginning of function
  if conf.debugObs>=1: print("\n\t\t[*] [*] [*] [*] [*] observer [*] [*] [*] [*] [*] [*] [*] [*] ")
  if conf.debugObs>=5:
    print("nest data!")
    arrPrint(nData)

  # num_svy      = pos[1] - pos[0]   
  if conf.debugObs>=4:
    print("\t|> num surveys for each nest:")
    arrPrint(num_svy)

  # discovered     = svysTilDiscovery < num_svy
  if conf.debugObs>=3:
    print("\t\t\t|> nest discovered? (svysTilDiscovery < num_svy)")
    arrPrint(discovered)

  # num_svy[~discovered] = 0
  if conf.debugObs>=4:
    print("\t\t\t|> surveys til discovery:")
    arrPrint(svysTilDiscovery)
    print("\t\t\t> total surveys while nest active:")
    arrPrint(num_svy)

  # out[:,5] = intFinal.astype(int) # length of final interval - transform to integer for the ndarray
  if conf.debugObs>=3:
    trueFate = nData[:,3]
    assignedFate = out[:,3][discovered]
    print(
        f"\t\t=>=> assigned fates - discovered (proportion):"
        f"\tH:{sum(assignedFate==0)}({sum(assignedFate==0)/sum(discovered)})|"
        f"D:{sum(assignedFate==1)}({sum(assignedFate==1)/sum(discovered)})|"
        f"Fl:{sum(assignedFate==2)}({sum(assignedFate==2)/sum(discovered)})"
        )
    print("\t\t\t>-> true fates (discovered nests):")
    arrPrint(trueFate[discovered])
    print("\t\t\t>-> assigned fates (discovered nests):")
    arrPrint(assignedFate)
  if conf.debugObs>=1: 
    print("\t\t>-> number discovered: ",
          np.sum(discovered==True),
          "\t\t>-> disc prob: ",
          par.discProb,
          )
    print("\t\t\t>-> assigned fates:")
    arrPrint(assignedFate)
    assHatch = ((out[:,3])==0)[discovered==True]
    nonUnk   = ((out[:,3])!=7)[discovered==True]
    trHatch  = ((nData[:,3]==0))[discovered==True]
    prop = np.sum(assHatch)/(np.sum(nonUnk))
    prop2 = np.sum(trHatch)/(np.sum(discovered==True))
    print(
      f"\t\t>> proportion non-unk nests assigned hatch fate"
      f" ({np.sum(assHatch)} / {np.sum(nonUnk)}):",
      np.round(prop,5),
      # np.sum(((out[:,3])==0)[discovered==True])/(sum(discovered==True)),
      # np.sum(((out[:,3])==0)[discovered==True])/(np.sum((out[:,3]!=7)[discovered==True])),
      "vs. period survival:",
      np.round(par.probSurv**par.hatchTime,5),
      f"vs. proportion of all nests: {prop2}"
      )
  if conf.debugObs==3 or conf.debugObs==2: 
    # trueFate = round(nData[:,3])
    trueFate = nData[:,3]
    assignedFate = out[:,3][discovered]
    print(
        "\t\t|>surveys til discovery; discovered T/F; total obs days;"
        " total active days; assigned fate; true fate:")
    active = nData[:,2]-nData[:,1]
    obsLen = (out[:,2]-out[:,0])
    for i in range(5):
      # print(f"\t\t\t{i:02}: {svysTilDiscovery[i]} | {discovered[i]} | {(out[:,2]-out[:,0])[i]}:03 | {(nData[:,2]-nData[:,1])[i]}")
      print(
          f"\t\t\t{i:03} : {svysTilDiscovery[i]:02} | {discovered[i]:>5} | {(obsLen[i]):>4} |"
          # f" {((nData[:,2]-nData[:,1])[i]):>4} | {assignedFate[i]} | {trueFate[i]}"
          f" {round(active[i]):>4} | {assignedFate[i]} | {round(trueFate[i])}"
          )
    for i in range(-5,0):
      print(
          f"\t\t\t{i:03} : {svysTilDiscovery[i]:02} | {discovered[i]:>5} | {(obsLen[i]):>4} |"
          f" {round(active[i]):>4} | {assignedFate[i]} | {round(trueFate[i])}"
          )
  if conf.debugObs>=3: 
    trueFate = nData[:,3]
    assignedFate = out[:,3]
    active = nData[:,2]-nData[:,1]
    obsLen = (out[:,2]-out[:,0])
    print(
        f"\t\t =>=> assigned fates - total (proportion):"
        f"\tH:{sum(assignedFate==0)}({sum(assignedFate==0)/par.numNests})|"
        f"D:{sum(assignedFate==1)}({sum(assignedFate==1)/par.numNests})|"
        f"F:{sum(assignedFate==2)}({sum(assignedFate==2)/par.numNests})"
        )
  if conf.debugObs>=4:
    print(
        "\t\t|>surveys til discovery; discovered T/F; total obs days;"
        " total active days; assigned fate; true fate:")
    for i in range(len(out)):
      print(
          f"\t\t\t{i:03} : {svysTilDiscovery[i]:02} | {discovered[i]:>2} | {(obsLen[i]):>4} |"
          f" {round(active[i]):>4} | {assignedFate[i]} | {round(trueFate[i])}"
          )



# -----------------------------------------------------------------------------
# +> make_obs:

  # hatched    = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
  if conf.debugNests>=4: print("\t\t|>hatched (before storms)=", hatched, sum(hatched))

  # end of function
  ndString = "\t\t\tID--init-end-fate---i---j---k---afate-nstm-fInt"
  if conf.debugSummary>=2: print(f"\nnestData:\n{ndString}\n", nestData[0:5,:], "\n. . . . . . \n", nestData[-5:,:])
  if conf.debugSummary>=3: print(f"\nnestData:\n{ndString}\n", nestData)


# +> MAKE_NESTS.PY -------------------------------------------------------------

# +> mk_surv:

  # survival[survival > hatchTime] = hatchTime # add some amt of error?
  if con.debugNests>=3:
    print("\t\t\t>> survival in days:\n") 
    arrPrint(survival)
  # hatched = survival >= hatchTime # the hatched nests survived for >= hatchTime days 
  # if con.debugNests: print("hatched (no storms):", hatched, hatched.sum())
  ## NOTE THIS IS NOT THE TRUE HATCHED NUMBER; DOESN'T TAKE STORMS INTO ACCOUNT

# -----------------------------------------------------------------------------
# +> mk_nests:

  # at beginning of function:
  print("\n\t\t[*] [*] [*] [*] [*] making nests [*] [*] [*] [*] [*] [*] [*] [*] ")

  # if conf.debug: print(">> end dates:\n", nestEnd, len(nestEnd)) 

  # nestData[:,2] = nestData[:,1] +survival
  if conf.debugNests==2:
    print("\n\t\t\t>> ID, init, & end:\n")
    arrPrint(nestData[0:5,:])
    print("\n\t\t\t\t. . . . . .\n")
    arrPrint(nestData[-5:,:])
  if conf.debugNests>=4:
    print("\n\t\t\t>> ID, init, & end:\n")
    arrPrint(nestData)





# -----------------------------------------------------------------------------
# +> mk_flood:

  # flP = rng.uniform(low=0, high=1, size=sum(numStorms)) 
  if config.debugFlood>=2:
    print("\t\t\t|>prob of flooding:")
    arrPrint(pMortFl)
  if config.debugFlood>=4:
    print("\t\t\t|>random probabilities, one per storm:")
    arrPrint(flP)

    # if numStorms[n] > 0:
    #   flood = np.zeros(len(stormDays), dtype=np.int32)
      if config.debugFlood>=3:
        print(f"\n\t\t\tnest {n} experienced >=1 storm", end=" ")

          # x=x+1 ##
          if config.debugFlood>=3:
            print(f"\t\tnest flooded day {s}?", flood[s], end=" ")

  # stormInfo[:,2] = flooded # true number flooded
  if con.debugFlood>=1: 
    print("\t\t\t|>prob of failure due to flooding:", pMortFl, end="")
    print("\t\t\t|>storm nests:", sum(stormNest), end=" ")
    if con.debugFlood>=3: print( stormNest[:10])
    if con.debugFlood>=3: print( stormNest[-10:])
    print("\t\t\t|>flooded & storm:", flooded.sum())
    if con.debugFlood>=3: print(flooded[:10])
    if con.debugFlood>=3: print(flooded[-10:])
    if con.debugFlood>=1: print(f"\n\t\t\tID, num storms, which, fl, stormIndex 1-5:")
    ## +> enumerate adds a counter to an iterable; can get index & value
    if con.debugFlood>=1 & con.debugFlood<3:
      for i,row in enumerate(np.concatenate((stormInfo[:5],stormIndex[:5]),axis=1)): 
        print(f"\t\t\t\t{i}: {row}")
      print("\t\t\t\t\t . . . . . . . ")
      for i,row in enumerate(np.concatenate((stormInfo[-5:],stormIndex[-5:]),axis=1)): 
        print(f"\t\t\t\t{i}: {row}")
    if con.debugFlood>=3:
      for i,row in enumerate(np.concatenate((stormInfo,stormIndex),axis=1)): 
        print(f"\t\t\t\t{i}: {row}")

# -----------------------------------------------------------------------------

# +> mk_fates:

  # trueFate[flooded == True] = 2  # should override the nests that "hatched" that were actually during storm
  if con.debugNests>=4: print("\t\t\t|>|> end date before storms accounted for:\n", nestDat[:,2], end=" ")

  # nestDat[:,2][flooded==True] = whichStorm[flooded==True]
  if con.debugNests>=1: print("\t\t\t|>|> hatch?", sum(hatched), end=" ")
  if con.debugNests>=3: arrPrint(hatched)
  if con.debugNests>=1: print( "\t\t\t\t|>|> flood?", sum(flooded))
  if con.debugNests>=3: arrPrint( flooded)

  # nestDat = np.concatenate((nestDat, trueFate[:,None]), axis=1)
  if con.debugNests>=2: print("\n\t\t>>>>> true final nest fates:\n")# # ---- TRUE DSR ------------------------------------------------------------
  arrPrint(trueFate)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
