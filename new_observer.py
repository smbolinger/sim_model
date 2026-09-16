

def mk_histories(storm, inits, par, rng, conf, inff, nw):
  """
    Make nest histories separate from observations
  """
  nd     = np.zeros(shape=(par.numNests, 3), dtype=np.int16)
  nData      = mk_nests(par, rng, nd, conf, inits, initff=inff, nWeek=nw)
  nestPeriod   = mk_per(nData[:,1], (nData[:,2]), con=conf) # changed output of mk_nests 
  stormOut     = storm_nest(par.stormFrq, nestPeriod, storm, con=conf)
  stormDat     = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests, con=conf,rng=rng)
  hatched    = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
  ## make true nest fates:
  nData    = mk_fates(nData, par.numNests, hatched, stormDat, storm, con=conf)
  if conf.other=="setn":
    nData = mk_set_nests(par.numNests,0.50, rng, conf)
  return nData

# def make_obs(trueNestDat, par,rng, obsVarNum, storm, survey, conf, initDat,stormUnk, nw=2, inff=True, pandas=False):
def make_obs(par,rng, obsVarNum, storm, survey, conf, initDat,stormUnk, nw=2, inff=True, pandas=False):
  """
    1. Call functions mk_nests, mk_per, storm_nest, mk_flood, mk_fates, & observer

    2. Combine the output into an array: 

     [0]:nest ID...........[1]:initiation.......[2]:end date..........
     [3]:true fate ........[4]:first found......[5]:last active.......
     [6]:last checked......[7]:assigned fate....[8]:num obs active.... 
     [9]:len(final int)....[10]:num obs total....

    --------
    RETURNS:
      numpy ndarray containing nest & observation data (column indices above)
      
      Can also uncomment lines to save nest data to .npy file
      
    And other lines to make nest data that's compatible with the old script.
  """

  #----------------------------------------------------------------------------
  # nd     = np.zeros(shape=(par.numNests, 3), dtype=np.int16)
  # nd2    = np.zeros(shape=(par.numNests, 7), dtype=np.int16)
  nd2    = np.zeros(shape=(par.numNests, 8), dtype=np.int16)
  colnames = ['ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs']
  # if conf.debugObs>=3: print(f"\t\t{len(colnames)=}")

  # NOTE should I make sure all nests live for at least a day?

  # +> ---- make the nests: ---------------------------------------------------
  if conf.debugNests>=2: print("\n\t[*] [*] [*] [*] [*] making nests [*] [*] [*] [*] [*] [*] [*] [*] ")
  trueNestDat = mk_histories(storm,initDat,par,rng,conf,inff,nw)
  # if conf.debugNests>=2: print("\n\t[*] [*] [*] [*] [*] making nests [*] [*] [*] [*] [*] [*] [*] [*] ")
  # nData      = mk_nests(par, rng, nd, conf, initDat, initff=inff, nWeek=nw)
  # nestPeriod   = mk_per(nData[:,1], (nData[:,2]), con=conf) # changed output of mk_nests 
  # stormOut     = storm_nest(par.stormFrq, nestPeriod, storm, con=conf)
  # stormDat     = mk_flood(storm, par.pMortFl, stormOut, numNests=par.numNests, con=conf,rng=rng)
  # hatched    = (nData[:,2]-nData[:,1]) >= par.hatchTime # hatched before storms accounted for
  # nData    = mk_fates(nData, par.numNests, hatched, stormDat, storm, con=conf)
  ## if config says make set-fate nests, replace nData:
  # if conf.msg=="setn":
  # if conf.other=="setn":
  #   nData = mk_set_nests(par.numNests,0.50, rng, conf)
    # print(f"{type(nData)=} {nData.shape=}")

  # +> ---- observer: ---------------------------------------------------------
  # if conf.debugObs>=3: print("\n\t[*] [*] [*] [*] [*] observer [*] [*] [*] [*] [*] [*] [*] [*] ")
  if conf.debugObs>=2: print(
      f"\n\t[*] [*] [*] [*] [*] observer ({par.stormFate=}) [*] [*] [*] [*] [*] [*] [*] [*] ")
  # obs = observer(nData,par,rng,surveys=survey,stormDays=storm,out=nd2,stormUnk=stormUnk,conf=conf)
  # obs = observer(nData,par,rng,survey,storm,nd2,stormUnk,conf)
  obs = observer(trueNestDat,par,rng,survey,storm,nd2,stormUnk,conf)

  # +> ---- concatenate to make data for the nest models: ---------------------
  # nestData = np.concatenate((nData, 
  nestData = np.concatenate((trueNestDat, 
                 obs #stormOut[0][:,None] # storms per nest
                 ), axis=1)

  if pandas: nestData = pd.DataFrame(nestData,columns=colnames)
  ## last day nests are active = max init date plus hatch time
  # maxNestDay = np.max(nestData[:,1]) + par.hatchTime
  # maxSurveyDay = maxNestDay + par.obsFreq


  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.testing=="yes":
  #   if conf.debugObs>=5:
  #     print("\t\tNEST DATA from make_obs():")
  #     dfPrint(nestData, names=colnames)
  #   # print(f"{nData=}")
  #   # if conf.debugObs>=3:
  #   #   print("\t\tnestdata:")
  #   #   dfPrint(nData, nprint=30)
  #   # if conf.debugObs>=6: 
  #   #   print(f"\t\t\t|>end - init >= hatch time:")
  #   #   for x in range(5):
  #   #     print(f"\t\t\t\t{nData[x,2]}-{nData[x,1]}>={par.hatchTime}" )
    if conf.debugObs>=3:
      # print_prop(nestData, 8)
      print_prop(nestData, obsVarNum)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return(nestData)

def discover_nests(nData, par, rng, conf, surveys, svysTilDisc,pos):
  if conf.debugObs>=4: 
    print("\t>>discover_nests: input - ",
          f"{svysTilDisc=}\n\t{pos=}\n\t{nData=}")
  # initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
  # print(f"{initiation=}")
  surveyDays, surveyInts,stormSurvey = surveys
  # print(f"{surveyDays=}")
  # pos = svy_position(initiation, end, surveys[0], cn=conf)
  # print(f"{pos=}")
  tot_svy      = pos[1] - pos[0]  #+> num surveys while nest is active
  # svysTilDiscovery = rng.negative_binomial(n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
  discovered     = svysTilDisc< tot_svy
  num_obsTrue = tot_svy - svysTilDisc#num_svy = tot_svy - svysTilDiscovery -1 # -1 for final survey
  if conf.debugObs>=4:
    print(f"\t\t\t|> num discovered  = {sum(svysTilDisc < tot_svy)=}", end=" ")
    print("\t\t\t|>true num obs:", end=" ")
    arrPrint(num_obsTrue, abval=20)
    print("\t\t\t|>total num surveys for each nest:", end=" ")
    arrPrint(tot_svy, abval=20)
    print("\t\t\t|> surveys til discovery = ", end=" ")
    arrPrint(svysTilDisc, abval=20)
  return discovered, num_obsTrue

def get_ijk(surveyDays, pos, svysTilDisc):
  # surveyDays, surveyInts = survey
  iVal = surveyDays[pos[0]+svysTilDisc] # i
  kVal = surveyDays[pos[1]]
  jVal = surveyDays[pos[1]-1]

  return iVal, kVal, jVal

def observer(nData, par, rng,surveys, stormDays, out,stormUnk, conf):
  """
    The observer searches for nests on survey days.
    Surveys til discovery (success) are calculated as random draws from a
    negative binomial distribution with daily success probability of discProb.
    If surveys til discovery is less than total number of surveys while nest
    is active, then nest is discovered. The observer then assigns fate in
    assign_fate. 
    -----
    ARGS
      nData = id, init, end, fate
      par = params for this set
      surveys = survey days, intervals, storm survey
      stormDays = storm days
      out = premade array for output
    -------
    RETURNS
      ndarray w/ nrows=numNests. 
      [columns = i, j, k, assigned fate, num obs int, intFinal, stormFinal]
    ---------
    NOTES
      Remember, pos[0] is the first survey after initiation, and pos[1] is the first survey after end.
      Now I'm only applying assign_fate to discovered nests to see if
      the proportions are better (02Aug 2026)
      Or could remove undiscovered nests in the observer functions

  """

  #----------------------------------------------------------------------------
  out = np.zeros(shape=(par.numNests, 8), dtype=np.int16)
  initiation, end, fate = nData[:,1], nData[:,2], nData[:,3]
  if conf.debugObs>=4:
    print("\t>> observer: getting survey position of init & end")
  pos = svy_position(initiation, end, surveys[0], cn=conf)
  svysTilDiscovery = rng.negative_binomial(
      n=1, p=par.discProb, size=par.numNests) # see above for explanation of p 
  surveyDays, surveyInts,stormSurvey = surveys
  if conf.debugObs>=4: print("\t>> observer: discovering nests")
  discovered,num_obsTrue = discover_nests(
      nData,par,rng,conf,surveys,svysTilDiscovery,pos)
  hatched        = nData[:,3] == 0
  # num_obsTrue[~discovered] = 0 #+> not discovered = 0 observations
  num_obs = num_obsTrue
  num_obs[~hatched] -= 1
  intFinal  = surveyInts[pos[1]] # +> actual length of final int for each nest
  iVal, kVal, jVal = get_ijk(surveyDays, pos, svysTilDiscovery)
  
  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.testing=="yes":
    if conf.debugObs>=6:
      print("nest data!")
      arrPrint(nData, abval=20) # if conf.debugObs>=4:
      print("\t\t\t\t>>>> init dates:",end=" ")
      arrPrint(nData[:,1], abval=20)
      print("\t\t\t\t>>>> position of initiation date in survey day list:",end=" ") 
      arrPrint( pos[0], abval=20)
      print("\t\t\t\t>>>> end dates:",end=" ")
      arrPrint(nData[:,2], abval=20)
      print("\t\t\t\t>>>> position of end date in survey day list:",end=" ")
      arrPrint( pos[1], abval=20) 
      print("\t\t\t\t>> survey days with index number:",end=" ")
      arrPrint( surveyDays,abval=20)
    if conf.debugObs>=3:
      print("\t\t\t|> nest discovered? (svysTilDiscovery < tot_svy)")
      arrPrint(discovered, abval=20)
      print("\t\t\t|>num obs while active:", end=" ")
      arrPrint(num_obs, abval=20)
      # print(f"\t\t\t{intFinal=}")
    if conf.debugObs>=6:
      print("\t\t\t\tj & k before:")
      arrPrint(jVal, abval=20)
      arrPrint(kVal, abval=20)

  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  jVal[fate==0] = kVal[fate==0]
  if conf.debugObs >=4:
    print(f"\t>> observer: fill in output matrix")
    print(f"\t\t>> observer: i j k columns")
  out[:,0][discovered] = iVal[discovered]
  out[:,1][discovered] = jVal[discovered] 
  out[:,2][discovered] = kVal[discovered]
  if conf.debugObs >=4: print(f"\t\t>> observer: final interval columns")
  finalInterv  = mk_per(out[:,1], out[:,2],con=conf)
  stormLoc = storm_nest(par.stormFrq, finalInterv, stormDays, conf)
  stormFinal = (stormLoc==1).any(axis=1) #+>storm in final interval?
  out[:,6] = num_obsTrue
  out[:,7] = stormFinal.astype(int)

  #+> need number of obs between i and j
  out[:,4] = num_obs 
  out[:,5] = intFinal.astype(int) # length of final interval - transform to integer for the ndarray
  longFinal  = out[:,5] > par.obsFreq
  # longFinal  = out[:,5][discovered] > par.obsFreq
  if conf.debugObs >=4: print(f"\t\t>> observer: assigned fate column")
  fate2 = fate[discovered]
  ifin2 = intFinal[discovered]
  sfin2 = stormFinal[discovered]
  if conf.debugObs >=4:
    print(f"\t>> observer: {fate2=} ({len(fate2)=})")
    # print(f"({len(out[:.3][discovered])=})")
  arrays = [stormFinal,longFinal,intFinal,discovered]
  # out[:,3] = assign_fate(par,rng,stormFinal,longFinal,fate,intFinal,stormUnk,conf)
  out[:,3] = assign_fate(par,rng,arrays,fate,stormUnk,conf)
  # out[:,3][discovered] = assign_fate(par,rng,sfin2,longFinal,fate2,ifin2,stormUnk,conf)

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if conf.testing=="yes":
  #   if conf.debugObs>=5:
  #     print("\t\t\t\tj & k:")
  #     arrPrint(jVal, abval=20)
  #     arrPrint(kVal, abval=20)
  #   if conf.debugObs>=4:
  #     print(f"\t\t\t{iVal=} \n\t\t\t{jVal=} \n\t\t\t{kVal=}")
  #   if conf.debugObs>=4:
  #     print(f"\t\t\t{np.where(out[:,3]==7)=}")
  #     print(f"\t\t\t{np.where(out[:,3]==2)=}")
    if conf.debugObs>=5:
      print(f"\t\t\t{stormLoc=}")
      print(f"\t\t\t{stormFinal=}{stormFinal.astype(int)=}")
      print(f"\t\t\t{np.where(stormFinal)=}{np.sum(stormFinal)=}")
      print(f"\t\t\t{np.where(longFinal)=}{np.sum(longFinal)=}")
  #   if conf.debugObs>=5:
  #     # print(f"\t\t\t{finalInterv=}") ##NOTE prints annoyingly
  #     print(f"\t\t\t{stormLoc=}")
  #     print(f"\t\t\t{stormFinal=}") ## not always caught if nest ends or begins on day of storm?
  #     print(f"\t\t\t{longFinal=}")
    if conf.debugObs>=4:
      print("\t\tout=")
      dfPrint(out)

  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return(out)

#-----------------------------------------------------------------------------
# @profile
     # [9]:len(final int)....[10]:num storms......[11]:num obs total....
# def make_obs(par,rng, obsVarNum, storm, survey, conf, initDat,stormUnk, nw=2, inff=True, pandas=False):

def add_misclass(par,rng,trueFate,assignedFate,cn,db=0):
  #+> has to be discovered nests!
  #NOTE don't reset seed each time; pass from main script as arg
  # val   = [0,9,11,112,13,1131,1211] if par.MCtype == "hatch2fail" else [1,2]
  
  if par.MCtype!="none":
    val   = [0] if par.MCtype == "hatch2fail" else [1,2]
    mcVal =  2 if par.MCtype == "hatch2fail" else 0
    uVal = 7
    eqval = np.isin(trueFate,val)
    nMisclass = int(np.round(par.propMC * np.sum(eqval)))
    nUnknown  = int(np.round(par.propUnk * np.sum(eqval)))
    tot       = nMisclass + nUnknown
  else:
    fval = [0,1,2]
    eqval = assignedFate[assignedFate!=7]
    nMisclass = int(np.round(0.05*np.sum(eqval)))
    nUnknown = int(np.round(0.0*np.sum(eqval))) ## unknowns already happen
    tot       = nMisclass + nUnknown
    val = rng.choice(fval, size=tot,replace=True)
    uVal = 7
    limit = np.max(fval)
    add = rng.choice(np.arange(20), size=tot, replace=True)
    mcVal = (val + add) % limit
  # tot       = nMisclass + nUnknown

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if db>=3: print(f"\t\t{eqval=}   {len(eqval)=} ")
    if db>=4: print(f"\t\t\t\t{eqval=}   {np.sum(eqval)=} ")
    if db>=2: print("\t\t\t NUMBER TO MISCLASSIFY:", nMisclass, end=" ")
    if db>=2: print("\t\t\t NUMBER TO MARK UNKNOWN:", nUnknown, end=" " )
    if db>=2: print(f"\t\t\t\t{val=} , {mcVal=} , {uVal=} , {tot=}")

    if db>=4: print(f"\t\t\tBEFORE <{len(assignedFate)}> : {assignedFate=}")

  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ## NOTE for some reason, returns a tuple with 1 array instead of just an array
  ind = np.where(eqval) ##+> select random indices to replace at 
  ind = ind[0]
  mask = rng.choice(ind, size=tot, replace=False)
  if par.MCtype!="none":
    rep_vals  =[np.repeat(mcVal,nMisclass), np.repeat(uVal,nUnknown)]
    rep_vals  = np.concatenate(rep_vals).tolist()
  else:
    rep_vals = mcVal
  if db>=4: print(f"\t{rep_vals=}")
  assignedFate[mask] = rep_vals

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if db>=5: print(f"\t\t\t{ind=} {len(ind)=}", end=" ")
    if db>=5: print(f"\t{mask=}", end=" ")
    # if db>=5: print(f"\t{rep_vals=}")
    if db>=4: print(f"\t\t\tAFTER: {assignedFate=}")

  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  return assignedFate
#-----------------------------------------------------------------------------

def assign_fate( par, rng,arr,trueFate, stormUnk,cn):
  """
  Observer assigns correct or incorrect fate based on some conditions:
    The observer assigns the correct fate based on a comparison of a set 
    probability to random draws from a uniform distribution. If observer is 
    incorrect, then they assign a fate of unknown unless stormFate==True, in
    which case all nests that ended in a period that contained a storm are 
    assumed to have failed due to the storm.

    Arguments: 
    assignVal = the value given for incorrect fates
  
    fateCuesProb=random values to compare
    fateCuesPres=probability of fate cues being present
      >-> created within the function using exp decay & final int length
    if fate percentages are fixed, fateCuesPres should be the same (=1) for all
  
    Returns: vector w/ assigned fate for each nest
  """

  decay="exp"
  # decay="lin"
  stormFin,longFin,intFinal,discovered = arr

  
  # assignedFate=np.empty(par.numNests)
  numNests = len(trueFate)
  assignedFate=np.empty(numNests)
  assignedFate.fill(7) # +> default=unk; fill w/ known fate if field cues allow
  if cn.debugObs>=4:
    print(f"{len(assignedFate)=}{len(trueFate)=}")
  if decay=="exp":
    fateCuesPresent   = expDecay(n0=1, k=par.decayRate, t=intFinal)
  else:
    fateCuesPresent   = 1 - (intFinal*par.decayRate)
  fateProb = rng.uniform(low=0, high=1, size=numNests)
  assignedFate[fateProb < fateCuesPresent] = trueFate[fateProb < fateCuesPresent] 
  wrongFateMask = fateProb > fateCuesPresent
  # print(f"{assignedFate[fateMask]=}")
  # print(f"\n\t\t{fateProb<fateCuesPresent=} | ")
  # print(f"{assignedFate[fateProb<fateCuesPresent]=}")

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if cn.debugObs>=5:
      # print(f"\t\t\tCORRECT FATE: {np.where(fateProb<fateCuesPresent)=}")
      # print(f"\t\t\t{intFinal=} {par.obsFreq=}")
      print(f"\t\t\tCORRECT FATE: np.where( {fateProb=} < {fateCuesPresent=}\n)"
            f"\t\t\t\t{(np.sum(fateProb<fateCuesPresent))=}")
      print(f"\t\t\t {assignedFate.shape=} {assignedFate=}")
    # if cn.debugObs>=3:
    #   # print(f"\t\t\t\t{np.where(fateProb<fateCuesPresent)=} {len(np.where(fateProb<fateCuesPresent))=}")
    #   print(f"\t\t\t\t{np.where(fateProb<fateCuesPresent)=}")
    if cn.debugObs>=2:
      print(f"\t\t***{decay=}***{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=} |"
            f" {np.sum(assignedFate==0)=}"
            f"\t\t{np.sum(fateProb<fateCuesPresent)=} | ")
      # with pd.option_context("display.max_columns",None):
    if cn.debugObs>=4:
      # print(f"{stormFin=}")
      print(f"\t\t\t{intFinal=} {len(intFinal)} ")#"\n\tcorrect={np.sum(intFinal<=par.obsFreq)=}")
      # print(f"\t\t\t{fateCuesPresent=}")
      # print(f"\t\t\t{fateProb=}")
    if cn.debugObs>=3:
      print(f"\t\t\t\t{np.where(fateProb<fateCuesPresent)=}")
      # print(f"\t\t\t{fateProb<fateCuesPresent=}")
      print(f"\t\t\t{assignedFate=} ;\n\t\t{assignedFate.shape=}")
      arrPrint(assignedFate, abval=20)

  tfd = trueFate[discovered]
  afd = assignedFate[discovered]
  if par.propMC>0 or par.propUnk>0:
    # assignedFate = add_misclass(par,rng,trueFate,assignedFate,cn,db=cn.debugObs)
    assignedFate[discovered] = add_misclass(par,rng,tfd,afd,cn,db=cn.debugObs)
    # if cn.debugObs>=2: print(f"\t>> mis-assigning fate based on proportions")
  else:
    # assignedFate = add_misclass(par,rng,trueFate,assignedFate,cn,db=cn.debugObs)
    ## when MCtype=="none", proportion MC is 0.05 and unknown is 0.0
    assignedFate[discovered] = add_misclass(par,rng,tfd,afd,cn,db=cn.debugObs)
    if par.stormFate:
      assignedFate[stormFin] = 2
      assignedFate[longFin] = 2
      ## both active during a storm AND no fate cues present
      ## "and" and "or" causee valueError
      # assignedFate[fateMask and (stormFin or longFin)] = 2
      ## these are all already unknown:
      # assignedFate[wrongFateMask & (stormFin | longFin)] = 2
      # assignedFate[fateMask][stormFin] = 2
      # assignedFate[fateMask][longFin] = 2
    if stormUnk and not par.stormFate:
      # print("marking storm nests unknown")
      # assignedFate[wrongFateMask & (stormFin | longFin)] = 7
      assignedFate[stormFin] = 7
      assignedFate[longFin] = 7
      # assignedFate[fateMask][stormFin] = 7
      # assignedFate[fateMask][longFin] = 7
    # if cn.debugObs>=2:
    #   print(f"\n\t\t{np.sum(fateProb<fateCuesPresent)=} | "
    #         f"\n\t\t{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=} |"
    #         f" {np.sum(assignedFate==0)=}")
    # if par.stormFate:
    #   assignedFate[longFin] = 2
    # # else:
    # if stormUnk and not par.stormFate:
    #   assignedFate[longFin] = 7

  # NOTE fate cues prob should affect all nest fates equally, not just failures
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #-*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if cn.testing=="yes":
    if cn.debugObs>=4:
      print(f"\t\t{trueFate[stormFin]=}")
      print(f"\t\t{np.where(trueFate[stormFin])=}")
      # stormFates = trueFate[stormFin]
      print(f"\t\t{trueFate[longFin]=}")
      print(f"\t\t{np.where(trueFate[longFin])=}")
      # print(f"\t\t{assignedFate[wrongFateMask & (stormFin | longFin)]=}")
      # print(f"\t\t{np.where(assignedFate[wrongFateMask & (stormFin | longFin)])=}")
      # stormFates = trueFate[longFin]
    if cn.debugObs>=2:
      print(f"\t\t\t\t.:.:{np.sum(assignedFate==7)=} | {np.sum(assignedFate==2)=}"
            f" | {np.sum(assignedFate==0)=}:.:.")
      # print(f"\n\t\tfateProb less than this number = correct fate "
      #       f"(unless long/storm final interval):\n\t{fateCuesPresent=}")
      # print(f"\twrong if cues_not_present==True and either long or storm final")
    if cn.debugObs>=3:
      print(f"\nMARK UNKNOWN IF CUES NOT PRESENT/LONG FINAL/STORM IN FINAL INTERVAL:")
      df2print = pd.DataFrame({
        # "init": 
        "true_fate": trueFate,
        "prob_of_cues": fateCuesPresent,
        "comparison_prob": fateProb,
        "cues_not_present": fateProb>fateCuesPresent,
        "long_final_int": longFin,
        "storm_final_int": stormFin,
        "assigned_fate": assignedFate,
        })
      # with pd.option_context("display.max_columns",60,"precision",2):
      with pd.option_context("display.max_columns",60,
                             'float_format', '{:.2f}'.format):
        print(df2print.T)
      # print(f"\t\t\t {assignedFate.shape=}")
    # if cn.debugObs>=3: arrPrint(assignedFate, abval=20)
  #-=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return(assignedFate)

