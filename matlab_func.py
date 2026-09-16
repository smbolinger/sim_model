
import functools
import matplotlib.pyplot as plt
import numdifftools as ndt
import numpy as np
import pandas as pd
import jax
# import jaxopt
import jax.numpy as jnp

from dsrCalc import calc_daily_expo
from helpers import print
from print_func import dfPrint
from rsettings import config, odir, atype
from scipy import optimize
# from jax.scipy.optimize import minimize
# from jaxopt import ScipyBoundedMinimize
# from statsmodels.tools.numdiff import approx_hess2 ## just an approximation
# print("loading optimization functions")
rng = np.random.default_rng(seed=config.rngSeed)
rng_key = jax.random.key(config.rngSeed)
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
np.set_printoptions(suppress=True,precision=5,edgeitems=50,threshold=200)
from jax import config as jconfig
jconfig.update("jax_enable_x64", True)
jconfig.update("jax_traceback_filtering", "off")
if atype=="norm":
  # jconfig.update("jax_log_compiles", True)
  jconfig.update("jax_explain_cache_misses", True)

@jax.jit
def jPolyLike(pZero,dVal,fateVal,sclf,db=0):
  """
    NOTE: things that rely on pZero need to be created in this function
    not the wrapper, so we can call hessian with same args
  """
  pZero = sclf*pZero ## pZero automatically promoted to jnp.array
  # pZero = jnp.array(pZero)
  # if db>=2:
  # jax.debug.print("transformed pZero: {pZ}", pZ=pZero)

  s0 = 1.0 - jnp.sum(pZero)
  pZero_0 = pZero[0] ## explicitly separate to avoid dim mismatch
  # m2 = 1 - s0 - pZero[0] # won't be used anyway if len(pVal) = 1
  m2 = 1.0 - s0 - pZero_0 
  # arr = [[s0,0,0],[pZero[0],1,0],[m2,0,1]]
  ## explicitly build row-by-row and don't create python obj at all!
  row0 = jnp.array([s0, 0.0, 0.0])
  row1 = jnp.array([pZero_0, 1.0, 0.0])
  row2 = jnp.array([m2, 0.0, 1.0])
  M = jnp.stack([row0, row1, row2])
  def power_step(current_M, _):
    return current_M @ M, current_M

  _, mPowStack = jax.lax.scan(power_step, jnp.eye(3), None, length=12)
  mPowFate = mPowStack[dVal, fateVal, 0]

  ## these vals should stay constant - should only print if changed:
  print(
    f"\t\t{dVal.shape[0]=}\t{fateVal.shape[0]=}\t{M}"
    f"\t{mPowStack.shape[0]=}\t{mPowFate.shape[0]=}") 

  # jax.debug.print("\nmPowFate={mpf}",mpf=mPowFate)
  # jax.debug.print("\ndVal={dval}",dval=dVal)
  # jax.debug.print("\jnp.log(mPowFate)={dval}",dval=jnp.log(mPowFate))
  ## need mPowFate at padded locations to == 1 so log is 0
  ## don't need a separate nll_lookup function
  return -jnp.sum(jnp.log(mPowFate))

def jplWrapper(pZero,dVals,fates,sclf):
  """ Calls jPolyLike thru ll_valgrad to get val & gradient """
  pZeroJ = jax.device_put(pZero)
  loss, grad = ll_valgrad(pZeroJ,dVals,fates,sclf)
  # print(f"{val=} {type(val)=}")
  # print(f"{grad=} {type(grad)=}")
  return float(loss), np.array(grad)

ll_valgrad = jax.value_and_grad(jPolyLike)

# def PolyMort(obsData,survey,config,useJax=False,scl_fac=0.1,plt=True,suff=""):
# def PolyMort(obsData,survey,par,config,useJax=True,scl_fac=0.05,plt=False,db=0,suff=""):
def PolyMort(obsData,survey,par,config,useJax=True,scl_fac=0.15,plt=False,db=0,suff=""):
  outLen = 2400 if par.numNests<=300 else 3200
  if plt: outLen = outLen + 1000

  # recreate the data to be used:
  obs = mk_obs_mat(obsData,survey,config,db=db)
  nObs = int(obs.shape[0])
  obsData = jnp.array(obsData.to_numpy())
  # dVal = obsData[:,1].astype(jnp.int64) fateVal = obsData[:,2].astype(jnp.int64)
  ## pick a random number and add one so not all intervals are same?
  # if par.stormFrq==0 or par.stormDur==0:
    # randn = int(rng.uniform(low=0,high=nObs,size=1)[0])
    # obs[randn,1]+=1
  dVal = obs[:,1].astype(jnp.int64)
  fateVal = obs[:,2].astype(jnp.int64)
  # print(f"{nObs=} {dVal=} {fateVal=}")

  ## use the same data used for logistic exposure:
  # obsData = jnp.array(obsData.to_numpy())
  # ID, status,expos = obsData.T
  # dVal = expos.astype(jnp.int64)
  # fateVal = status.astype(jnp.int64)
  # nObs = int(obsData.shape[0])

  pWidth = ((0,outLen-nObs))

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if outLen<nObs: print(f"\t\t!! {pWidth=}")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # dVal = jnp.pad(dVal,pad_width=pWidth,constant_values=-99) ## should pad w/-99
  dVal = jnp.pad(dVal,pad_width=pWidth) ## should pad w/zeros
  fateVal = jnp.pad(fateVal,pad_width=pWidth) ## should pad w/zeros
  ## padding with 0 gets correct index to make mPowFate == 1 so log = 0
  # fateVal = jnp.pad(fateVal,pad_width=pWidth,constant_values=-2) ## should pad w/zeros
  # print(dVal)

  K = len(np.unique(fateVal)) - 1 ## number of different fates minus 1
  gtolr = 1e-6
  ftolr = 1e-6 # opt = {'gtol':gtolr,'ftol':ftolr,'disp':True}
  opt = {'disp':True}
  # fun1 = jPolyLike # met = config.optimizer
  met = "L-BFGS-B"
  # arg = (obs,nObs,scl_fac) ## extra args to pass to jplWrapper
  arg = (dVal,fateVal,scl_fac) ## extra args to pass to jplWrapper
  lb, ub = 0.001, 1.0
  bnd = optimize.Bounds(lb,ub)
  # ll_obj =  functools.partial(fun1,obs=obs,nObs =nObs,scl_fac=scl_fac)
  fun = jplWrapper 
  jaco = True 
  pZero = rng.uniform(low=0.2,high=0.7,size=(K)) 
  # pZero = rng.uniform(low=0.1,high=0.9,size=(K)) 

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # if config.testing=="yes" and db>=2:
  if db>=4:
    print(f"\t\t\t{nObs=} {pWidth=}\n\t\t{dVal=}\n\t\t{fateVal=}")
    if plt: plot_jac(fun,arg,suff)
  if db>=2:
    print(f"\n\t\t>> PolyMort: {K=} {type(K)=} {nObs=} {type(nObs)=} "
          f"{type(pZero)=} {pZero.dtype=} {pZero.shape=}\n"
          f"\t\t\t {type(dVal)=} {dVal.dtype=} {type(fateVal)=} {fateVal.dtype=}")
    # print(f"\t\t {type(obs)=} {obs.dtype=} {obs.shape=} ")
    ## NOTE specifying scl_fac inside of jPolyLike insteead of as arg
    print(f"\t\t>> PolyMort: run optimizer - {met=} {gtolr=};"
          f" untransformed {pZero=}\n\t\t{fun=} {bnd=}")
    ## NOTE can use keep_feasible w/trust-constr to stay w/in bounds throughout
    ## NOTE don't need to pass arg if obs is in the class instance

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  out = optimize.minimize(fun,
                          pZero,
                          args=arg,
                          method = met,
                          jac=jaco,
                          # jac=ndt.Jacobian(lambda x: fun(x,obs)),
                          # hess=ndt.Hessian(lambda x: fun(x,obs)),
                          bounds = bnd, # constraints = con,
                          # options={'gtol':1e-12, 'disp':True},
                          options=opt,
                          # options={'disp':True},
                 )

  hess = np.asarray(jax.hessian(jPolyLike)(out.x,*arg),dtype=np.float64)
  se   = np.sqrt(np.diag(np.linalg.inv(hess))) * scl_fac
  ans  = out.x * scl_fac
  s    = 1-sum(ans) ## one minus sum of fitted values
  # print(f"\t>> PolyMort: {out.success=} {out.message=} {out.nit=} {out.nfev=}")

  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if db>=2:
    print(f"\t\t>> PolyMort: {out.x=} {out.success=} {out.message=} {out.nit=} {out.nfev=}")
    print(f"\t\t*** PolyMort: {ans.dtype=} {ans.shape=} {ans=} {s=} ",end=" ")
  #-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ## this was summing along each axis
  # seS = np.sqrt(np.sum(np.sum(np.linalg.inv(hess)))) * scl_fac
  # seS = np.sqrt(np.sum(np.linalg.inv(hess))) * scl_fac
  # print(f"{se=} {seS=}",end=" ")
  # print(f" {hess.flatten()=}")

  # return (s, seS, ans[0], se[0])
  return (s, ans[0], se[0])

def mk_obs_mat(obsData,survey,config,exp1=False,db=0):
  ## can use dat2S to avoid having to call calc_daily_expo? even tho not inside optimization
  if isinstance(obsData, pd.DataFrame):
    # obsData = obsData.to_numpy()
    # obsData = jnp.array(obsData.to_numpy())
  # elif not isinstance(obsData, jnp.ndarray):
    # obsData = jnp.array(obsData)
    obsData = np.array(obsData.to_numpy())
  elif not isinstance(obsData, np.ndarray):
    obsData = np.array(obsData)

  if db>=4:
    print(f"\tmk_obs_mat: in=")
    dfPrint(obsData,nprint=30)
  nNest = obsData.shape[0]
# colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs', 'sint')
  ID, init,end,tfate,ff, la, lc, afate,nObs = obsData.T
  nrows    = int(np.sum(nObs))
  # out = np.ones(shape=(nrows,3), dtype=np.int64)
  out = np.zeros(shape=(nrows,3), dtype=np.int64)
  nObs = nObs.astype(int)

  if exp1:
    first,last,fate = init,end,tfate
  else:
    # first,last,fate = ff,la,afate
    first,last,fate = ff,lc,afate
  if db>=4:
    print(f"\t\tmk_obs_mat: {nrows=} {len(first)=}"
          f" {len(last)=} {len(fate)=}\n\t\t {nObs=}")
    print(f"\t\tmk_obs_mat: {type(out)=} {out.dtype=}")
  if db>=5:
    print(f"\t\tmk_obs_mat: {fate=} {first=} {last=}")
    # print(f"mk_obs_mat: {fate=} {expos.T=}")
  #   print(f"mk_obs_mat: {endDay.T=} {ID.dtype=}{expos.dtype=} {init.dtype=}")
  # # if config.debugLL>=2:
  # if db>=2:
  #   print(f"mk_obs_mat: input matrix for optim:")
  #   dfPrint(out) # out = jnp.asarray(out)
  #   print(f"\t\tmk_obs_mat: {type(out)=} {np.isnan(out).sum(axis=0)=}")
  #

  # expos, obsDay = expo
  expos, obsDay = calc_daily_expo(nNest,survey,first,last,config)
  if db>=4:
    print(f"\tmk_obs_mat: {ID.dtype=}{expos.dtype=} {init.dtype=}")
    print(f"\tmk_obs_mat: {len(ID)=}{len(expos)=} {len(init)=}")
  endDay = np.cumsum(nObs) -1 #+> zero-indexed
  endDay = endDay.astype(int)
  ## NOTE: DON'T switch the fate coding - the way it is set up, 0 just means
  ## NOTE: the value in the 0,0 position of the matrix, which is pSurv
  ## NOTE: and that is why importing dat2S doesn't work!
  # fate = np.array([0 if i in [1,2,7] else 1 for i in fate])
  out[:,0] = np.repeat(ID,nObs)
  # print(f"mk_obs_mat: {type(out)=} {out=}")
  out[:,1] = expos
  if db>=3:
    print(f"\tmk_obs_mat: {endDay.T=} {fate=} {expos.T=}")
  if db>=4:
    print(f"\tmk_obs_mat: in=")
    dfPrint(obsData,nprint=30)
    print(f"\tmk_obs_mat:  out=")
    dfPrint(out,nprint=30)
  # out[:,2][endDay] = fate + 1 
  out[:,2][endDay] = fate 
  # print(f"mk_obs_mat: {type(out)=} {out=}")

  # if config.testing=="yes":
  #   if config.debugLL>=2:
  #     print(f"mk_obs_mat: input matrix for optim:")
  #     dfPrint(out) # out = jnp.asarray(out)
  # print(f"\t\tmk_obs_mat: {type(out)=} {np.isnan(out).sum(axis=0)=}")

  return out



##-------------------
class Optimized:
  ## should allow to make calculations once and then reference from
  ## the two separate functions for value and gradient?
  def __init__(self, obs):
    self.obs = obs
    self.nObs = obs.shape[0]
    self.cached_x = None
    self.cached_val = None
    self.cached_grad = None
    self.cached_hess = None

  # def jplCompute(self,pZero,obs):
  def jplCompute(self,pZero):
    # nObs = int(self.obs.shape[0]) 
    pZero = np.asarray(pZero, dtype=np.float64).flatten()
    if self.cached_x is not None and np.array_equal(pZero, self.cached_x):
            return
    # loss, grad = ll_valgrad(pZero,obs)
    loss, grad, hess = ll_valgrad(pZero,self.obs,self.nObs)
    # val = float(loss) # grad = np.array(grad,dtype=np.float64)
    self.cached_x = np.copy(pZero)
    self.cached_val = float(loss)
    self.cached_grad = np.array(grad, dtype=np.float64)
    self.cached_hess = np.array(hess, dtype=np.float64)

  def objective(self,pZero):
    # self.jplCompute(pZero,obs)
    self.jplCompute(pZero)
    return self.cached_val

  def gradient(self,pZero):
    # self.jplCompute(pZero,obs)
    self.jplCompute(pZero)
    return self.cached_grad

# def plot_jac(fun,obs,suff=""):
def plot_jac(fun,arg,suff=""):
  beta1 = jnp.linspace(0,1,21)
  beta2 = jnp.linspace(0,1,21)
  beta = jnp.array([beta1,beta2])
  print(f">> {beta=}")
  print(">> creating jax gradient",end=" ")
  ll_grad = jax.grad(fun,argnums=0) ## vectorized gradient function
  print(">> creating mapped jax gradient",end=" ")
  ll_vecgrad = jax.vmap(ll_grad, in_axes=(0,None,None,None)) ## vectorized gradient function
  print(">> making plot",end=" ")
  # fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))
  plt.plot(beta,ll_vecgrad(beta,*arg))
  # ax1.plot(beta,ll_vecgrad(beta,obs), lw=2)
  print(">> adding axis",end=" ")
  # ax1.plot(beta,ll_vecgrad(beta,*arg), lw=2)
  print(">> saving plot to file")
  # ax2.plot(beta,ll_obj(beta), lw=2)
  plt.savefig(f"figs/jac{suff}.png")


## define globally for some reason?
valgrad =   jax.jit(jax.value_and_grad(jPolyLike), static_argnums=2 )
hessFun =   jax.jit(jax.hessian(jPolyLike), static_argnums=2 )

# def jplWrapper(pZero,obs,nObs):
# def jplWrapper(pZero,obs,nObs,sclf):
# def jplWrapper(pZero,dVals,fates,nObs,sclf):
# def ll_valgrad(x,obs,nObs): ## obs is passed from w/in the Class instance
# def ll_valgrad(x,obs,nObs,sclf): ## obs is passed from w/in the Class instance
#   # val, grad = jax.value_and_grad(jPolyLike)(x,obs,nObs) 
#   val, grad = jax.value_and_grad(jPolyLike)(x,obs,nObs,sclf) 
#   # hess      = jax.hessian(jPolyLike)(x,obs,nObs)
#   # return val, grad, hess
#   return val, grad

def jrandom(rng_key,K,scl_fac):
  minv = scl_fac
  maxv = 10 * scl_fac
  rng_key, subkey = jax.random.split(rng_key)
  # ret = jax.random.uniform(subkey,shape=K,minval=0.1,maxval=1)
  ret = jax.random.uniform(subkey,shape=K,minval=minv,maxval=maxv)
  return ret

  # M = jnp.array(arr,dtype=jnp.float64) # print(f"\t\t{mat=}") 
  # NLL = 0 ## initial NLL value NLL = nll_lookup(M,dVal,fateVal)
  # return nll_lookup(M,dVal,fateVal)
