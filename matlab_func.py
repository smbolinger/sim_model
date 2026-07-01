
## transform everything to jax.numpy?
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

rng = np.random.default_rng(seed=config.rngSeed)
rng_key = jax.random.key(config.rngSeed)
np.set_printoptions(suppress=True,precision=7)
# print("loading optimization functions")
from jax import config as jconfig
jconfig.update("jax_enable_x64", True)
# if config.testing=="yes":
if atype=="norm":
  jconfig.update("jax_log_compiles", True)
  jconfig.update("jax_explain_cache_misses", True)

def jrandom(rng_key,K,scl_fac):
  minv = scl_fac
  maxv = 10 * scl_fac
  rng_key, subkey = jax.random.split(rng_key)
  # ret = jax.random.uniform(subkey,shape=K,minval=0.1,maxval=1)
  ret = jax.random.uniform(subkey,shape=K,minval=minv,maxval=maxv)
  return ret


def nll_lookup(M, dVal, fateVal):
  ## NOTE called from w/in jPolyLike
  Mpower = [jnp.eye(3), M]
  for _ in range(9):
    Mpower.append(Mpower[-1] @ M)
  ## violates jax rule to keep shape static:
  # dVal = dVal[dVal>0] fateVal = fateVal[dVal>0]
  # Mpower = [Mpower[-1]@M for _ in range(9)]
  mPowStack = jnp.stack(Mpower)
  mPowMatch = mPowStack[dVal][:,:,0].flatten()
  # print(f"\t\t{mPowStack=} {mPowMatch=}")
  mPowInd = jnp.arange(len(dVal)) * 3
  newInd = mPowInd + fateVal
  mPowFate = mPowMatch[newInd]
  
  # print(f"\t\t{newInd=} {mPowFate=}")
  if atype=="norm":
    print(
        f"\t\t{len(dVal)=} {len(fateVal)=}\n{len(M)}"
        f"\t\t{len(mPowStack)=} {len(mPowMatch)=} {len(mPowInd)=}"
        f"\t\t{len(newInd)=} {len(mPowFate)=}\n{mPowFate=}") ## these vals should stay constant
  ## need mPowFate at padded locations to == 1 so log is 0
  return -jnp.sum(jnp.log(mPowFate))
  
## things that rely on pZero need to be created in this function
## not the wrapper, so we can call hessian with same args
@jax.jit
def jPolyLike(pZero,dVal,fateVal,sclf):

  pZero = sclf*pZero
  # pZero = jnp.array(pZero)

  s0 = 1 - jnp.sum(pZero)
  m2 = 1 - s0 - pZero[0] # won't be used anyway if len(pVal) = 1
  arr = [[s0,0,0],[pZero[0],1,0],[m2,0,1]]
  M = jnp.array(arr,dtype=jnp.float64) # print(f"\t\t{mat=}") 

  NLL = 0 ## initial NLL value
  NLL = nll_lookup(M,dVal,fateVal)

  return NLL

def jplWrapper(pZero,dVals,fates,sclf):
  ## Calls jPolyLike thru ll_valgrad to get val & gradient
  loss, grad = ll_valgrad(pZero,dVals,fates,sclf)
  # print(f"{val=} {type(val)=}")
  # print(f"{grad=} {type(grad)=}")
  return float(loss), np.array(grad)

ll_valgrad = jax.value_and_grad(jPolyLike)

# def PolyMort(obsData,survey,config,useJax=False,scl_fac=0.1,plt=True,suff=""):
def PolyMort(obsData,survey,par,config,useJax=True,scl_fac=0.05,plt=False,suff=""):

  # obsData = jnp.array(obsData.to_numpy())
  # dVal = obsData[:,1].astype(jnp.int64)
  # fateVal = obsData[:,2].astype(jnp.int64)
  obs = mk_obs_mat(obsData,survey,config)
  dVal = obs[:,1].astype(jnp.int64)
  fateVal = obs[:,2].astype(jnp.int64)
  nObs = int(obs.shape[0])
  outLen = 1300 if par.numNests==250 else 2400
  pWidth = ((0,outLen-nObs))
  # dVal = jnp.pad(dVal,pad_width=pWidth,constant_values=-99) ## should pad w/-99
  dVal = jnp.pad(dVal,pad_width=pWidth) ## should pad w/zeros
  fateVal = jnp.pad(fateVal,pad_width=pWidth) ## should pad w/zeros
  ## padding with 0 gets correct index to make mPowFate == 1 so log = 0
  # fateVal = jnp.pad(fateVal,pad_width=pWidth,constant_values=-2) ## should pad w/zeros
  # print(dVal)

  K = len(np.unique(obs[:,2])) - 1 ## number of different fates minus 1
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

  if plt: plot_jac(fun,arg,suff)

  # print(f"\n\t>> PolyMort: {K=} {nObs=} {type(pZero)=} {pZero.dtype=} {pZero.shape=}")
  # print(f"\t\t {type(obs)=} {obs.dtype=} {obs.shape=} ")
  ## NOTE specifying scl_fac inside of jPolyLike insteead of as arg
  # print(f"\t>> PolyMort: run optimizer - {met=} {gtolr=}; untransformed {pZero=}\n{fun=} {bnd=}")
  ## NOTE can use keep_feasible w/trust-constr to stay w/in bounds throughout
  ## NOTE don't need to pass arg if obs is in the class instance

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
  # print(f"*** PolyMort: {ans.dtype=} {ans.shape=} {ans=} {s=} ",end=" ")

  ## this was summing along each axis
  # seS = np.sqrt(np.sum(np.sum(np.linalg.inv(hess)))) * scl_fac
  # seS = np.sqrt(np.sum(np.linalg.inv(hess))) * scl_fac
  # print(f"{se=} {seS=}",end=" ")
  # print(f" {hess.flatten()=}")

  # return (s, seS, ans[0], se[0])
  return (s, ans[0], se[0])

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

def mk_obs_mat(obsData,survey,config,exp1=False):
  if isinstance(obsData, pd.DataFrame):
    # obsData = obsData.to_numpy()
    obsData = jnp.array(obsData.to_numpy())
  elif not isinstance(obsData, jnp.ndarray):
    obsData = jnp.array(obsData)

  nNest = obsData.shape[0]
# colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs', 'sint')
  ID, init,end,tfate,ff, la, lc, afate,nObs = obsData.T
  nrows    = int(jnp.sum(nObs))
  # out = np.ones(shape=(nrows,3), dtype=np.int64)
  out = np.zeros(shape=(nrows,3), dtype=np.int64)
  nObs = nObs.astype(int)

  if exp1:
    first,last,fate = init,end,tfate
  else:
    first,last,fate = ff,la,afate
  # print(f"mk_obs_mat: {fate=}")

  # expos, obsDay = expo
  expos, obsDay = calc_daily_expo(nNest,survey,first,last,config)
  endDay = np.cumsum(nObs) -1 #+> zero-indexed
  endDay = endDay.astype(int)
  # print(f"mk_obs_mat: {endDay.T=} {ID.dtype=}{expos.dtype=} {init.dtype=}")
  # print(f"mk_obs_mat: {type(out)=} {out.dtype=}")
  out[:,0] = np.repeat(ID,nObs)
  # print(f"mk_obs_mat: {type(out)=} {out=}")
  out[:,1] = expos
  # print(f"mk_obs_mat: {type(out)=} {out=}")
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

