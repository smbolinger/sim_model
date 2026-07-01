
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
from rsettings import config, odir
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
#   jconfig.update("jax_log_compiles", True)
#   jconfig.update("jax_explain_cache_misses", True)

def jrandom(rng_key,K,scl_fac):
  minv = scl_fac
  maxv = 10 * scl_fac
  rng_key, subkey = jax.random.split(rng_key)
  # ret = jax.random.uniform(subkey,shape=K,minval=0.1,maxval=1)
  ret = jax.random.uniform(subkey,shape=K,minval=minv,maxval=maxv)
  return ret


def nll_lookup(M, dVal, fateVal):
# def nll_lookup(Mpower, dVal, fateVal):
  ## NOTE called from w/in jPolyLike
  Mpower = [jnp.eye(3), M]
  # for _ in range(9):
    # Mpower.append(Mpower[-1] @ M)
  Mpower = [Mpower[-1]@M for _ in range(9)]
  mPowStack = jnp.stack(Mpower)
  mPowMatch = mPowStack[dVal][:,:,0].flatten()
  # print(f"\t\t{mPowStack=} {mPowMatch=}")
  mPowInd = jnp.arange(len(dVal)) * 3
  newInd = mPowInd + fateVal
  mPowFate = mPowMatch[newInd]
  # print(f"\t\t{newInd=} {mPowFate=}")

  return -jnp.sum(jnp.log(mPowFate))
  
# def jPolyLike(pZero,obs,nObs):
# def jPolyLike(pZero,obs,nObs,scl_fac):
# def jPolyLike(pZero,obs,nObs):
## things that rely on pZero need to be created in this function
## not the wrapper, so we can call hessian with same args
# def jPolyLike(Mpower,dVal,fateVal):
# @jax.jit
def jPolyLike(pZero,dVal,fateVal,sclf):
  # K=2
  # nObs = obs.shape[0] ## jax cannot evaluate?
  ## NOTE: make sure this scaling constant matches scl_fac in PolyMort
  # pZero = 0.05*pZero
  # print(f"{pZero=}")
  pZero = sclf*pZero
  pZero = jnp.array(pZero)
  s0 = 1 - jnp.sum(pZero)
  m2 = 1 - s0 - pZero[0] # won't be used anyway if len(pVal) = 1
  arr = [[s0,0,0],[pZero[0],1,0],[m2,0,1]]
  M = jnp.array(arr,dtype=jnp.float64) # print(f"\t\t{mat=}") 
  NLL = 0 ## initial NLL value

  NLL = nll_lookup(M,dVal,fateVal)

  # M2 = M @ M
  # M3 = M2 @ M 
  # M4 = M2 @ M2
  # M5 = M3 @ M2 
  # M6 = M3 @ M3

  # Mpower = jnp.array([M,M2,M3,M4,M5,M6])
  # ## this is actually MUCH slower
  # LL = Mpower[dVal,fateVal,0]
  # NLL = -sum(jnp.log(LL)) # print(f"{Mpower=}")
  # print(f"jPolyLike: {type(d)=} {d.dtype=} {np.unique(d,return_counts=True)=}"
        # f"\n\t {type(fate)=} {fate.dtype=} {np.unique(fate,return_counts=True)=}")

  ## the problem is still that the matrix power cannot easily be vectorized
  # for n in jnp.arange(nObs):
  #   # d = int(obs[n,1]) # fate = int(obs[n,2])
  #   d = dVal[n]
  #   fate = fateVal[n]
  #   M_to_the_d = jnp.linalg.matrix_power(M,d)  ## doesn't work w/tracers
  #   L = M_to_the_d[fate,0] # L = M_to_the_d[fate-1,0]
  #   NLL = NLL - jnp.log(L + 1e-15)
    # L = Mpower[d,fate,0]
    # def matpow(i,mat): return jnp.dot(mat,M)
    # ## creates the identity matrix and multiplies it by M, d times
    # ## incrementally changes value (see jax.lax.fori_loop documentation
    # M_to_the_d = jax.lax.fori_loop(0,d,matpow,jnp.eye(3))

    ## will print when atype==small2 (which makes debugLL=4)
    # if config.debugLL>=4: print(f"\t\tnest ID: {obs[n,0]}",end=" ")
    # if config.debugLL>=4: print(f" {fate=} {d=}")
    # print(f" {M_to_the_d=}",end=" ")
    # if config.debugLL>=4: print(f"\t\tM_to_the_d[fate-1,0] {-jnp.log(L)=} {L=}")

  # if config.debugLL>=4: print(f"\tjPolyLike: {NLL=} {type(NLL)=}")
  return NLL

def jplWrapper(pZero,dVals,fates,sclf):
  # if config.debugLL>=4: print(f"\t {type(M)=} {M.dtype=} {M.shape=}\n{M.flatten()=}")
  ## Calls jPolyLike thru ll_valgrad to get val & gradient
  loss, grad = ll_valgrad(pZero,dVals,fates,sclf)
  # print(f"{val=} {type(val)=}")
  # print(f"{grad=} {type(grad)=}")
  return float(loss), np.array(grad)

ll_valgrad = jax.value_and_grad(jPolyLike)

# def PolyMort(obsData,survey,config,useJax=False,scl_fac=0.1,plt=True,suff=""):
def PolyMort(obsData,survey,config,useJax=True,scl_fac=0.05,plt=False,suff=""):

  obs = mk_obs_mat(obsData,survey,config)
  dVal = obs[:,1].astype(jnp.int64)
  fateVal = obs[:,2].astype(jnp.int64)

  K = len(np.unique(obs[:,2])) - 1 ## number of different fates minus 1
  gtolr = 1e-6
  ftolr = 1e-6
  # opt = {'gtol':gtolr,'ftol':ftolr,'disp':True}
  opt = {'disp':True}
  fun = jPolyLike
  # met = config.optimizer
  met = "L-BFGS-B"
  nObs = int(obs.shape[0])
  # arg = (obs,nObs,scl_fac) ## extra args to pass to jplWrapper
  arg = (dVal,fateVal,scl_fac) ## extra args to pass to jplWrapper
  lb, ub = 0.001, 1.0
  bnd = optimize.Bounds(lb,ub)
  # ll_obj =  functools.partial(fun,obs=obs,nObs =nObs)
  # ll_obj =  functools.partial(fun,obs=obs,nObs =nObs,scl_fac=scl_fac)

  if useJax:
    fun = jplWrapper
    jaco = True 
    # prob = Optimized(obs=obs)
    # fun = prob.objective
    # jaco = prob.gradient
  else:
    fun = PolyLikelihood
    ll_obj =  functools.partial(fun,obs=obs,K=K,sclf=scl_fac)
    jaco=False # jaco = jax.grad(fun)

  pZero = rng.uniform(low=0.2,high=0.7,size=(K)) # print(f"untransformed {pZero=}")
  s0 = 1 - np.sum(pZero)
  # print(f"\n\t>> PolyMort: {K=} {nObs=} {type(pZero)=} {pZero.dtype=} {pZero.shape=}")
  # print(f"\t\t {type(obs)=} {obs.dtype=} {obs.shape=} ")

  ## NOTE specifying scl_fac inside of jPolyLike insteead of as arg
  # arg = (obs,nObs,)

  if plt: plot_jac(fun,arg,suff)
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

  # print(f"\t>> PolyMort: {out.success=} {out.message=} {out.nit=} {out.nfev=}")
  # print(f"*** PolyMort: {ans.dtype=} {ans.shape=} {ans=} {s=} ",end=" ")
  # hess      = np.asarray(jax.hessian(jPolyLike)(out.x,obs,nObs,scl_fac),dtype=np.float64)
  hess      = np.asarray(jax.hessian(jPolyLike)(out.x,*arg),dtype=np.float64)
  se = np.sqrt(np.diag(np.linalg.inv(hess))) * scl_fac
  ans = out.x * scl_fac
  s = 1-sum(ans) ## one minus sum of fitted values
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

