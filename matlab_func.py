
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
from jax import config as jconfig
jconfig.update("jax_enable_x64", True)

def jrandom(rng_key,K,scl_fac):
  minv = scl_fac
  maxv = 10 * scl_fac
  rng_key, subkey = jax.random.split(rng_key)
  # ret = jax.random.uniform(subkey,shape=K,minval=0.1,maxval=1)
  ret = jax.random.uniform(subkey,shape=K,minval=minv,maxval=maxv)
  return ret

def jPolyLike(pZero,obs,nObs):
  # K=2
  # nObs = obs.shape[0] ## jax cannot evaluate?
  # pZero = sclf*pZero
  ## NOTE: make sure this scaling constant matches scl_fac in PolyMort
  pZero = 0.05*pZero
  # print(f"{pZero=}")

  NLL = 0 ## initial NLL value
  s0 = 1 - jnp.sum(pZero)
  m2 = 1 - s0 - pZero[0] # won't be used anyway if len(pVal) = 1
  arr = [[s0,0,0],[pZero[0],1,0],[m2,0,1]]
  # arr = [[s0,0,0],[pZero[0],1,0],[pZero[1],0,1]]
  M = jnp.array(arr,dtype=jnp.float64) # print(f"\t\t{mat=}") 
  # mat = jnp.array(arr,dtype=jnp.float64)
  if config.debugLL>=4: print(f"\t {type(M)=} {M.dtype=} {M.shape=}\n{M.flatten()=}")
  dVal = obs[:,1].astype(jnp.int64)
  fateVal = obs[:,2].astype(jnp.int64)
  # print(f"jPolyLike: {type(d)=} {d.dtype=} {np.unique(d,return_counts=True)=}"
        # f"\n\t {type(fate)=} {fate.dtype=} {np.unique(fate,return_counts=True)=}")

  ## the problem is still that the matrix power cannot easily be vectorized
  for n in jnp.arange(nObs):
    # d = int(obs[n,1])
    # fate = int(obs[n,2])
    d = dVal[n]
    fate = fateVal[n]
    # M_to_the_d = jnp.linalg.matrix_power(M,d)  ## doesn't work w/tracers
    def matpow(i,mat):
      return jnp.dot(mat,M)
    ## creates the identity matrix and multiplies it by M, d times
    ## incrementally changes value (see jax.lax.fori_loop documentation
    M_to_the_d = jax.lax.fori_loop(0,d,matpow,jnp.eye(3))
    
    L = M_to_the_d[fate,0] # L = M_to_the_d[fate-1,0]
    NLL = NLL - jnp.log(L + 1e-15)
  # for n in range(nObs):
  # for n in obsStep:
    ## will print when atype==small2 (which makes debugLL=4)
    if config.debugLL>=4: print(f"\t\tnest ID: {obs[n,0]}",end=" ")
    # print(f" {fate[n]=} {d[n]=}")
    if config.debugLL>=4: print(f" {fate=} {d=}")
    # M_to_the_d = jnp.linalg.matrix_power(M,d[n]) 
    # print(f" {M_to_the_d=}",end=" ")
    # L = M_to_the_d[fate[n]-1,0]
    # L = M_to_the_d[fate[n],0]
    if config.debugLL>=4: print(f"\t\tM_to_the_d[fate-1,0] {-jnp.log(L)=} {L=}")

  if config.debugLL>=4: print(f"\tjPolyLike: {NLL=} {type(NLL)=}")
  return NLL

## define globally for some reason?
valgrad =   jax.jit(jax.value_and_grad(jPolyLike), static_argnums=2 )
hessFun =   jax.jit(jax.hessian(jPolyLike), static_argnums=2 )

def PolyMort_new(obsData,survey,config,scl_fac=0.05,suff=""):
  obs = mk_obs_mat(obsData,survey,config)
  K = len(np.unique(obs[:,2])) - 1 ## number of different fates minus 1
  obs = jnp.asarray(obs,dtype=jnp.float64)
  gtolr = 1e-8
  nObs = int(obs.shape[0]) 
  pZero = rng.uniform(low=0.1,high=0.9,size=(K)) # print(f"untransformed {pZero=}")
  s0 = 1 - np.sum(pZero)
  print(f"PolyMort: {K=} {type(pZero)=} {pZero.dtype=} {pZero.shape=}",end=" ")
  print(f"\t\t {type(obs)=} {obs.dtype=} {obs.shape=} ")

  def obj_fun(x):
    val, _ = valgrad(x, obs, nObs)
    return float(val)

  def grad_fun(x):
    _, grad = valgrad(x, obs, nObs)
    ## scipy requires flattened input?
    return np.asarray(grad, dtype=np.float64).flatten()

  lb = 0.00001
  ub = 1.0
  bnd = optimize.Bounds(lb,ub)
  met = config.optimizer
  ## don't need args because func being minimized is obj_fun?

  out = optimize.minimize(obj_fun,
                          pZero,
                          method = met,
                          bounds = bnd, # constraints = con,
                          jac=grad_fun,
                          options={'gtol':gtolr, 'disp':True},
                          # options={'disp':True},
                          )
  print(f">> PolyMort: {out.success=} {out.message=} {out.nit=} {out.nfev=}")
  ans = out.x * scl_fac
  s = 1-sum(ans) ## one minus sum of fitted values
  print(f"*** PolyMort: {ans.dtype=} {ans.shape=} {ans=} {s=} ",end=" ")

  hess = np.asarray(hessFun(pZero,obs,nObs),dtype=np.float64)
  se = np.sqrt(np.diag(np.linalg.inv(hess))) * scl_fac
  seS = np.sqrt(np.sum(np.sum(np.linalg.inv(hess)))) * scl_fac
  print(f"{se=} {seS=}",end=" ")
  print(f" {hess.flatten()=}")
  return (s, seS, ans[0], se[0])


# optimizer = jaxopt.LBFGSB(fun=jPolyLike) ##+> should automatically calculate gradients

# def run_optim(pZero,bnd,obs):
# @jax.jit
# def run_optim(pZero,obs,lb,ub):
#
#   # bnd = (jnp.array([0.00001,0.00001]),jnp.array([1.0,1.0]))
#   bnd = (lb,ub)
#   solvstate = optimizer.run(pZero, bounds=bnd, obs=obs)
#   hess = jax.hessian(jPolyLike)(solvstate.params,obs)
#   se = jnp.sqrt(jnp.diag(jnp.linalg.inv(hess)))
#   seS = jnp.sqrt(jnp.sum(jnp.sum(jnp.linalg.inv(hess))))
#
#   return solvstate.params, hess, se, seS

# NOTE try evaluating everything inside jax - doesn't use class Optimized
# NOTE can't because of the for loop and indexing
# def jPolyMort(obsData,survey,config,scl_fac=0.05,plt=False,suff=""):
#
#   obs = mk_obs_mat(obsData,survey,config) 
#   obs = jnp.array(obs)
#   K = len(np.unique(obs[:,2])) - 1 ##+> number of different fates minus 1
#
#   pZero = rng.uniform(low=0.1,high=0.9,size=(K)) # print(f"untransformed {pZero=}")
#   # bnd = (jnp.array([0.00001,0.00001]),jnp.array([1.0,1.0]))
#   lb = jnp.array([0.00001,0.00001])
#   ub = jnp.array([1.0,1.0])
#   if K==1:
#     print("*** single value in pZero")
#     ## zero-pad if only one input val so shape is same
#     pZero = np.array([pZero[0],0.0]) 
#     ## change second bound to 0 and 0 when using padded array
#     # bnd = (jnp.array([0.00001,0.0]),jnp.array([1.0,0.0]))
#     lb = jnp.array([0.00001,0.0])
#     ub = jnp.array([1.0,0.0])
#
#   # s0 = 1 - np.sum(pZero)
#
#   print(f"PolyMort: {K=}",end=" ")
#   print(f"\t\t {type(obs)=} {obs.dtype=} {obs.shape=} ")
#
#   pZero = jnp.array(pZero)
#   # ans, hess = run_optim(pZero,bnd,obs)
#   print(f">> PolyMort: run jaxopt.LBFGSB optimizer - untransformed {pZero=} {lb=} {ub=}")
#   ans, hess, se, seS = run_optim(pZero,obs,lb,ub)
#   ans = ans * scl_fac
#   se = se * scl_fac
#   seS = seS * scl_fac
#   s = 1-sum(ans) ## one minus sum of fitted values
#   print(f"*** PolyMort: {ans.dtype=} {ans.shape=} {ans=} {s=} ",end=" ")
#
#   # se = np.sqrt(np.diag(np.linalg.inv(hess))) * scl_fac
#   # seS = np.sqrt(np.sum(np.sum(np.linalg.inv(hess)))) * scl_fac
#   ## NOTE: do I need to multiple se by the scaling factor?
#   print(f"{se=} {seS=}",end=" ")
#   print(f" {hess.flatten()=}")
#
#   return(s, seS, ans[0], se[0])
#
#
# # ll_valgrad = jax.value_and_grad(jPolyLike) ## value & gradient function
# #   return out, jax.grad(lambda x

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

  if config.testing=="yes":
    if config.debugLL>=2:
      print(f"mk_obs_mat: input matrix for optim:")
      dfPrint(out)
  # out = jnp.asarray(out)
  print(f"\t\tmk_obs_mat: {type(out)=} {np.isnan(out).sum(axis=0)=}")

  return out


# @jax.jit
def ll_valgrad(x,obs): ## obs is passed from w/in the Class instance
  val, grad = jax.value_and_grad(jPolyLike)(x,obs) 
  hess      = jax.hessian(jPolyLike)(x,obs)
  return val, grad, hess

class Optimized:
  ## should allow to make calculations once and then reference from
  ## the two separate functions for value and gradient?
  def __init__(self, obs):
    self.obs = obs
    self.cached_x = None
    self.cached_val = None
    self.cached_grad = None
    self.cached_hess = None

  # def jplCompute(self,pZero,obs):
  def jplCompute(self,pZero):
    pZero = np.asarray(pZero, dtype=np.float64).flatten()
    if self.cached_x is not None and np.array_equal(pZero, self.cached_x):
            return
    # loss, grad = ll_valgrad(pZero,obs)
    loss, grad, hess = ll_valgrad(pZero,self.obs)
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

# def PolyMort(obsData,survey,config,useJax=False,scl_fac=0.1,plt=True,suff=""):
def PolyMort(obsData,survey,config,useJax=True,scl_fac=0.05,plt=False,suff=""):
# def PolyMort(obsData,survey,config,useJax=False,scl_fac=0.05):

  obs = mk_obs_mat(obsData,survey,config)
  K = len(np.unique(obs[:,2])) - 1 ## number of different fates minus 1
  gtolr = 1e-8
  nObs = int(obs.shape[0]) 

  if useJax:
    ## better to convert to jax w/in wrapper bc scipy takes np arrays, not jax
    # pZero = jrandom(rng_key,K,scl_fac)
    # s0 = 1 - jnp.sum(pZero)

    # fun = jplWrapper
    # jaco = True 
    prob = Optimized(obs=obs)
    fun = prob.objective
    jaco = prob.gradient

  else:
    # pZero = rng.uniform(0.01,1,size=(K)) # print(f"untransformed {pZero=}")
    # pZero = rng.uniform(low=0.1,high=0.9,size=(K)) # print(f"untransformed {pZero=}")
    # s0 = 1 - np.sum(pZero) m0 = rng.uniform(0,0.4) m1 = rng.uniform(0,0.2)
    # pZero = 0.1 * np.array([m0,m1])
    fun = PolyLikelihood
    ll_obj =  functools.partial(fun,obs=obs,K=K,sclf=scl_fac)
    # jaco = ndt.Jacobian(lambda x: fun(x,obs))
    jaco=False # jaco = jax.grad(fun)

  pZero = rng.uniform(low=0.1,high=0.9,size=(K)) # print(f"untransformed {pZero=}")
  s0 = 1 - np.sum(pZero)
  print(f"PolyMort: {K=} {type(pZero)=} {pZero.dtype=} {pZero.shape=}",end=" ")
  print(f"\t\t {type(obs)=} {obs.dtype=} {obs.shape=} ")
  # fun1 = PolyLikelihood # print("creating partial function")

  # bnd = [(0.00001,0.1),(0.00001,0.1)] bnd = np.repeat((0.00001,0.1),K)
  # bnd = (0.00001,0.1)
  lb = 0.00001
  ub = 1.0
  # lb = [lb]*K
  # ub = [ub]*K
  # bnd = optimize.Bounds([lb]*K,[ub]*K)
  bnd = optimize.Bounds(lb,ub)
  # bnd = (0.00001,1)
  # bnd = [bnd] * K
  met = config.optimizer
  # arg = (obs,K,scl_fac,)
  ## NOTE specifying scl_fac inside of jPolyLike insteead of as arg
  arg = (obs,)
  # print(f">> PolyMort: run optimizer - {met=} {pZero=} {fun=} {jaco=} {bnd=}")
  if plt: plot_jac(fun,arg,suff)

  print(f">> PolyMort: run optimizer - {met=} {gtolr=}; untransformed {pZero=}\n{fun=} {bnd=}")

  ## NOTE can use keep_feasible w/trust-constr to stay w/in bounds throughout
  ## NOTE don't need to pass arg if obs is in the class instance
  out = optimize.minimize(fun,
                          pZero,
                          # args=arg,
                          method = met,
                          # jac=jaco,
                          # jac=ndt.Jacobian(lambda x: fun(x,obs)),
                          # hess=ndt.Hessian(lambda x: fun(x,obs)),
                          bounds = bnd, # constraints = con,
                          # options={'gtol':1e-12, 'disp':True},
                          # options={'gtol':gtolr, 'disp':True},
                          options={'disp':True},
                 )

  print(f">> PolyMort: {out.success=} {out.message=} {out.nit=} {out.nfev=}")
  ans = out.x * scl_fac
  s = 1-sum(ans) ## one minus sum of fitted values
  print(f"*** PolyMort: {ans.dtype=} {ans.shape=} {ans=} {s=} ",end=" ")

  prob.jplCompute(ans)
  hess = np.asarray(prob.cached_hess, dtype=np.float64)
  # hess = ndt.Hessian(ll_obj,method="complex")(ans) # hess2 = approx_hess2(out.x, ll_obj)
  # hess = ndt.Hessian(ll_obj,step=1e-12)(ans) # hess2 = approx_hess2(out.x, ll_obj)
  se = np.sqrt(np.diag(np.linalg.inv(hess))) * scl_fac
  seS = np.sqrt(np.sum(np.sum(np.linalg.inv(hess)))) * scl_fac
  print(f"{se=} {seS=}",end=" ")
  print(f" {hess.flatten()=}")

  return (s, seS, ans[0], se[0])

# def PolyLikelihood(pZero,obs,M):
def PolyLikelihood(pZero,obs,K,sclf):
  # K = 2
  nObs = obs.shape[0] # print(f"{nObs=}")
  pZero = sclf*pZero
  s0 = 1 - np.sum(pZero)
  m2 = 1 - s0 - pZero[0] # won't be used anyway if len(pVal) = 1
  arr = [[s0,0,0],[pZero[0],1,0],[m2,0,1]]
  # arr = [[s0,0,0],[pZero[0],1,0],[pZero[1],0,1]]
  ## need to calculate here bc it's not contant (pZero changes)
  M = np.array(arr,dtype=np.float64) 
  # if config.debugLL>=4: print(f"\t>> PolyLikelihood: {nObs=} {M.dtype=} {M.shape=} {M.flatten()=}")
  NLL = 0 ## initial NLL value

  for n in range(nObs):
    d = obs[n,1] # print(f"{d=}",end=" ")
    fate = obs[n,2] # print(f" {fate=}")
    # if config.debugLL>=4: print(f"\t\tPolyLikelihood: nest ID: {obs[n,0]} {d=} {fate=}",end=" ")
    M_to_the_d = np.linalg.matrix_power(M,d) # print(f"\t\t\t {M_to_the_d.flatten()=}",end=" ")
    L = M_to_the_d[fate,0]
    # if config.debugLL>=4: print(f"\t\t M_to_the_d[fate,0] {L=}")
    NLL = NLL - np.log(L)

  # if config.debugLL>=4: print(f"\t>> PolyLikelihood: {NLL=}")
  return NLL

# def jplWrapper(pZero,obs,K,sclf):
def jplWrapper(pZero,obs):
  jpZero = jnp.array(pZero)
  # loss, grad = ll_valgrad(pZero,obs,K,sclf)
  loss, grad = ll_valgrad(jpZero,obs)
  val = float(loss)
  grad = np.array(grad,dtype=np.float64)
  print(f"{val=} {type(val)=}")
  print(f"{grad=} {type(grad)=}")
  # return val, grad
  # return val, grad
  return float(loss), np.array(grad)


def j_obs_mat(obsData,survey,config,exp1=False):
  if isinstance(obsData, pd.DataFrame):
    # obsData = obsData.to_numpy()
    obsData = jnp.array(obsData.to_numpy())
  elif not isinstance(obsData, jnp.ndarray):
    obsData = jnp.array(obsData)
  print(f"j_obs_mat: input data: {obsData=}")
  nNest = obsData.shape[0]
# colnames = c('ID', 'init', 'end', 'fate', 'i', 'j', 'k', 'afate', 'nobs', 'fint', 'totobs', 'sint')
  print(f"j_obs_mat: unpack input data")
  ID, init,end,tfate,ff, la, lc, afate,nObs = obsData.T
  nrows    = int(jnp.sum(nObs))
  nObs = jnp.array(nObs)
  nObs = nObs.astype(jnp.int64)
  if exp1:
    first,last,fate = init,end,tfate
  else:
    first,last,fate = ff,la,afate
  print(f"j_obs_mat: calculate exposure days")
  expos, obsDay = calc_daily_expo(nNest,survey,first,last,config)
  endDay = jnp.cumsum(nObs) -1 #+> zero-indexed
  endDay = endDay.astype(jnp.int64)
  print(f"j_obs_mat: {endDay=} {endDay.dtype=}")
  print(f"j_obs_mat: {type(endDay)=} {type(nrows)=} {type(nObs)=} {type(nNest)=}")

  IDcol = jnp.repeat(ID,nObs)
  print(f"j_obs_mat: {IDcol=} {IDcol.dtype=}")
  fatecol = jnp.ones(nrows,dtype=int)
  fatecol = fatecol.at[endDay]=fate + 1
  print(f"j_obs_mat: {fatecol=} {fatecol.dtype=}")

  out = jnp.column_stack([IDcol,expos,fatecol])
  ##+> check for NAs:
  print(f"\t\tmk_obs_mat: {type(out)=} {np.isnan(out).sum(axis=0)=}\n {out[0:8,:]=}")
  return out

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

