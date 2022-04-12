# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:58:24 2022.

@author: anhdu
"""
from .parallel import sample_paths_parallel
import numpy as np

class GeomBM:
    r"""The geometric Brownian motion.
    
    Parameters
    ----------
    :math:`\mu`: double
        the constant drift of :math:`X`.
    :math:`\sigma`: double
        the constant volatility of :math:`X`.
    :math:`T`: double
        the terminal timepoint up to which :math:`X` is defined.
    
    Returns
    -------
    A stochastic process :math:`X`.
    :math:`X` is characterized by the following SDE:

        .. math:: dX_t = \mu X_t dt + \sigma X_t dW_t, t\in [0,T]
    where :math:`W_t` is the standard Brownian motion.
    :math:`X` has a closed-form solution:
        
    .. math:: X_t = X_0\exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t
       + \sigma W_t\right)
    """
    
    module = 'Models'
    name = 'Geometric Brownian motion'
    
    def __init__(self,mu,sigma,T):
        self.mean   = mu
        self.vola   = sigma
        self.end_T  = T
    
    def sample_paths(
        self,
        initial_state=100,
        n_per=100,
        n_path=100,
        seed=1000,
        method=None):
        r"""Simulate a sample path of :math:`X`.

        Parameters
        ----------
        X_0: double
            initial value of :math:`X`. The default is 100.
        n_per: integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100
        n_path: integer
            number of sample paths to be simulated. The default is 100.
        seed: integer
            the seed used in sampling random variables.    
        method: None
            unused.

        Returns
        -------
        An 2d array of length (n_path,n_per + 1) containing the sample path.
        """        
        n_path = np.int64(n_path)
        n_per = np.int64(n_per)
        paths = np.zeros(
            shape=(n_path,n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        dt = self.end_T/n_per
        paths[:,0] = initial_state
        BM_samples = np.random.Generator(
            np.random.MT19937(
                np.int64(seed))).standard_normal((n_path,n_per))
        for per in np.arange(n_per, dtype = np.int64):
            paths[:,per+1] = (
                np.exp(
                    (self.mean - 1 / 2 * self.vola**2) * dt
                    + self.vola * dt 
                    * BM_samples[:,per])
                * paths[:,per]) 
        return(paths)
    
    def sample_paths_parallel(
        self,
        initial_state=100,
        n_workers = None,
        n_job=10,
        size_job=100,
        n_per=100,
        seeds=np.arange(10,dtype = np.int64),
        method=None,):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        initial_state : double
            initial point of :math:`X`. The default is 100.
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        seeds: integer
            a list of rng seeds.
            The length of the seed vector must be equal to n_job.
        n_workers: integer
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count). 
        n_job: integer
            How many 
        size_job: integer
            The number of sample paths to simulate for each job. 
            The default is 100.
        method: None
            unused.

        Returns
        -------
        A list of one 2d array of dimension (n_path, n_per + 1) containing
        simulated sample paths and one vector of size (n_per + 1) containing
        the discretization time-grid. Each sub-array [x, :] is one sample path.
        n_path = size_job * n_job
        """        
        results = sample_paths_parallel(
            model = self,
            n_workers=n_workers,
            n_job=n_job,
            size_job=size_job,
            seeds=seeds,
            method=method,
            initial_state=initial_state)
        return(results)
        
class HoLeeTimeConst:
    r"""The time-invariant HoLee model.

    Parameters
    ----------
    :math:`\mu`: double
        the constant drift of :math:`X`.
    :math:`\sigma`: double
        the constant volatility of :math:`X`.
    :math:`T`: double
        the terminal timepoint up to which :math:`X` is defined.

    Returns
    -------
    A stochastic process :math:`X`.
    :math:`X` is basically an Itô process driven by the following SDE:
        .. math:: dX_t = \mu dt + \sigma dW_t, t\in[0,T]
    where :math:`W_t` is the standard Brownian motion.
    :math:`X` has a closed-form solution:
        
    .. math:: X_t = X_0 + \mu t + \sigma W_t
    """

    module = 'Models'
    name = 'time-invariant Ho-Lee model'
    def __init__(self, mu, sigma, T):
        self.mean = mu
        self.vola = sigma
        self.end_T = T
    
    def sample_paths(
        self,
        initial_state=0.05,
        n_per=100,
        n_path=100,
        seed=1000,
        method=None):
        r"""Simulate a sample path of :math:`X`.
        
        Parameters
        ----------
        initial_state: double
            initial value of :math:`X`. The default is 0.05.
        n_per: integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100
        n_path: integer
            number of sample paths to be simulated. The default is 100.
        seed: integer
            the seed used in sampling random variables.
        method: None
            unused.


        Returns
        -------
        An 2d array of length (n_path,n_per + 1) containing the sample path.
        """
        n_path = np.int64(n_path)
        n_per = np.int64(n_per)
        paths = np.zeros(
            shape=(n_path,n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        dt = self.end_T/n_per
        paths[:,0] = initial_state
        BM_samples = np.random.Generator(
            np.random.MT19937(
                np.int64(seed))).standard_normal((n_path,n_per))
        for per in np.arange(n_per, dtype = np.int64):
            paths[:,per+1] = (
                paths[:,per] + self.mean * dt
                + self.vola * np.sqrt(dt)
                * BM_samples[:,per])
        return(paths)
    
    def sample_paths_parallel(
        self,
        initial_state=0.05,
        n_workers = None,
        n_job=10,
        size_job=100,
        n_per=100,
        seeds=np.arange(10,dtype = np.int64),
        method=None,):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        initial_state : double
            initial point of :math:`X`. The default is 0.05.
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        seeds: integer
            a list of rng seeds.
            The length of the seed vector must be equal to n_job.
        n_workers: integer
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count). 
        n_job: integer
            How many 
        size_job: integer
            The number of sample paths to simulate for each job. 
            The default is 100.
        method: None
            unused.

        Returns
        -------
        A list of one 2d array of dimension (n_path, n_per + 1) containing
        simulated sample paths and one vector of size (n_per + 1) containing
        the discretization time-grid. Each sub-array [x, :] is one sample path.
        n_path = size_job * n_job
        """
        results = sample_paths_parallel(
            model = self,
            n_workers=n_workers,
            n_job=n_job,
            size_job=size_job,
            seeds=seeds,
            method=method,
            initial_state=initial_state)
        return(results)


class VasicekTimeConst:
    r"""The time-invariant Vasicek model.
    
    Parameters
    ----------
    :math:`b`: double
        the long-run, mean-reverting level of :math:`X`
    :math:`\sigma`: double
        the constant volatility of :math:`X`.
    :math:`\alpha`: double
        the velocity of mean-reversion.
    T: double
        the terminal timepoint up to which :math:`X` is defined.

    Returns
    -------
    A stochastic process :math:`X`.
    :math:`X` is characterized by the following SDE:
        
    .. math:: dX_t = \alpha (b -X_t)dt + \sigma dW_t, t\in[0,T]
    """
    
    module = 'Models'
    name = 'time-invariant Vasicek model'
    
    def __init__(self,b,sigma,alpha,T):
        self.mean = b
        self.vola = sigma
        self.velo = alpha
        self.end_T = T
  
    def sample_paths(
        self,
        initial_state=0.05,
        n_per=100,
        n_path=100,
        seed=1000,
        method='Exact'):
        r"""Simulate a sample path of :math:`X`.
        
        Parameters
        ----------
        initial_state: double
            initial value of :math:`X`. The default is 0.05.
        n_per: integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100
        n_path: integer
            number of sample paths to be simulated. The default is 100.            
        seed: integer
            the seed used in sampling random variables.
        method: text
            the simulation method used to simulate the sample path. 
            Denote n_per as :math:`n` and let 
            :math:`(Z_i)_{i \in \{1, \ldots, n\}} \sim IID(\mathcal{N}(0,1))`.
            Options are:
                
            - 'Exact': an exact scheme which formulates as follows:
            .. math:: \hat{X}_{t_{i+1}} =\; 
                &e^{-\alpha(t_{i+1}-t_i)}\hat{X}_{t_i}
                + b\left(1 - e^{-\alpha(t_{i+1}-t_i)}\right)
                
                &+ \sqrt{\frac{\sigma^2}{2\alpha}
                \left(1-e^{-2\alpha(t_{i+1}-t_i)}\right)}Z_{i+1}
            
            - 'Euler': a 1st order approximation  the
              Euler-Maruyama method and formulates as follows:
            .. math :: \hat{X}_{t_{i+1}} = \hat{X}_{t_{i}}
                + \alpha(b - \hat{X}_{t_{i}})(t_{i+1}-t_i)
                + \sigma \sqrt{t_{i+1}-t_i} Z_{i+1}
            
        Returns
        -------
        An 2d array of length (n_path,n_per + 1) containing the sample path.
        """
        n_path = np.int64(n_path)
        n_per = np.int64(n_per)
        paths = np.zeros(
            shape=(n_path,n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        dt = self.end_T/n_per
        paths[:,0] = initial_state
        BM_samples = np.random.Generator(
            np.random.MT19937(
                np.int64(seed))).standard_normal((n_path,n_per))
        for per in np.arange(0,n_per):
            if method=='Euler':
                paths[:,per+1] = (
                    paths[:,per]
                    + self.velo * (self.mean - paths[:,per]) * dt
                    + self.vola * np.sqrt(dt)
                    * BM_samples[:,per])
            elif method=='Exact':
                paths[:,per+1] = (
                    np.exp(-self.velo * dt) * paths[:,per]
                    + self.mean * (1 - np.exp(-self.velo * dt))
                    + self.vola * np.sqrt(
                        (1 - np.exp(-2 * self.velo * dt))
                        / (2 * self.velo))
                    * BM_samples[:,per])
            else:
                raise ValueError('wrong keyword for method')
        return(paths)
    
    def sample_paths_parallel(
        self,
        initial_state=0.05,
        n_per=100,
        seed=1000,
        n_workers=None,
        n_path=100,
        method='Exact'):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        initial_state : double
            initial point of :math:`X`. The default is 0.05.
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        seeds: integer
            a list of rng seeds.
            The length of the seed vector must be equal to n_job.
        n_workers: integer
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count). 
        n_job: integer
            How many 
        size_job: integer
            The number of sample paths to simulate for each job. 
            The default is 100.
        method: text
            the method used to simulate the sample path. Denote n_per as
            :math:`n` and let :math:`(Z_i)_{i \in \{1, \ldots, n\}} 
            \sim IID(\mathcal{N}(0,1))`. Options for simulation methods are:
                
            - 'Exact': an exact scheme which formulates as follows:
            .. math:: \hat{X}_{t_{i+1}} =\; 
                &e^{-\alpha(t_{i+1}-t_i)}\hat{X}_{t_i}
                + b\left(1 - e^{-\alpha(t_{i+1}-t_i)}\right)
                
                &+ \sqrt{\frac{\sigma^2}{2\alpha}
                \left(1-e^{-2\alpha(t_{i+1}-t_i)}\right)}Z_{i+1}
            
            - 'Euler': a 1st order approximation following the
              Euler-Maruyama method and formulates as follows:
            .. math :: \hat{X}_{t_{i+1}} = \hat{X}_{t_{i}}
                + \alpha(b - \hat{X}_{t_{i}})(t_{i+1}-t_i)
                + \sigma \sqrt{t_{i+1}-t_i} Z_{i+1}
        
        Returns
        -------
        A list of one 2d array of dimension (n_path, n_per + 1) containing
        simulated sample paths and one vector of size (n_per + 1) containing
        the discretization time-grid. Each sub-array [x, :] is one sample path.
        """        
        results = sample_paths_parallel(
            model = self,
            n_workers=n_workers,
            n_path=n_path,
            n_per=n_per,
            seed=seed,
            method=method,
            initial_state=initial_state)
        return(results)
                
