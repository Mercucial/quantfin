# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:10:06 2021

@author: anhdu
"""
import numpy as np
from multiprocessing import Pool
from itertools import repeat
            
def sample_paths(
        model,
        n_workers = None,
        n_path=100,
        n_per=100,
        seed=1000,
        method=None,
        **kwargs):
    r"""Simulate multiple sample paths (helper function).
    
    Parameters
    ----------
    model: class instance
        A stochastic model.
    n_workers: integer, optional
        Number of workers to parallelize the simulation process. The default 
        is None. In this case, the number of workers is the number of
        processors (i.e. CPU core count).
    n_path: integer
        The number of sample paths to simulate. The default is 100
    n_per: integer
        number of intervals used to discretize the time interval
        :math:`[0,T]`. The discretized time grid is equidistant. 
        The default is 100.
    seed: integer
        rng seed. Each increment uses a seed larger than the previous seed by
        10. Each sample path uses a vector of seeds larger than the one
        of the preceding path by 10. The default is 1000.
    method: text
        Simulation method. For details check the sample_path method under
        each class
    **kwargs: double
        The initial value(s) of the stochstic process(es). Must be
        X_0 for Ho-Lee, Vasicek and CIR models and (S_0,V_0) for 
        the Heston model.

    Returns
    -------
    For CIR, Vasicek and Ho-Lee model:
    A list of one 2d array of dimension (n_path, n_per + 1) containing
    simulated sample paths and one vector of size (n_per + 1) containing
    the discretization time-grid. Each sub-array [x, :] is one sample path.
    
    For Heston model:
    A list of one 3d array of dimension (n_path, n_per + 1, 4) containing the
    simulated sample paths and one vector of size (n_per + 1) containing the
    discretization time-grid. Each sub-array [x, :, :] is one sample path.
    
    """
    times=np.linspace(
        start=0,
        stop=model.end_T,
        endpoint=True,
        num = n_per+1)
    start_seeds = np.linspace(
        start = np.int64(seed),
        stop = seed + (np.int64(n_path) - 1) * 10,
        num = np.int64(n_path),
        endpoint = True,
        dtype = np.int64)
    # test code
    if type(model).__name__ in ('CIR','HoLeeTimeConst','VasicekTimeConst'):
        X_0 = kwargs.get('X_0')
        params = zip(
            repeat(X_0),
            repeat(n_per),
            start_seeds,
            repeat(method))
        with Pool(processes = n_workers) as pool:
            paths = np.transpose(
                np.stack(
                    pool.starmap(model.sample_path,params),
                    axis = 1))
    elif type(model).__name__ == 'Heston':
        S_0 = kwargs.get('S_0')
        V_0 = kwargs.get('V_0')
        params = zip(
            repeat(S_0),
            repeat(V_0),
            repeat(n_per),
            start_seeds,
            repeat(method))
        with Pool(processes = n_workers) as pool:
            paths = np.transpose(
                np.stack(
                    pool.starmap(model.sample_path,params),
                    axis = 2))

    return([times,paths])

# Each class is a separate models used in quant fin (Vasicek,
# Ho-Lee, Heston etc)
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
    :math:`X` is basically an Ito process following the SDE below:
        .. math:: dX_t = \mu dt + \sigma dW_t, t\in[0,T]
    """

    name = 'time-invariant Ho-Lee model'
    def __init__(self, mu, sigma, T):
        self.mean = mu
        self.vola = sigma
        self.end_T = T
    
    def sample_path(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        method=None):
        r"""Simulate a sample path of :math:`X`.
        
        Parameters
        ----------
        seed: integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10.
        method: text
            syntax sugar. Normally the method used to simulate
            the sample path is defined here, but for the
            time-invariant Ho-Lee model, an exact simulation will
            suffice and will be used
        n_per: integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
        X_0: double
            initial point of X
        Returns
        -------
        An 1d array of length n_per + 1 containing the sample path.
        """
        path = np.zeros(
            shape=(n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        seeds = np.linspace(
            start=seed,
            stop=seed+10*(n_per-1),
            num=n_per,
            endpoint=True,
            dtype=np.int64)
        dt = self.end_T/n_per
        path[0] = X_0
        for per in np.arange(0,n_per):
            path[per+1] = (
                path[per] + self.mean * dt
                + self.vola * np.sqrt(dt)
                * np.random.Generator(
                    np.random.MT19937(
                        np.int64(seeds[per])))
                .standard_normal())
        return(path)
    
    def sample_paths(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        n_workers=None,
        n_path=100,
        method=None):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        X_0 : double
            initial point of X. The default is 0.05.
        seed : integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10. The default is 1000
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count).        
        n_path: integer
            The number of sample paths to simulate. The default is 100
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        method : text
            syntax sugar. Normally the method used to simulate
            the sample path is defined here, but for the
            time-invariant Ho-Lee model, an exact simulation will
            suffice.

        Returns
        -------
        A list of one 2d array of dimension (n_path, n_per + 1) containing
        simulated sample paths and one vector of size (n_per + 1) containing
        the discretization time-grid. Each sub-array [x, :] is one sample path.
        """
        results = sample_paths(
            model = self,
            n_workers=n_workers,
            n_path=n_path,
            n_per=n_per,
            seed=seed,
            method=method,
            X_0=X_0)
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
        the terminal timepoint up to which
        X is defined.

    Returns
    -------
    A stochastic process :math:`X`.
    :math:`X` is characterized by the following SDE:
        
    .. math:: dX_t = \alpha (b -X_t)dt + \sigma dW_t, t\in[0,T]
    """
    
    name = 'time-invariant Vasicek model'
    
    def __init__(self,b,sigma,alpha,T):
        self.mean = b
        self.vola = sigma
        self.velo = alpha
        self.end_T = T
  
    def sample_path(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        method='Exact'):
        r"""Simulate a sample path of :math:`X`.
        
        Parameters
        ----------
        X_0: double
            initial value of X. The default is 0.05
        n_per: integer
            number of intervals used to discretize the time interval
            [0,endT]. The discretized time grid is equidistant. THe default
            is 100
        seed: integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10. The default is 1000
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
            
            - 'Euler': a 1st order approximation following the
              Euler-Maruyama method and formulates as follows:
            .. math :: \hat{X}_{t_{i+1}} = \hat{X}_{t_{i}}
                + \alpha(b - \hat{X}_{t_{i}})(t_{i+1}-t_i)
                + \sigma \sqrt{t_{i+1}-t_i} Z_{i+1}
            
        Returns
        -------
        An 1d array of length n_per + 1 containing the sample path.
        """
        path = np.zeros(
            shape=(n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        seeds = np.linspace(
            start=seed,
            stop=seed+10*(n_per-1),
            num=n_per,
            endpoint=True,
            dtype=np.int64)
        dt = self.end_T/n_per
        path[0] = X_0
        for per in np.arange(0,n_per):
            if method=='Euler':
                path[per+1] = (
                    path[per]
                    + self.velo * (self.mean - path[per]) * dt
                    + self.vola * np.sqrt(dt)
                    * np.random.Generator(
                        np.random.MT19937(
                            np.int64(seeds[per])))
                    .standard_normal())
            elif method=='Exact':
                path[per+1] = (
                    np.exp(-self.velo * dt) * path[per]
                    + self.mean * (1 - np.exp(-self.velo * dt))
                    + self.vola * np.sqrt(
                        (1 - np.exp(-2 * self.velo * dt))
                        / (2 * self.velo))
                    * np.random.Generator(
                        np.random.MT19937(
                            np.int64(seeds[per])))
                    .standard_normal())
            else:
                raise ValueError('wrong keyword for method')
        return(path)
    
    def sample_paths(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        n_workers=None,
        n_path=100,
        method='Exact'):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        X_0 : double
            initial point of X. The default is 0.05.
        seed : integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10. The default is 1000
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count).        
        n_path: integer
            The number of sample paths to simulate. The default is 100
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
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
        results = sample_paths(
            model = self,
            n_workers=n_workers,
            n_path=n_path,
            n_per=n_per,
            seed=seed,
            method=method,
            X_0=X_0)
        return(results)
                
class CIR:
    r"""The Cox-Ingersoll-Rox model.
    
    Parameters
    ----------
    :math:`a`: double
        the long-run mean-reverting level of X
    :math:`\sigma`: double
        the constant volatility of X.
    :math:`k`: double
        the velocity of mean-reversion.
    :math:`T`: double
        the terminal timepoint up to which
        X is defined.
        
    Returns
    -------
    An instance of a stochastic process X following the Cox-Ingersoll-Rox
    model. X is characterized by the following SDE:
        
    .. math:: dX_t = k(a-X_t)dt + \sigma\sqrt{X_t}dW_t, t\in[0,T]
    """
    
    name = 'Cox-Ingersoll-Rox model'
    def __init__(self,a,sigma,k,T):
        self.mean = a
        self.vola = sigma
        self.velo = k
        self.end_T = T
    
    def __zeta(self,k,t):
        r"""Return the value of the auxilliary zeta function.
        
        Parameters
        ----------
        k : double
        
        t : double
        
        Returns
        -------
        :math:`\zeta_{k}(t) = \frac{1}{k}(1-e^{-kt})` if :math:`k\not=0`
        
        :math:`\zeta_{k}(t) = t` if :math:`k=0`
        """
        if k != 0:
            return(1/k*(1-np.exp(-k*t)))
        else:
            return(t)
        
    def __X0(self,x,t):
        r"""Generate sample path.
        
        Part of the 3rd potential order scheme.
        
        Parameters
        ----------
        x: double
            value of the prior step in the simulation scheme
        t: double
            size of the timestep
            
        Returns
        -------
        The next step in the simulation scheme:
            .. math:: X_0(x) = x + (a - \sigma^2/4) \zeta_{-k}(t)
        """
        result = x + (self.mean - self.vola**2/4) * self.__zeta(-self.velo,t)
        return (result)
    
    def __X1(self,x,t,seed):
        r"""Generate sample path.
        
        Part of the 3rd potential order scheme.

        Parameters
        ----------
        x: double
            value of the prior step in the simulation scheme.
        t: double
            size of the timestep.
        seed: integer
            seed for rng.

        Returns
        -------
        The next step in the simulation scheme:
            .. math:: X_1(x) = (\sqrt{x} + \sigma\sqrt{\zeta_{-k}(t)}Y/2)^2
        where Y is a random variable with the following distribution:
            .. math:: 
                \mathbb{P}\left[Y = \sqrt{3}\right] 
                = \mathbb{P}\left[Y = -\sqrt{3}\right] = 1/6,\;
                \mathbb{P}\left[Y = 0\right] = 2/3
        """
        Y = (
            np.random.Generator(np.random.MT19937(np.int64(seed)))
            .choice([-np.sqrt(3),0,np.sqrt(3)],1,p=[1/6,2/3,1/6]))
        result = (
            np.sqrt(x)
            + self.vola * np.sqrt(self.__zeta(-self.velo, t)) * Y / 2)**2
        return (result)  
    
    def __Xtilde(self, x, t, seed):
        r"""Generate sample path.
        
        Part of the 3rd potential order scheme.

        Parameters
        ----------
        x: double
            value of the prior step in the simulation scheme.
        t: double
            size of the timestep.
        seed: integer
            seed for rng.

        Returns
        -------
        The next step in the simulation scheme:
            .. math:: 
                \tilde{X}(x) = x + \sigma/\sqrt{2}
                \sqrt{\vert a - \sigma^2/4 \vert}
                \epsilon \zeta_{-k}(t)
        
        where :math:`\epsilon` is a random variable with the following 
        distribution:
            .. math::
                \mathbb{P}\left[\epsilon = 1\right]
                =\mathbb{P}\left[\epsilon = -1\right]
                = 1/2
        """
        eps = (
            np.random.Generator(np.random.MT19937(np.int(seed)))
            .choice([1,-1],1, p=[0.5,0.5]))
        result = (x 
                  + self.vola/np.sqrt(2)
                  * np.sqrt(np.abs(self.mean - self.vola**2/4))
                  * eps * self.__zeta(-self.velo, t))
        return(result)
            
    def moment(self,x,q,t):
        r"""Compute the central moments of :math:`X_t`.
        
        Parameters
        ----------
        q: integer
            The order of moment (k = 1, 2, 3). Non-integer value will be
            truncated to nearest integer.
        t: double
            The time point
        x: double
            The initial value of process :math:`X` at time 0
        Returns
        -------
        :math:`\mathbb{E}\left[(X_t^x)^s\right]`, 
        for :math:`s \leq q`

        """
        order = np.trunc(q)
        if order < 1 or order > 3 :
            raise ValueError('k must be in the interval [1,4)')
        else:
            moment1 = (
                x*np.exp(-self.velo*t)
                + self.mean*self.__zeta(self.velo, t))
        if order >= 2:
            moment2 =(
                (moment1)**2
                + self.vola**2*self.__zeta(self.velo,t)
                * (
                    self.mean*self.__zeta(self.velo,t)/2
                    + x*np.exp(-self.velo*t)))
        
        if order == 3:
            moment3 = (
                moment1 * moment2
                + self.vola**2 * self.__zeta(self.velo, t)
                * (2 * x**2 * np.exp(-2 * self.velo * t)
                   + self.__zeta(self.velo, t)
                   * (self.mean + self.velo**2 / 2)
                   * (
                       3 * x * np.exp(-self.velo * t)
                       + self.mean * self.__zeta(self.velo, t))))
            return([moment3, moment2, moment1])
        elif order==2:
            return([moment2, moment1])
        else: #order ==1
            return(moment1)

    def sample_path(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        method='Alfonsi2'):
        r"""Simulate a sample path of :math:`X`.
        
        Parameters
        ----------
        X_0: double
            initial value of X. The default is 0.05
        n_per: integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100
        seed: integer
            the seed used in sampling the Gaussian increment. Each
            increment uses a seed larger than the previous seed by 10. The
            default is 1000
        method: text
            the simulation method used to simulate the sample path. 
            Denote n_per as :math:`n`. Options are:
                
            - Exact: an exact scheme using non-central chi square distribution
            (implementation still in progress).
            
            - Brigo-Alfonsi: an implicit Euler-Maruyama scheme. Well-defined
              when :math:`\sigma^2\geq 2a` and :math:`1+kT/n>0`:
            .. math:: \hat{X}_{t_{i+1}} =  \hat{X}_{t_{i}} + \
              (a-k\hat{X}_{t_{i+1}}-\frac{\sigma^2}{2})\frac{T}{n} + \
              \sigma \sqrt{\hat{X}_{t_{i+1}}}(W_{t_{i+1}}-W_{t_i})
              
            If these conditions are not satistifed, the function will raise
            the ValueError
            
            - Daelbaen-Deelstra: a modified explicit Euler-Maruyama scheme:
            .. math::
                \hat{X}_{t_{i+1}} = \hat{X}_{t_{i}} + (a-k\hat{X}_{t_{i}}) \
                \frac{T}{n} + \sigma \sqrt{(\hat{X}_{t_{i}})^+} \
                (W_{t_{i+1}}-W_{t_i})
            - Lord: another modified explicit Euler-Maruyama scheme:
            .. math::
                \hat{X}_{t_{i+1}}=\hat{X}_{t_{i}} + (a-k(\hat{X}_{t_{i}})^+)\
                \frac{T}{n} + \sigma \sqrt{(\hat{X}_{t_{i}})^+} \
                (W_{t_{i+1}}-W_{t_i})
            - Alfonsi2: Alfonsi's 2nd order potential scheme. For details
              see [1].
            - Alfonsi3: Alfonsi's 3rd order potential scheme. For details
              see [1].
              
        Returns
        -------
        An 1d array of length n_per + 1 containing the sample path.

        References
        ----------
        [1] Alfonsi, Aurélien. "High order discretization schemes for the
        CIR process: application to affine term structure and Heston models."
        Mathematics of Computation 79.269 (2010): 209-237.

        [2] Deelstra, G., Delbaen, F.: Convergence of discretized stochastic
        (interest rate) processes with stochastic drift term.
        Appl. Stoch. Models Data Anal. 14(1), 77–84 (1998)

        [3] Lord, R., Koekkoek, R., Van Dijk, D.: A comparison of biased
        simulation schemes for stochastic volatility models.
        Quant. Finance 10(2), 177–194 (2010)
        
        [4] Brigo, D., Alfonsi, A.: Credit default swap calibration and 
        derivatives pricing with the SSRD stochastic intensity model. 
        Financ. Stoch. 9(1), 29–42 (2005)
        """
        path = np.zeros(
            shape=(n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        seeds = np.linspace(
            start=seed,
            stop=seed+10*(n_per-1),
            num=n_per,
            endpoint=True,
            dtype=np.int64)
        dt = self.end_T/n_per
        path[0] = X_0
        if not method in ['Exact','Brigo-Alfonsi','Daelbaen-Deelstra',
                          'Lord','Alfonsi2','Alfonsi3']:
            raise ValueError('wrong keyword for method')
        elif method in ['Brigo-Alfonsi','Daelbaen-Deelstra','Lord']:
            BM_increments = (
                np.random.Generator(np.random.MT19937(np.int64(seeds)))
                .standard_normal())
            if method=='Brigo-Alfonsi':
                if self.vola**2>=2*self.mean and 1 + self.velo*dt > 0:
                    for per in np.arange(0,n_per+1):
                        path[per+1] = (
                    (self.vola*BM_increments[per]
                     + np.sqrt(
                         self.vola**2*BM_increments[per]**2
                         + 4 
                         * (path[per]
                            + (self.velo - self.vola**2)*dt)
                         * (1 + self.velo*dt)))
                    / (2*(1+self.velo*dt)))**2
                else: raise ValueError('unsuitable parameters')
            elif method=='Daelbaen-Deelstra':
                for per in np.arange(0,n_per):
                    path[per+1] = (
                path[per]
                + (self.mean - self.velo*path[per]) * dt 
                + self.vola
                * np.sqrt(max(path[per],0))
                * BM_increments[per])
            elif method=='Lord': 
                for per in np.arange(0,n_per):
                    path[per+1]  = (
                path[per] 
                + (self.mean - max(self.velo*path[per],0)) * dt
                + self.vola
                    *np.sqrt(max(path[per],0))
                    *BM_increments[per])
        elif method == 'Alfonsi2':
            if self.vola**2 <= 4*self.mean:
                threshold = 0
                threshold = 0
            else:
                threshold = (
            np.exp(self.velo*dt/2) 
            * (
                self.__zeta(self.velo, dt/2)*(self.vola**2/4-self.mean)
                + (
                    np.sqrt(
                        np.exp(self.velo*dt/2)
                        * self.__zeta(self.velo, dt/2)
                        *(self.vola**2 / 4 - self.mean))
                    + self.vola/2*np.sqrt(3*dt))**2))
            for per in np.arange(0,n_per):
                if path[per] >=threshold:
                    path[per+1] = (
                np.exp(-self.velo*dt/2)
                * (
                    np.sqrt(
                    (self.mean - self.vola**2/4)
                    * self.__zeta(self.velo, dt/2)
                    + np.exp(-self.velo*dt/2)
                    * path[per])
                    + self.vola / 2
                    * np.random.Generator(
                        np.random.MT19937(
                            np.int64(seeds[per])))
                    .choice(
                        [-np.sqrt(3),0,np.sqrt(3)],
                        1,
                        p=[1/6,2/3,1/6]))**2
                + (self.mean - self.vola**2/4)
                * self.__zeta(self.velo,dt/2))
                else:
                    u1,u2 = self.moment(path[per],2,dt)
                    p = (1 - np.sqrt(1 - u1**2/u2))/2
                    U = (
                        np.random.Generator(
                            np.random.MT19937(
                                np.int64(seeds[per])))
                        .uniform())
                    if U < p:
                        path[per+1] = u1/(2*p)
                    else:
                        path[per+1] = u1/(2*(1-p))
        elif method=='Alfonsi3':
            if self.vola**2 <= 4 * self.mean/3:
                threshold = (
                    self.vola/np.sqrt(2)
                    * np.sqrt(self.mean - self.vola**2/4))
            elif (self.vola**2 > 4 * self.mean/3 
                and self.vola**2 <= 4*self.mean):
                threshold = (
                    np.sqrt(
                        self.vola**2/4 - self.mean 
                        + self.vola/np.sqrt(2) 
                        * np.sqrt(self.mean - self.vola**2/4))
                    + self.vola/2 * np.sqrt(3 + np.sqrt(6)))
            else: #self.vola**2 > 4* self.mean
                threshold = (
                    self.vola**2 - 4 * self.mean
                    + (
                        np.sqrt(self.vola/np.sqrt(2)
                                * np.sqrt(self.vola**2/4 - self.mean))
                        + self.vola / np.sqrt(2)
                        * np.sqrt(3 + np.sqrt(6))))
       
            for per in np.arange(0,n_per):
                if (path[per] >= 
                    self.__zeta(-self.velo, dt) * threshold):
                    z = (
                        np.random.Generator(
                            np.random.MT19937(
                                np.int64(np.int64(seeds[per]))))
                        .choice([1,2,3],1,p=[1/3,1/3,1/3]))
                    if z == 1:
                        if self.vola**2 <= 4 * self.mean:
                            path[per+1] = (
                        self.__Xtilde(
                            self.__X0(
                                self.__X1(
                                    path[per],
                                    dt, seeds[per])
                                ,dt)
                            , dt, seeds[per]))
                        else:
                            path[per+1] = (
                        self.__Xtilde(
                            self.__X1(
                                self.__X0(
                                    path[per],dt)
                                ,dt, seeds[per])
                            , dt, seeds[per]))
                    elif z == 2:
                        if self.vola**2 <= 4 * self.mean:
                            path[per+1] = (
                        self.__X0(
                            self.__Xtilde(
                                self.__X1(
                                    path[per], 
                                    dt, seeds[per])
                                ,dt, seeds[per])
                            , dt))
                        else:
                            path[per+1] = (
                        self.__X1(
                            self.__Xtilde(
                                self.__X0(
                                    path[per],dt)
                                ,dt, seeds[per])
                            , dt, seeds[per]))                        
                    else: # z == 3
                        if self.vola**2 <= 4 * self.mean:
                            path[per+1] = (
                        self.__X0(
                            self.__X1(
                                self.__Xtilde(
                                    path[per], 
                                    dt, seeds[per])
                                ,dt, seeds[per])
                            , dt))
                        else:
                            path[per+1] = (
                        self.__X1(
                            self.__X0(
                                self.__Xtilde(
                                    path[per],
                                    dt, seeds[per])
                                ,dt)
                            , dt, seeds[per]))   
                    path[per+1] *= np.exp(-self.velo * dt)
                
                else: # under threshold, switch to alternate sample method
                    U = (
                        np.random.Generator(
                            np.random.MT19937(
                                np.int64(seeds[per])))
                        .uniform())
                    u1,u2,u3 = self.moment(path[per],3,dt)
                    s = (u3 - u1 * u2) / (u2 - u1**2)
                    p = (u1 * u3 - u2**2) / (u2 - u1**2)
                    delta = np.sqrt(s**2 - 4 * p)
                    pi = (u1 - (s - delta) / 2) / delta
                    if U < pi:
                        path[per+1] = (s + delta) / 2
                    else:
                        path[per+1] = (s - delta) / 2
        return(path)
    
    def sample_paths(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        n_workers=None,
        n_path=100,
        method='Alfonsi2'):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        X_0 : double
            initial point of X. The default is 0.05.
        seed : integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10. The default is 1000
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count).        
        n_path: integer
            The number of sample paths to simulate. The default is 100
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        method: text
            the simulation method used to simulate sample paths. 
            Denote n_per as :math:`n`. Options are:
                
            - Exact: an exact scheme using non-central chi square distribution
            (implementation still in progress).
            
            - Brigo-Alfonsi: an implicit Euler-Maruyama scheme. Well-defined
              when :math:`\sigma^2\geq 2a` and :math:`1+kT/n>0`:
            .. math:: \hat{X}_{t_{i+1}} =  \hat{X}_{t_{i}} + \
              (a-k\hat{X}_{t_{i+1}}-\frac{\sigma^2}{2})\frac{T}{n} + \
              \sigma \sqrt{\hat{X}_{t_{i+1}}}(W_{t_{i+1}}-W_{t_i})
              
            If these conditions are not satistifed, the function will raise
            the ValueError
            
            - Daelbaen-Deelstra: a modified explicit Euler-Maruyama scheme:
            .. math::
                \hat{X}_{t_{i+1}} = \hat{X}_{t_{i}} + (a-k\hat{X}_{t_{i}}) \
                \frac{T}{n} + \sigma \sqrt{(\hat{X}_{t_{i}})^+} \
                (W_{t_{i+1}}-W_{t_i})
            - Lord: another modified explicit Euler-Maruyama scheme:
            .. math::
                \hat{X}_{t_{i+1}}=\hat{X}_{t_{i}} + (a-k(\hat{X}_{t_{i}})^+)\
                \frac{T}{n} + \sigma \sqrt{(\hat{X}_{t_{i}})^+} \
                (W_{t_{i+1}}-W_{t_i})
            - Alfonsi2: Alfonsi's 2nd order potential scheme. For details
              see [1].
            - Alfonsi3: Alfonsi's 3rd order potential scheme. For details
              see [1].
        
        Returns
        -------
        A list of one 2d array of dimension (n_path, n_per + 1) containing
        simulated sample paths and one vector of size (n_per + 1) containing
        the discretization time-grid. Each sub-array [x, :] is one sample path.

        References
        ----------
        [1] Alfonsi, Aurélien. "High order discretization schemes for the
        CIR process: application to affine term structure and Heston models."
        Mathematics of Computation 79.269 (2010): 209-237.

        [2] Deelstra, G., Delbaen, F.: Convergence of discretized stochastic
        (interest rate) processes with stochastic drift term.
        Appl. Stoch. Models Data Anal. 14(1), 77–84 (1998)

        [3] Lord, R., Koekkoek, R., Van Dijk, D.: A comparison of biased
        simulation schemes for stochastic volatility models.
        Quant. Finance 10(2), 177–194 (2010)
        
        [4] Brigo, D., Alfonsi, A.: Credit default swap calibration and 
        derivatives pricing with the SSRD stochastic intensity model. 
        Financ. Stoch. 9(1), 29–42 (2005)        
        """        
        results = sample_paths(
            model = self,
            n_workers=n_workers,
            n_path=n_path,
            n_per=n_per,
            seed=seed,
            method=method,
            X_0=X_0)
        return(results)
    
class Heston:
    r"""The Heston model.
    
    Parameters
    ----------
    :math:`r`: double
        the drift of S
    :math:`\rho`: double
        the weight of the diffusion process driving S that also drives V.
        It can also be interpreted as the correlation between 2 diffusion
        terms driving S and V.
    :math:`a`: double
        the long-run mean-reverting level of V
    :math:`\sigma`: double
        the constant volatility of V.
    :math:`k`: double
        the velocity of mean-reversion of V.
    T: double
        the terminal timepoint up to which S and V are defined.
        
    Returns
    -------
    An instance of 2 stochastic process :math:`S` and :math:`V`. 
    :math:`S` follows an Itô process, but its volatility :math:`\sqrt{V}` is 
    stochastic and follows the Cox-Ingersoll-Rox model. S and V are 
    characterized by the following SDE system:
        
    .. math::
        dS_t &= r S_t dt + \sqrt{V_t}S_t\left(\rho dW_t^S + \sqrt{1-\rho
            ^2}dW_t^V\right)
        
        dV_t &= k(a-V_t)dt + \sigma\sqrt{V_t}dW_t^V, t\in[0,T]
        
    """
    
    name = 'Heston model'    
    
    def __init__(self,r,rho,a,sigma,k,T):
        self.meanS  = r
        self.corr   = rho
        self.meanV  = a
        self.volaV  = sigma
        self.veloV  = k
        self.end_T  = T
    
    def __L1(self,x,x2_hat,t):
        r"""Generate sample path.
        
        Part of the 2nd potential order scheme.
        
        Parameters
        ----------
        x : array double
            The initial state :math:`x = (x_1, x_2, x_3, x_4)`.
        x2_hat : double
            :math:`\hat{x_2}`, the next step of :math:`x_2`.
            Sampled separately as a CIR process.
        t : double
            Size of the time step.

        Returns
        -------
        The next step of the sample paths
        :math:`\hat{x} = (\hat{x_1}, \hat{x_2}, \hat{x_3}, \hat{x_4})`,
        where :math:`\hat{x_2}` is simply the same as the input x2_hat.
        For other coordinates:
            
        .. math::
            \hat{x_3} &= x_3 + \frac{x_2 + \hat{x_2}}{2}t
            
            \hat{x_1} &= x_1 + \left(r - a\frac{\rho}{\sigma}\right)t
            + \left(\frac{k\rho}{\sigma} - \frac{1}{2}\right)(\hat{x_3}- x_3)
            + \frac{\rho}{\sigma}(\hat{x_2} - x_2)
            
            \hat{x_4} &= x_4 + \frac{\exp(x_1) + \exp(\hat{x_1})}{2}t

        """
        x3_hat = x[2] + (x[1] + x2_hat) / 2 * t
        x1_hat = (x[0]
            + (self.meanS - self.meanV * self.corr / self.volaV) * t
            + (self.veloV * self.corr / self.volaV - 1 / 2)
            * (x3_hat - x[2])
            + self.corr / self.volaV * (x2_hat - x[1])) 
        x4_hat = x[3] + (np.exp(x[0]) + np.exp(x1_hat)) / 2 * t
        
        return([x1_hat, x2_hat, x3_hat, x4_hat])

    def __L2(self,x,seed,t):
        r"""Generate sample path.
        
        Part of the 2nd potential order scheme.
        
        Parameters
        ----------
        x : array double
            The initial state :math:`x = (x_1, x_2, x_3, x_4)`.
        t : double
            Size of the time step.
        seed : integer
            The RNG seed.

        Returns
        -------
        The next step of the sample paths
        :math:`\hat{x} = (\hat{x_1}, \hat{x_2}, \hat{x_3}, \hat{x_4})`,
        where only :math:`\hat{x_1}` differs from the input :math:`x_1`.
        In specific:
        :math:`\hat{x_1} = x_1 + \sqrt{x_2}\sqrt{1-\rho^2}\sqrt{t}Z`,
        where :math:`Z \sim \mathcal{N}(0,1)`.
        """
        z = (
            np.random.Generator(
                np.random.MT19937(
                    np.int64(seed)))
            .standard_normal())
        x1_hat = (x[0]
            + np.sqrt(x[1]) * np.sqrt(1 - self.corr**2) * np.sqrt(t) * z)
        return([x1_hat,x[1],x[2],x[3]])
    
    def sample_path(
        self,
        S_0 = 100,
        V_0 = 1,
        n_per = 100,
        seed = 1000,
        method = 'Alfonsi2'):
        r"""Simulate a sample path of :math:`X`.
        
        Parameters
        ----------
        S_0: double, optional
            initial value of :math:`S`. The default is 0.05.
        V_0: double, optional
            initial value of :math:`V`. The default is 0.03.
        n_per: integer, optional
            number of intervals used to discretize the time interval
            [0,endT]. The discretized time grid is equidistant. The default is
            100.
        seed: integer, optional
            the seed used in sampling the Gaussian increment. Each
            increment uses a seed larger than the previous seed by 10. The
            default is 1000            
        method: text, optional
            the simulation method for :math:`V`. The default is 'Alfonsi2'. 
            See also the CIR class.
        
        Returns
        -------
        A 2d array of size (4,n_per + 1) containing the sample paths as rows.
        They are:
            
        :math:`((\hat{X_t})_1,(\hat{X_t})_2,(\hat{X_t})_3,(\hat{X_t})_4) = 
        (log(S_t), V_t, \int_0^t V_sds, \int_0^t S_t dt)`
        
        For details see [1].

        References
        ----------
        [1] Alfonsi, Aurélien. "High order discretization schemes for the
        CIR process: application to affine term structure and Heston models."
        Mathematics of Computation 79.269 (2010): 209-237.
            
        """
        path = np.zeros(
            shape=(4,n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        seeds = np.linspace(
            start=seed,
            stop=seed+10*(n_per-1),
            num=n_per,
            endpoint=True,
            dtype=np.int64)
        dt = self.end_T/n_per
        path[:,0] = [np.log(S_0),V_0,0,0]
        if not method in ['Exact','Brigo-Alfonsi','Daelbaen-Deelstra',
                          'Lord','Alfonsi2','Alfonsi3']:
            raise ValueError('wrong keyword for method')
        elif method =='Exact':
            pass # implementation still in progress
        else:
            path[1,:] = (
                CIR(self.meanV,self.volaV,self.veloV,self.end_T)
                .sample_path(
                    X_0 = V_0,
                    n_per = n_per,
                    seed = seed,
                    method = method))
            for per in np.arange(0,n_per):
                b = (
                    np.random.Generator(
                        np.random.MT19937(
                            np.int64(seeds[per])))
                    .choice([0,1]))
                if b == 0:
                    path[:,per+1] = (
                        self.__L1(
                            self.__L2(
                                path[:,per],
                                seeds[per],
                                dt),
                            path[1,per+1],
                            dt))
                else:
                    path[:,per+1] = (
                        self.__L2(
                            self.__L1(
                                path[:,per],
                                path[1,per+1],
                                dt),
                            seeds[per],
                            dt))                
        return(path)

    def sample_paths(
        self,
        X_0=0.05,
        n_per=100,
        seed=1000,
        n_workers=None,
        n_path=100,
        method='Alfonsi2'):
        r"""Simulate multiple sample paths of :math:`X`.
        
        Parameters
        ----------
        X_0 : double
            initial point of X. The default is 0.05.
        seed : integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10. The default is 1000
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count).        
        n_path: integer
            The number of sample paths to simulate. The default is 100
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        method: text, optional
            the simulation method for :math:`V`. The default is 'Alfonsi2'. 
            See also the CIR class.
        
        Returns
        -------
        A list of one 3d array of dimension (n_path, n_per + 1, 4) containing
        the simulated sample paths and one vector of size (n_per + 1) 
        containing the discretization time-grid. Each sub-array [x, :, :] 
        is one sample path. Each contains:
            
        :math:`((\hat{X_t})_1,(\hat{X_t})_2,(\hat{X_t})_3,(\hat{X_t})_4) = 
        (log(S_t), V_t, \int_0^t V_sds, \int_0^t S_t dt)`
        
        For details see [1].

        References
        ----------
        [1] Alfonsi, Aurélien. "High order discretization schemes for the
        CIR process: application to affine term structure and Heston models."
        Mathematics of Computation 79.269 (2010): 209-237.     
        """        
        results = sample_paths(
            model = self,
            n_workers=n_workers,
            n_path=n_path,
            n_per=n_per,
            seed=seed,
            method=method,
            X_0=X_0)
        return(results)