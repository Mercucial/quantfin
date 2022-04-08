# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:37:35 2022

@author: anhdu
"""

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
            initial value of :math:`X`. The default is 0.05
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
            initial point of :math:`X`. The default is 0.05
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100
        seed : integer
            the seed used in sampling the Gaussian distribution. Each
            increment uses a seed larger than the previous seed by 10.
            Each sample path uses a vector of seeds larger than the one
            of the preceding path by 10. The default is 1000
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The 
            default is None. In this case, the number of workers is 
            the number of processors (i.e. CPU core count)      
        n_path: integer
            The number of sample paths to simulate. The default is 100
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
                
