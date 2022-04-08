# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:38:36 2022

@author: anhdu
"""
import numpy as np
# TODO: centralize all the sample draws into one function call for each
# distribution. Send only the realized values into _X% function calls
class CIR:
    r"""The Cox-Ingersoll-Rox model.
    
    Parameters
    ----------
    :math:`a`: double
        the long-run mean-reverting level of :math:`X`
    :math:`\sigma`: double
        the constant volatility of :math:`X`.
    :math:`k`: double
        the velocity of mean-reversion.
    :math:`T`: double
        the terminal timepoint up to which :math:`X` is defined.

        
    Returns
    -------
    A stochastic process :math:`X`.
    :math:`X` is characterized by the following SDE:
        
    .. math:: dX_t = k(a-X_t)dt + \sigma\sqrt{X_t}dW_t, t\in[0,T]
    """
    
    module = 'Models'
    name = 'Cox-Ingersoll-Rox model'
    def __init__(self,a,sigma,k,T):
        self.mean = a
        self.vola = sigma
        self.velo = k
        self.end_T = T
    
    def __zeta(self,k,t):
        r"""Return the value of the auxilliary zeta function.
        
        Allows vectorization.
            
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
        
        Part of the 3rd potential order scheme. Allows vectorization.
        
        Parameters
        ----------
        x : double
            value of the prior step in the simulation scheme
        t : double
            size of the timestep
            
        Returns
        -------
        The next step in the simulation scheme:
            .. math:: X_0(x) = x + (a - \sigma^2/4) \zeta_{-k}(t)
        """
        result = x + (self.mean - self.vola**2/4) * self.__zeta(-self.velo,t)
        return (result)
    
    def __X1(self,x,t,Y):
        r"""Generate sample path.
        
        Part of the 3rd potential order scheme. Allows vectorization.

        Parameters
        ----------
        x : double
            value of the prior step in the simulation scheme.
        t : double
            size of the timestep.
        Y : double
            realized values of the random variables Y (see the description). 
            Must have the same length as x.

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
        # Y = (
        #     np.random.Generator(np.random.MT19937(np.int64(seed)))
        #     .choice([-np.sqrt(3),0,np.sqrt(3)],1,p=[1/6,2/3,1/6]))
        result = (
            np.sqrt(x)
            + self.vola * np.sqrt(self.__zeta(-self.velo, t)) * Y / 2)**2
        return (result)  
    
    def __Xtilde(self, x, t, eps):
        r"""Generate sample path.
        
        Part of the 3rd potential order scheme. Allows vectorization.

        Parameters
        ----------
        x : double
            value of the prior step in the simulation scheme.
        t : double
            size of the timestep.
        eps : double
            realized values of the random variables eps (see the description). 
            Must have the same length as x.
        Returns
        -------
        The next step in the simulation scheme:
            .. math:: 
                \tilde{X}(x) = x + \sigma/\sqrt{2}
                \sqrt{\vert a - \sigma^2/4 \vert}
                \varepsilon \zeta_{-k}(t)
        
        where :math:`\varepsilon` is a random variable with the following 
        distribution:
            .. math::
                \mathbb{P}\left[\varepsilon = 1\right]
                =\mathbb{P}\left[\varepsilon = -1\right]
                = 1/2
        """
        # eps = (
        #     np.random.Generator(np.random.MT19937(np.int(seed)))
        #     .choice([1,-1],1, p=[0.5,0.5]))
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
    
