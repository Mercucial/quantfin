# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:38:36 2022.

@author: anhdu
"""
from .parallel import sample_paths_parallel
import numpy as np

class CIR:
    r"""The Cox-Ingersoll-Rox model.
    
    Parameters
    ----------
    :math:`\theta`: double
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
    
    .. math:: dX_t = \kappa(\theta-X_t)dt + \sigma\sqrt{X_t}dW_t, t\in[0,T]
    """

    module = 'Models'
    name = 'Cox-Ingersoll-Rox model'

    def __init__(self, theta, sigma, kappa, T):
        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa
        self.a = theta*kappa
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
            return 1/k*(1-np.exp(-k*t))
        else:
            return t

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
        where :math:`a = \kappa\theta`.
        """
        result = x + (self.a - self.sigma**2/4) * self.__zeta(-self.kappa,t)
        return (result)

    def __X1(self,x,t,pseudo_norm):
        r"""Generate sample path.

        Part of the 3rd potential order scheme. Allows vectorization.

        Parameters
        ----------
        x : double
            value of the prior step in the simulation scheme.
        t : double
            size of the timestep.
        pseudo_norm : double
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
        and :math:`a = \kappa\theta`.
        """
        result = (
            np.sqrt(x)
            + self.sigma * np.sqrt(self.__zeta(-self.kappa, t)) * pseudo_norm
            / 2)**2
        return (result)

    def __Xtilde(self, x, t, rade):
        r"""Generate sample path.

        Part of the 3rd potential order scheme. Allows vectorization.

        Parameters
        ----------
        x : double
            value of the prior step in the simulation scheme.
        t : double
            size of the timestep.
        rade : double
            realized values of the Rademacher random variables eps
            (see the description). Must have the same length as x.

        Returns
        -------
        The next step in the simulation scheme:
            .. math::
                \tilde{X}(x) = x + \sigma/\sqrt{2}
                \sqrt{\vert a - \sigma^2/4 \vert}
                \varepsilon \zeta_{-k}(t)

        where :math:`\varepsilon` is a Rademacher random variable
        with the following distribution:
            .. math::
                \mathbb{P}\left[\varepsilon = 1\right]
                =\mathbb{P}\left[\varepsilon = -1\right]
                = 1/2
        and :math:`a = \kappa\theta`.
        """
        result = (
            x
            + self.sigma/np.sqrt(2)
            * np.sqrt(np.abs(self.a - self.sigma**2/4))
            * rade
            * self.__zeta(-self.kappa, t)
        )
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
                x*np.exp(-self.kappa*t)
                + self.a*self.__zeta(self.kappa, t))
        if order >= 2:
            moment2 =(
                (moment1)**2
                + self.sigma**2*self.__zeta(self.kappa,t)
                * (
                    self.a*self.__zeta(self.kappa,t)/2
                    + x*np.exp(-self.kappa*t)))

        if order == 3:
            moment3 = (
                moment1 * moment2
                + self.sigma**2 * self.__zeta(self.kappa, t)
                * (2 * x**2 * np.exp(-2 * self.kappa * t)
                   + self.__zeta(self.kappa, t)
                   * (self.a + self.kappa**2 / 2)
                   * (
                       3 * x * np.exp(-self.kappa * t)
                       + self.a * self.__zeta(self.kappa, t))))
            return([moment3, moment2, moment1])
        elif order==2:
            return([moment2, moment1])
        else: # order ==1
            return(moment1)
        
    @np.errstate(invalid = 'ignore')
    def sample_paths(
        self,
        initial_state=0.05,
        n_per=100,
        n_path=100,
        seed=1000,
        method='Alfonsi2'):
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
            Denote n_per as :math:`n` and :math:`a = \kappa\theta`.
            Options are:

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
        n_path = np.int64(n_path)
        n_per = np.int64(n_per)
        seed = np.int64(seed)
        paths = np.zeros(
            shape=(n_path,n_per+1),
            dtype=np.float64)
        np.seterr(over='ignore')
        dt = self.end_T/n_per
        paths[:,0] = initial_state
        BM_samples = (
            np.random.Generator(np.random.MT19937(seed))
            .standard_normal(size = (n_path,n_per)))
        if not method in ['Exact','Brigo-Alfonsi','Daelbaen-Deelstra',
                          'Lord','Alfonsi2','Alfonsi3']:
            raise ValueError('wrong keyword value for method')
        elif method in ['Brigo-Alfonsi','Daelbaen-Deelstra','Lord']:
        # Modified Euler-Mariyama methods
            if method=='Brigo-Alfonsi':
                if self.sigma**2>=2*self.a and 1 + self.kappa*dt > 0:
                    for per in np.arange(n_per, dtype = np.int64):
                        paths[:,per+1] = (
                    (self.sigma*BM_samples[:,per]
                     + np.sqrt(
                         self.sigma**2*BM_samples[:,per]**2
                         + 4
                         * (paths[:,per]
                            + (self.kappa - self.sigma**2)*dt)
                         * (1 + self.kappa*dt)))
                    / (2*(1+self.kappa*dt)))**2
                else: raise ValueError('unsuitable parameters')
            elif method=='Daelbaen-Deelstra':
                for per in np.arange(n_per, dtype = np.int64):
                    paths[:,per+1] = (
                paths[:,per]
                + (self.a - self.kappa*paths[:,per]) * dt
                + self.sigma
                * np.sqrt(max(paths[:,per],0))
                * BM_samples[:,per])
            elif method=='Lord':
                for per in np.arange(n_per, dtype = np.int64):
                    paths[:,per+1]  = (
                paths[:,per]
                + (self.a - max(self.kappa*paths[:,per],0)) * dt
                + self.sigma
                    *np.sqrt(max(paths[:,per],0))
                    *BM_samples[:,per])
        elif method == 'Alfonsi2':
        # Second potential order scheme
            if self.sigma**2 <= 4*self.a:
                threshold = 0
            else:
                threshold = (
            np.exp(self.kappa*dt/2)
            * (
                self.__zeta(self.kappa, dt/2)*(self.sigma**2/4-self.a)
                + (
                    np.sqrt(
                        np.exp(self.kappa*dt/2)
                        * self.__zeta(self.kappa, dt/2)
                        *(self.sigma**2 / 4 - self.a))
                    + self.sigma/2*np.sqrt(3*dt))**2))
            pseudo_norm_samples = (
                # used to replace the sampled Gaussian r.v. when sample path
                # goes below threshold
                np.random.Generator(
                    np.random.MT19937(seed))
                .choice(
                    [-np.sqrt(3), 0, np.sqrt(3)],
                    size=(n_path, n_per),
                    p=[1/6, 2/3, 1/6]))
            unif_samples = (
                # used to determine which formula to use when sample path
                # goes below threshold
                np.random.Generator(
                    np.random.MT19937(
                        np.int64(seed)))
                .uniform(size=(n_path, n_per)))

            for per in np.arange(n_per, dtype=np.int64):
                paths_over_threshold = (
                    np.exp(-self.kappa*dt/2)
                    * (
                        np.sqrt(
                            (self.a - self.sigma**2/4)
                            * self.__zeta(self.kappa, dt/2)
                            + np.exp(-self.kappa*dt/2)
                            * paths[:, per])
                        + self.sigma / 2
                        * pseudo_norm_samples[:, per])**2
                    + (self.a - self.sigma**2/4)
                    * self.__zeta(self.kappa, dt/2)
                )
                u1, u2 = self.moment(paths[:, per], 2, dt)
                p = (1 - np.sqrt(1 - u1**2/u2))/2
                paths_under_threshold = np.where(
                    unif_samples[:, per] < p,
                    u1/(2*p),
                    u1/(2*(1-p))
                )
                paths[:, per+1] = np.where(
                    paths[:, per] >= threshold,
                    paths_over_threshold,
                    paths_under_threshold
                )
        elif method=='Alfonsi3':
            pseudo_norm_samples = (
                # used to replace the sampled Gaussian r.v. when sample path
                # goes below threshold
                np.random.Generator(
                    np.random.MT19937(seed))
                .choice(
                    [-np.sqrt(3), 0, np.sqrt(3)],
                    size=(n_path, n_per),
                    p=[1/6, 2/3, 1/6]))
            unif_samples = (
                # used to determine which formula to use when sample path
                # goes below threshold
                np.random.Generator(
                    np.random.MT19937(
                        np.int64(seed)))
                .uniform(size=(n_path, n_per)))
            choice_samples = (
                # used to determine which formula to use when sample path
                # goes below threshold in the 3rd order scheme
                np.random.Generator(
                    np.random.MT19937(seed))
                .choice(a=[1, 2, 3],
                        size=(n_path, n_per),
                        p=[1/3, 1/3, 1/3])
            )
            rademacher_samples = (
                # used in the 3rd order scheme
                np.random.Generator(
                    np.random.MT19937(seed))
                .choice(a=[-1, 1],
                        size=(n_path, n_per),
                        p=[0.5, 0.5])
            )
            if self.sigma**2 <= 4 * self.a/3:
                threshold = (
                    self.sigma/np.sqrt(2)
                    * np.sqrt(self.a - self.sigma**2/4))
            elif (self.sigma**2 > 4 * self.a/3
                and self.sigma**2 <= 4*self.a):
                threshold = (
                    np.sqrt(
                        self.sigma**2/4 - self.a
                        + self.sigma/np.sqrt(2)
                        * np.sqrt(self.a - self.sigma**2/4))
                    + self.sigma/2 * np.sqrt(3 + np.sqrt(6)))
            else: #self.sigma**2 > 4* self.a
                threshold = (
                    self.sigma**2 - 4 * self.a
                    + (
                        np.sqrt(self.sigma/np.sqrt(2)
                                * np.sqrt(self.sigma**2/4 - self.a))
                        + self.sigma / np.sqrt(2)
                        * np.sqrt(3 + np.sqrt(6))))

            for per in np.arange(n_per, dtype = np.int64):
                if self.sigma**2 <= 4 * self.a:
                    paths_over_threshold = (
                        np.exp(-self.kappa * dt)
                        * np.select(
                            [
                                np.equal(choice_samples[:, per], 1),
                                np.equal(choice_samples[:, per], 2),
                                np.equal(choice_samples[:, per], 3)
                            ],
                            [
                                self.__Xtilde(
                                    self.__X0(
                                        self.__X1(
                                            paths[:, per],
                                            dt, pseudo_norm_samples[:, per]),
                                        dt),
                                    dt, rademacher_samples[:, per]),
                                self.__X0(
                                    self.__Xtilde(
                                        self.__X1(
                                            paths[:, per],
                                            dt, pseudo_norm_samples[:, per]),
                                        dt, rademacher_samples[:, per]),
                                    dt),
                                self.__X0(
                                    self.__X1(
                                        self.__Xtilde(
                                            paths[:, per],
                                            dt, rademacher_samples[:, per]),
                                        dt, pseudo_norm_samples[:, per]),
                                    dt)
                            ]
                        )
                    )
                else:
                    paths_over_threshold = (
                        np.exp(-self.kappa * dt)
                        * np.select(
                            [
                                np.equal(choice_samples[:, per], 1),
                                np.equal(choice_samples[:, per], 2),
                                np.equal(choice_samples[:, per], 3)
                            ],
                            [
                                self.__Xtilde(
                                    self.__X1(
                                        self.__X0(
                                            paths[:, per], dt),
                                        dt, pseudo_norm_samples[:, per]),
                                    dt, rademacher_samples[:, per]),
                                self.__X1(
                                    self.__Xtilde(
                                        self.__X0(
                                            paths[:, per], dt),
                                        dt, rademacher_samples[:, per]),
                                    dt, pseudo_norm_samples[:, per]),
                                self.__X1(
                                    self.__X0(
                                        self.__Xtilde(
                                            paths[:, per],
                                            dt, rademacher_samples[:, per]),
                                        dt),
                                    dt, pseudo_norm_samples[:, per])
                            ]
                        )
                    )
                u1, u2, u3 = self.moment(paths[:, per], 3, dt)
                s = (u3 - u1 * u2) / (u2 - u1**2)
                p = (u1 * u3 - u2**2) / (u2 - u1**2)
                delta = np.sqrt(s**2 - 4 * p)
                pi = (u1 - (s - delta) / 2) / delta
                paths_under_threshold = np.where(
                    unif_samples[:, per] < pi,
                    (s + delta) / 2,
                    (s - delta) / 2
                )
                paths[:, per+1] = np.where(
                    paths[:, per] >= threshold,
                    paths_over_threshold,
                    paths_under_threshold
                )
        return(paths)

    def sample_paths_parallel(
        self,
        initial_state=0.05,
        n_workers = None,
        n_job=10,
        size_job=100,
        n_per=100,
        seeds=np.arange(10,dtype = np.int64),
        method='Alfonsi2'):
        r"""Simulate multiple sample paths of :math:`X`.

        Parameters
        ----------
        initial state : double
            initial point of :math:`X`. The default is 0.05.
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        seeds: integer
            a list of rng seeds.
            The length of the seed vector must be equal to n_job.
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The
            default is None. In this case, the number of workers is
            the number of processors (i.e. CPU core count).
        n_job: integer
            How many jobs are created to simulate the sample paths.
        size_job: integer
            The number of sample paths to simulate for each job. 
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
        results = sample_paths_parallel(
            model = self,
            n_workers=n_workers,
            n_job=n_job,
            size_job=size_job,
            seeds=seeds,
            method=method,
            initial_state=initial_state)
        return(results)

class Heston:
    r"""The Heston model.
    
    Parameters
    ----------
    :math:`r`: double
        the drift of :math:`S`. Also the (constant) risk-free interest rate
        under the risk-neutral probability measure.
    :math:`\rho`: double
        the correlation between 2 diffusion terms driving :math:`S` 
        and :math:`V`.
    :math:`\theta`: double
        the long-run mean-reverting level of :math:`V`
    :math:`\sigma`: double
        the constant volatility of :math:`V`.
    :math:`\kappa`: double
        the velocity of mean-reversion of :math:`V`.
    T: double
        the terminal timepoint up to which 
        :math:`S` and :math:`V` are defined.
        
    Returns
    -------
    An instance of 2 stochastic process :math:`S` and :math:`V`. 
    :math:`S` follows an Itô process, but its volatility :math:`\sqrt{V}` is 
    stochastic and follows the Cox-Ingersoll-Rox model. :math:`V` 
    and :math:`V` are characterized by the following SDE:
        
    .. math::
        dS_t &= r S_t dt + \sqrt{V_t}S_t\left(\rho dW_t^S + \sqrt{1-\rho
            ^2}dW_t^V\right)
        
        dV_t &= \kappa(\theta-V_t)dt + \sigma\sqrt{V_t}dW_t^V, t\in[0,T]
        
    """
    
    module = 'Models'
    name = 'Heston model'    
    
    def __init__(self, r, rho, theta, sigma, kappa, T):
        self.r = r
        self.rho = rho
        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa
        self.end_T = T
        self.a = theta*kappa
        self.vol_model = CIR(theta=theta, sigma=sigma, kappa=kappa, T=T)

    def __L1(self, x, x2_hat, t):
        r"""Generate sample path.
        
        Part of the 2nd potential order scheme.
        
        Parameters
        ----------
        x : double
            A 2d array of size (4, n_path) containing the initial state 
            :math:`x = (x_1, x_2, x_3, x_4)` for each sample path in n_path.
        x2_hat : double
            A 2d array of size (1, n_path) containing :math:`\hat{x_2}`,
            the next state of :math:`x_2`, which is the volatility process
            following the CIR model.
        t : double
            Size of the time step.

        Returns
        -------
        A 2d array of size (4, n_path) containing the next state 
        :math:`\hat{x} = (\hat{x_1}, \hat{x_2}, \hat{x_3}, \hat{x_4})`
        for each sample path in n_path. 
        :math:`\hat{x_2}` is simply the same as the input x2_hat.
        For other coordinates:
            
        .. math::
            \hat{x_3} &= x_3 + \frac{x_2 + \hat{x_2}}{2}t
            
            \hat{x_1} &= x_1 + \left(r - a\frac{\rho}{\sigma}\right)t
            + \left(\frac{k\rho}{\sigma} - \frac{1}{2}\right)(\hat{x_3}- x_3)
            + \frac{\rho}{\sigma}(\hat{x_2} - x_2)
            
            \hat{x_4} &= x_4 + \frac{\exp(x_1) + \exp(\hat{x_1})}{2}t

        """
        x3_hat = x[2,:] + (x[1,:] + x2_hat) / 2 * t
        x1_hat = (x[0]
            + (self.r - self.a * self.rho / self.sigma) * t
            + (self.kappa * self.rho / self.sigma - 1 / 2)
            * (x3_hat - x[2])
            + self.rho / self.sigma * (x2_hat - x[1])) 
        x4_hat = x[3] + (np.exp(x[0]) + np.exp(x1_hat)) / 2 * t
        
        return(np.stack([x1_hat, x2_hat, x3_hat, x4_hat], axis=1).transpose())

    def __L2(self, x, norm, t):
        r"""Generate sample path.
        
        Part of the 2nd potential order scheme.
        
        Parameters
        ----------
        x : double
            A 2d array of size (4, n_path) containing the initial state 
            :math:`x = (x_1, x_2, x_3, x_4)` for each sample path in n_path.
        t : double
            Size of the time step.
        norm : double
            A vector of size n_path containing realized values of Z,
            which follows the standard normal distribution.

        Returns
        -------
        A 2d array of size (4, n_path) containing the next state 
        :math:`\hat{x} = (\hat{x_1}, \hat{x_2}, \hat{x_3}, \hat{x_4})`
        for each sample path in n_path. Only :math:`\hat{x_1}` differs 
        from the input :math:`x_1`.
        In specific:
        :math:`\hat{x_1} = x_1 + \sqrt{x_2}\sqrt{1-\rho^2}\sqrt{t}Z`,
        where :math:`Z \sim \mathcal{N}(0,1)`.
        """
        # z = (
        #     np.random.Generator(
        #         np.random.MT19937(
        #             np.int64(seed)))
        #     .standard_normal())
        x1_hat = (x[0,:]
            + np.sqrt(x[1,:]) * np.sqrt(1 - self.rho**2) 
            * np.sqrt(t) * norm)
        return(np.concatenate([x1_hat[np.newaxis,:], x[1:, :]], axis=0))
    
    @np.errstate(invalid='ignore')
    def sample_paths(
        self,
        initial_state = dict(S_0 = 100, V_0 = 0.03),
        n_per = 100,
        n_path = 1e3,
        seed = 1000,
        method = 'Alfonsi2'):
        r"""Simulate a sample path of :math:`S` and :math:`V`.
        
        Parameters
        ----------
        initial_state: double, optional
            A dictionary containing the initial valuess of :math:`(S,V)`. 
            The default is :math:`S_0 = 100, V_0 = 0.03`.
        n_per: integer
            number of intervals used to discretize the time interval
            [0,endT]. The discretized time grid is equidistant. The default is
            100.
        n_path: integer
            number of sample paths to be simulated. The default is 100.
        seed: integer
            the seed used in sampling random variables.          
        method: text
            the simulation method for :math:`V`. The default is 'Alfonsi2'. 
            See the CIR class for details.
        
        Returns
        -------
        A 3d array of size (n_path,n_per + 1,4) containing the sample paths.
        Each 2d array of size (n_path, n_per + 1) contains a different type
        of process.
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
        S_0 = initial_state['S_0']
        V_0 = initial_state['V_0']
        n_path = np.int64(n_path)
        n_per = np.int64(n_per)
        seed = np.int64(seed)
        paths = np.zeros(
            shape=(4,n_path, n_per+1),
            dtype=np.float64)
        dt = self.end_T/n_per
        paths[:, :, 0] = np.broadcast_to(
            [np.log(S_0), V_0, 0, 0], (n_path, 4)).transpose()
        if not method in ['Exact', 'Brigo-Alfonsi', 'Daelbaen-Deelstra',
                          'Lord', 'Alfonsi2', 'Alfonsi3']:
            raise ValueError('wrong keyword for method')
        elif method =='Exact':
            pass # implementation still in progress
        else:
            paths[1, :, :] = (
                self.vol_model.sample_paths(
                    initial_state=V_0,
                    n_per=n_per,
                    n_path=n_path,
                    seed=seed,
                    method=method))
            choice_samples = (
                np.random.Generator(
                    np.random.MT19937(seed))
                .choice([0, 1],size = (n_per)))
            BM_samples = (
                np.random.Generator(
                    np.random.MT19937(seed))
                .standard_normal(size = (n_path,n_per)))
            for per in np.arange(n_per, dtype=np.int64):
                # b = (
                #     np.random.Generator(
                #         np.random.MT19937(
                #             np.int64(seeds[per])))
                #     .choice([0,1]))
                if choice_samples[per] == 0:
                    paths[:,:, per+1] = (
                        self.__L1(
                            self.__L2(
                                paths[:,:,per],
                                BM_samples[:,per],
                                dt),
                            paths[1,:,per+1],
                            dt))
                else:
                    paths[:,:,per+1] = (
                        self.__L2(
                            self.__L1(
                                paths[:,:,per],
                                paths[1,:,per+1],
                                dt),
                            BM_samples[:,per],
                            dt))                
        return(paths)

    def sample_paths_parallel(
        self,
        initial_state = dict(S_0 = 100, V_0 = 0.03),
        n_workers = None,
        n_job=10,
        size_job=1e3,
        n_per=100,
        seeds=np.arange(10,dtype = np.int64),
        method='Alfonsi2'):
        r"""Simulate multiple sample paths of :math:`S` and :math:`V`.
        
        Parameters
        ----------
        initial_state: double, optional
            A dictionary containing the initial valuess of :math:`(S,V)`. 
            The default is :math:`S_0 = 100, V_0 = 0.03`.
        n_per : integer
            number of intervals used to discretize the time interval
            :math:`[0,T]`. The discretized time grid is equidistant.
            The default is 100.
        seeds: integer
            a list of rng seeds.
            The length of the seed vector must be equal to n_job.
        n_workers: integer, optional
            Number of workers to parallelize the simulation process. The
            default is None. In this case, the number of workers is
            the number of processors (i.e. CPU core count).
        n_job: integer
            How many jobs are created to simulate the sample paths.
        size_job: integer
            The number of sample paths to simulate for each job. 
            The default is 100.
        method: text, optional
            the simulation method for :math:`V`. The default is 'Alfonsi2'. 
            See also the CIR class.
        
        Returns
        -------
        A list of one 3d array of dimension (n_path, n_per + 1, 4) containing
        the simulated sample paths and one vector of size (n_per + 1) 
        containing the discretization time-grid. 
        
        Each datapoint [x,y,z] corresponds path number x, simulated at
        time period y and belongs to process z. There are
        4 different processes simulated simultaneously as follows:
            
        :math:`((\hat{X_t})_1,(\hat{X_t})_2,(\hat{X_t})_3,(\hat{X_t})_4) = 
        (log(S_t), V_t, \int_0^t V_sds, \int_0^t S_t dt)`
        
        For details see [1].

        References
        ----------
        [1] Alfonsi, Aurélien. "High order discretization schemes for the
        CIR process: application to affine term structure and Heston models."
        Mathematics of Computation 79.269 (2010): 209-237.     
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