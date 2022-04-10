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
    def __init__(self,theta,sigma,kappa,T):
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
        # Y = (
        #     np.random.Generator(np.random.MT19937(np.int64(seed)))
        #     .choice([-np.sqrt(3),0,np.sqrt(3)],1,p=[1/6,2/3,1/6]))
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
        n_chunk=10,
        size_chunk=100,
        n_per=100,
        seeds=np.arange(10,dtype = np.int64),
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
        results = sample_paths_parallel(
            model = self,
            n_workers=n_workers,
            n_chunk=n_chunk,
            size_chunk=size_chunk,
            seeds=seeds,
            method=method,
            initial_state=initial_state)
        return(results)