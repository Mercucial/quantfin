# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:38:55 2022

@author: anhdu
"""

class Heston:
    r"""The Heston model.
    
    Parameters
    ----------
    :math:`r`: double
        the drift of :math:`S`
    :math:`\rho`: double
        the correlation between 2 diffusion terms driving :math:`S` 
        and :math:`V`.
    :math:`a`: double
        the long-run mean-reverting level of :math:`V`
    :math:`\sigma`: double
        the constant volatility of :math:`V`.
    :math:`k`: double
        the velocity of mean-reversion of :math:`V`.
    T: double
        the terminal timepoint up to which 
        :math:`S` and :math:`V` are defined.
        
    Returns
    -------
    An instance of 2 stochastic process :math:`S` and :math:`V`. 
    :math:`S` follows an Itô process, but its volatility :math:`\sqrt{V}` is 
    stochastic and follows the Cox-Ingersoll-Rox model. :math:`V` 
    and :math:`V` are characterized by the following SDE system:
        
    .. math::
        dS_t &= r S_t dt + \sqrt{V_t}S_t\left(\rho dW_t^S + \sqrt{1-\rho
            ^2}dW_t^V\right)
        
        dV_t &= k(a-V_t)dt + \sigma\sqrt{V_t}dW_t^V, t\in[0,T]
        
    """
    
    module = 'Models'
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
        r"""Simulate a sample path of :math:`S` and :math:`V`.
        
        Parameters
        ----------
        S_0: double, optional
            initial value of :math:`S`. The default is 100
        V_0: double, optional
            initial value of :math:`V`. The default is 0.03
        n_per: integer, optional
            number of intervals used to discretize the time interval
            [0,endT]. The discretized time grid is equidistant. The default is
            100
        seed: integer, optional
            the seed used in sampling the Gaussian increment. Each
            increment uses a seed larger than the previous seed by 10. The
            default is 1000            
        method: text, optional
            the simulation method for :math:`V`. The default is 'Alfonsi2'. 
            See also the CIR class
        
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
        S_0=100,
        V_0=0.03,
        n_per=100,
        seed=1000,
        n_workers=None,
        n_path=100,
        method='Alfonsi2'):
        r"""Simulate multiple sample paths of :math:`S` and :math:`V`.
        
        Parameters
        ----------
        S_0: double, optional
            initial value of :math:`S`. The default is 100
        V_0: double, optional
            initial value of :math:`V`. The default is 0.03
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
        results = sample_paths(
            model = self,
            S_0=S_0,
            V_0=V_0,
            n_workers=n_workers,
            n_path=n_path,
            n_per=n_per,
            seed=seed,
            method=method)
        return(results)