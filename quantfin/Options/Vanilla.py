# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 01:15:02 2021

@author: anhdu
"""
import numpy as np

def fair_val(
        option,
        asset,
        method='Sim',
        **kwargs):
    r"""Compute the fair value of the option.
    
    The market is assumed to be arbitrage-free and the underlying asset
    has already been fitted to a model under a risk-neutral probability
    measure.
    
    Parameters
    ----------
    option : class instance
        The option created using the Options subpackage
    asset : class instance
        The underlying asset, created from the Models subpackage
    method : text
        The pricing method. Options are: 
        
        - Sim: Monte-Carlo simulation of the underlying
          asset's price paths
        - Lat: Lattice method (still in implementation)
        - Fourier: Fourier Pricing (still in implementation) 
    kwargs : keyword arguments which depends on the pricing method
    
    Returns
    -------
    The current fair value of the option.
    """
    if asset.__module__ [:15] != 'quantfin.Models':
        raise TypeError(
            'S must be a class instance from the Models sub-package')
        
    if method == ' Sim':
        if option.type == 0:
            mult = -1
        else: mult = 1
        
        if type(asset).__name__ == 'Heston':
            S_0 = kwargs.get('S_0')
            V_0 = kwargs.get('V_0')
            payoffs = (
                np.max(
                    mult * (np.exp(paths[1][:,-1,0]) - option.K)
                    , 0))
            
        else:
            payoffs = (
                np.max(
                    mult * (paths[1][:,-1] - self.K)
                    , 0))
    return(np.mean(payoffs))

class VanillaOptions:
    r"""The class of vanilla (European) options.
    
    Parameters
    ----------
    typ : integer
        Option type: 0 for call options and 1 for put options
    K : double
        Strike price.
    T : double
        Maturity period.
    
    Return
    ------
    A vanilla (European) option.
    At maturity, the payoff is
    :math:`max(S-K,0)` for call options and
    :math:`max(K-S,0)` for put options.
    
    """
    
    name = 'Vanilla options'
    module = 'Options'
    
    def __init__(self,typ,K,T):
        self.type   = typ
        self.K      = K
        self.T      = T
    
