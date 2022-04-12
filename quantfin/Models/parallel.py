# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:10:06 2021.

@author: anhdu
"""
from multiprocessing import Pool
from itertools import repeat
import numpy as np


def sample_paths_parallel(
        model,
        n_workers = None,
        n_job=10,
        size_job=100,
        n_per=100,
        seeds=np.arange(10,dtype = np.int64),
        method=None,
        initial_state=None):
    r"""Simulate multiple sample paths (helper function).
    
    Parameters
    ----------
    model: class instance
        A stochastic model.
    n_workers: integer
        Number of workers to parallelize the simulation process. The default 
        is None. In this case, the number of workers is the number of
        processors (i.e. CPU core count).
    n_job: integer
        The number of chunks, each chunk is multiple sample paths. 
        The default is 1.
    size_job: integer
        The size of each chunk as the number of sample paths.
    n_per: integer
        number of intervals used to discretize the time interval
        :math:`[0,T]`. The discretized time grid is equidistant. 
        The default is 100.
    seeds: integer
        a list of rng seeds.
        The length of the seed vector must be equal to n_job.
    method: text
        Simulation method. For details check the sample_path method under
        each class
    initial_state: double
        The initial state of the stochstic process(es). Must be
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
    if len(seeds) != n_job:
        raise ValueError('length of the seed vector must be equal to n_job')
    if type(model).__name__ in (
            'GeomBM',
            'CIR',
            'HoLeeTimeConst',
            'VasicekTimeConst'):
        params = zip(
            repeat(initial_state),
            repeat(n_per),
            repeat(size_job),
            seeds,
            repeat(method))
        with Pool(processes = n_workers) as pool:
            paths = np.concatenate(
                pool.starmap(model.sample_paths,params),
                axis=0) 
    elif type(model).__name__ == 'Heston':
        params = zip(
            repeat(initial_state),
            repeat(n_per),
            repeat(size_job),
            seeds,
            repeat(method))
        with Pool(processes = n_workers) as pool:
            paths = np.concatenate(
                pool.starmap(model.sample_paths,params),
                axis=1) 
    return([paths,times])


