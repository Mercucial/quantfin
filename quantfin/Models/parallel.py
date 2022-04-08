# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:10:06 2021.

@author: anhdu
"""
from multiprocessing import Pool
from itertools import repeat
import numpy as np

# def apply_args_and_kwargs(fn, args, kwargs):
#     """
#     Snippet function which passes args and kwargs to function call fn.

#     Parameters
#     ----------
#     fn: python function or class method
#         The function/method to be called.
#     args: 
#         Named arguments
#     kwargs :
#         Keyword arguments

#     Returns
#     -------
#     fn(*args, **kwargs)
    
#     Reference
#     ---------
#     https://stackoverflow.com/questions/45718523/
    
#     """
#     return fn(*args, **kwargs)

# def starmap_with_kwargs(fn, args_iter, n_workers = None, kwargs_iter = None):
#     """
#     Snippet function which passes kwargs to function calls inside starmap.

#     Parameters
#     ----------
#     pool: Pool object
#         The pool of workers to execute the parallelized function calls.
#     fn: python function or class method
#         The function/method to be paralellized by pool.starmap
#     args_iter: iterable
#         An iterable containing named arguments to be passed onto fn.
#     kwargs_iter: iterable
#         An iterable containing optional keyword arguments to be passed onto fn.
#     n_workers: integer
#         Number of processes to execute the parallelized function calls. The 
#         default is None. In this case, the number of workers is the number of
#         processors (i.e. CPU core count).

#     Returns
#     -------
#     fn(args, kwargs), paralellized.

#     Reference
#     ---------
#     https://stackoverflow.com/questions/45718523/
    
#     """
#     args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
#     with Pool(processes = n_workers) as pool:
#         return pool.starmap(apply_args_and_kwargs, args_for_starmap) 
def sample_paths_parallel(
        model,
        n_workers = None,
        n_chunk=10,
        size_chunk=100,
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
    n_chunk: integer
        The number of chunks, each chunk is multiple sample paths. 
        The default is 1.
    size_chunk: integer
        The size of each chunk as the number of sample paths.
    n_per: integer
        number of intervals used to discretize the time interval
        :math:`[0,T]`. The discretized time grid is equidistant. 
        The default is 100.
    seeds: integer
        a list of rng seeds.
        The length of the seed vector must be equal to n_chunk.
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
    if len(seeds) != n_chunk:
        raise ValueError('length of the seed vector must be equal to n_chunk')
    if type(model).__name__ in (
            'GeomBM',
            'CIR',
            'HoLeeTimeConst',
            'VasicekTimeConst'):
        params = zip(
            repeat(initial_state),
            repeat(n_per),
            repeat(size_chunk),
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
            repeat(size_chunk),
            seeds,
            repeat(method))
        with Pool(processes = n_workers) as pool:
            paths = np.transpose(
                np.stack(
                    pool.starmap(model.sample_paths,params),
                    axis = 2))
    return(paths)


