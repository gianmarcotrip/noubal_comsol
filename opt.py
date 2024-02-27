"""The two different optimization algorithms are defined and started.

All the optimization algorithms parameters
are set in main.py.

"""


import numpy as np
import pyswarms as ps
import dfols
import fun


def dfo(settings):
    """Runs the DFO-LS optimization algorithm.

    Parameters
    ----------
    settings : dictionary
        Contains all the information necessary to run the optimization.
        In order the experimental measurement to be fitted,
        the parameters to be optimized and their boundary values plus
        an indication of the objective function.
        On top of this all the DFO-LS parameters are defined.

    Returns
    -------
    soln.x : ndarray
        Optimized parameters.
    opt_output : DataFrame
        Results of the simulation ['time', 'voltage', 'current', 'capacity'].
    soln.resid : ndarray
        Residuals of the objective function.


    References
    ----------
    [1] https://numericalalgorithmsgroup.github.io/dfols/build/html/index.html
    """

    # Optimizer input arguments
    bpars_list = settings["balancing_pars"]
    x0 = np.hstack(np.array(bpars_list))
    e = settings["exp"]
    exp = tuple(e[['time', 'voltage', 'current', 'capacity']].apply(tuple, axis=1))

    # Create bounds
    bounds = settings["bounds"]
    lbounds = np.array(bounds[0])
    ubounds = np.array(bounds[1])

    # Perform optimization with either DVA of discharge curve as objective function
    if settings["dva"] == 'True':
        soln = dfols.solve(
                           fun.obj_fun_dva,
                           x0,
                           args=exp,
                           bounds=(lbounds, ubounds),
                           scaling_within_bounds=settings['scaling_within_bounds'],
                           maxfun=settings['maxfun'],
                           user_params={"restarts.use_restarts": settings['use_restarts'],
                                        "restarts.use_soft_restarts": settings['use_soft_restarts'],
                                        "restarts.max_unsuccessful_restarts": settings['max_unsuccessful_restarts']
                                        },
                           rhobeg=settings['rhobeg'],
                           rhoend=settings['rhoend'],
                           print_progress=settings['print_progress'],
                           )
    else:
        soln = dfols.solve(
                           fun.obj_fun,
                           x0,
                           args=exp,
                           bounds=(lbounds, ubounds),
                           scaling_within_bounds=settings['scaling_within_bounds'],
                           maxfun=settings['maxfun'],
                           user_params={"restarts.use_restarts": settings['use_restarts'],
                                        "restarts.use_soft_restarts": settings['use_soft_restarts'],
                                        "restarts.max_unsuccessful_restarts": settings['max_unsuccessful_restarts']
                                        },
                           rhobeg=settings['rhobeg'],
                           rhoend=settings['rhoend'],
                           print_progress=settings['print_progress'],
                           )
    print(soln)
    opt_output = fun.sim(soln.x)

    # Return in order: Optimized parameters, results of the simulation and residuals
    return soln.x, opt_output, soln.resid


def pso(settings, options):
    """Runs the PSO optimization algorithm.

    Parameters
    ----------
    settings : dictionary
        Contains all the information necessary to run the optimization.
        In order: the experimental measurement to be fitted, the parameters to be optimized
        and their boundary values plus an indication of the objective function.
        On top of this the PSO iterations and number of particles are defined.
    options : dictionary
        Contains all the hyperparameters necessary to define the PSO behaviour.

    Returns
    -------
    pos : ndarray
        Optimized parameters.
    opt_output : DataFrame
        Results of the simulation ['time', 'voltage', 'current', 'capacity'].
    cost : float64
        Root mean square error of the objective function.


    References
    ----------
    [1] https://pyswarms.readthedocs.io/en/latest/#
    """

    # Create bounds
    bounds = settings["bounds"]
    lbounds = np.array(bounds[0])
    ubounds = np.array(bounds[1])
    bounds = (lbounds, ubounds)

    # Optimizer input arguments
    e = settings["exp"]
    exp = tuple(e[['time', 'voltage', 'current', 'capacity']].apply(tuple, axis=1))
    exp_dict = {'exp': exp}

    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=settings['n_particles'],
                                       dimensions=len(lbounds),
                                       options=options,
                                       bounds=bounds
                                       )

    # Perform optimization with either DVA of discharge curve as objective function
    if settings["dva"] == 'True':
        cost, pos = optimizer.optimize(fun.obj_fun_dva,
                                       iters=settings['iters'],
                                       **exp_dict
                                       )
    else:
        cost, pos = optimizer.optimize(fun.obj_fun,
                                       iters=settings['iters'],
                                       **exp_dict
                                       )

    opt_output = fun.sim(pos)   # Return in order: Optimized parameters, results of the simulation and residuals
    return pos, opt_output, cost
