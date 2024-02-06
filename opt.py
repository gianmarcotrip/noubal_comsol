import pandas as pd
import numpy as np
import dfols
import mph
import fun
import pyswarms as ps


def dfo(settings):
    # Optimizer input arguments
    bpars_list = settings["balancing_pars"]
    x0 = np.hstack(np.array(bpars_list))
    e = fun.get_exp()
    exp = tuple(e[['voltage', 'soc']].apply(tuple, axis=1))

    # Create bounds
    bounds = settings["bounds"]
    lbounds = np.array(bounds[0])
    ubounds = np.array(bounds[1])

    # Perform optimization
    soln = dfols.solve(fun.obj_fun, x0, args=exp,
                       bounds=(lbounds, ubounds),
                       scaling_within_bounds=True, maxfun=1,
                       user_params={"restarts.use_restarts": False, "restarts.use_soft_restarts": False,
                                    "restarts.max_unsuccessful_restarts": 5},
                       rhobeg=0.2, rhoend=1e-7, print_progress=True,
                       )

    print(soln)
    opt_output = fun.sim(soln.x)

    # Return in order: Optimized parameters, results of the simulation and residuals
    return soln.x, opt_output, soln.resid


def pso(settings):
    # Create bounds
    bounds = settings["bounds"]
    lbounds = np.array(bounds[0])
    ubounds = np.array(bounds[1])
    bounds = (lbounds, ubounds)

    # Optimizer input arguments
    e = fun.get_txt()
    exp = tuple(e[['voltage', 'soc']].apply(tuple, axis=1))
    exp_dict = {'exp': exp}

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 5, 'p': 5}

    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=5, options=options, bounds=bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(fun.obj_fun, iters=1, **exp_dict)

    opt_output = fun.sim(pos)
    # Return in order: Optimized parameters, results of the simulation and residuals
    return pos, opt_output, cost
