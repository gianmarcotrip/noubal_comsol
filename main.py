""" Initializes the optimization process.

Here the optimization method (DFOLS or PSO) is decided
and the main parameters are set.
Additionally, the objective function is chosen
between discharge and differential voltage.

"""


import fun
import opt
import plot
# import time


exp = fun.get_txt()
settings = {"exp": exp,
            "balancing_pars": [0.0601063, 0.05835396, 0.01807015, 4.20545254],  # [0.1, 0.05, 0.05, 0.05],
            # 'h1', 'LI_loss', 'epss_ia_pos1', 'epss_ia_pos2', 'epss_ia_neg1', 'epss_ia_neg2'
            "bounds": [[0.05, 0, 0, 4.1], [0.2, 0.2, 0.2, 4.3]],   # [min], [max]
            "dva": 'False',
            "PSO": 'False',
            }
# Set-up hyperparameters
options = {'c1': 0.5,
           'c2': 0.3,
           'w': 0.9,
           'k': len(settings["bounds"]),
           'p': len(settings["bounds"])
           }

if settings["PSO"] == 'True':

    settings['n_particles'] = 20
    settings['iters'] = 100

    x, opt_output, resid = opt.pso(settings, options)
else:

    settings['scaling_within_bounds'] = True
    settings['maxfun'] = 50
    settings['rhobeg'] = 0.2
    settings['rhoend'] = 1e-7
    settings['print_progress'] = True
    settings['use_restarts'] = True
    settings['use_soft_restarts'] = False
    settings['max_unsuccessful_restarts'] = 5

    x, opt_output, resid = opt.dfo(settings)


# print(time.perf_counter())

plot.plot(exp, opt_output, resid)
plot.plot_dva(exp, opt_output)
