import pandas as pd
import numpy as np
import dfols
import mph
import time


def opt(settings):
    bpars_list = settings["balancing_pars"]
    x0 = np.hstack(np.array(bpars_list))
    e = get_exp()
    exp = tuple(e[['voltage', 'soc']].apply(tuple, axis=1))
    bounds = settings["bounds"]
    lbounds = np.array(bounds[0])
    ubounds = np.array(bounds[1])

    soln = dfols.solve(obj_fun, x0, args=(exp),
                       bounds=(lbounds, ubounds),
                       scaling_within_bounds=True, maxfun=50,
                       user_params={"restarts.use_restarts": False, "restarts.use_soft_restarts": False,
                                    "restarts.max_unsuccessful_restarts": 5},
                       rhobeg=0.2, rhoend=1e-7, print_progress=True,
                       )

    print(soln)
    x = soln.x
    resid = soln.resid
    opt_output = sim(x)

    return x, opt_output, resid


def obj_fun(x, *args):
    e = args
    r = sim(x)
    re = tuple(r[['voltage', 'soc']].apply(tuple, axis=1))

    minexp = len(e) - 10
    minres = len(re)
    # min_min = min([minexp, minres])-10
    skip = 5

    if minres >= minexp:
        e = e[skip:minexp + skip]
        exp = [element[0] for element in e]
        re = re[skip:minexp + skip]
        results = [element[0] for element in re]
    else:
        e = e[skip:minexp + skip]
        exp = [element[0] for element in e]
        # re = re[minres]
        results = [element[0] for element in re]
        zeros = ([0] * (minexp - minres))
        results.extend([0] * (minexp - minres))

    # interp_res =

    res = [a - b for a, b in zip(results, exp)]
    res = np.array(res)

    #print(time.perf_counter())
    return res


def sim(x):
    client = mph.start()
    model = client.load('li_battery_2d_NMC-Gr.mph')

    # model.parameter('h1', x[0])              #Share of heterogeneous region 1 [h2=1-h1]
    model.parameter('LI_loss', x[0])  # Loss of lithium inventory
    model.parameter('epss_ia_pos1', x[1])  # Inactive phase volume fraction, pos el, region 1
    model.parameter('epss_ia_pos2', x[2])  # Inactive phase volume fraction, pos el, region 2
    model.parameter('epss_ia_neg1', x[3])  # Inactive phase volume fraction, neg el, region 1
    model.parameter('epss_ia_neg2', x[4])  # Inactive phase volume fraction, neg el, region 1

    model.solve('Discharge')
    model.export()
    client.clear()

    results = get_results()
    return results


def get_results():
    results = pd.DataFrame(columns=['voltage', 'soc'])
    with open('Discharge.txt') as d:
        dch = d.read()
        dch = dch.rstrip()
        dch = dch.split(';')

        results['voltage'] = (dch[1::2])
        results['soc'] = (dch[::2])

        results['voltage'] = results['voltage'].astype(float)
        results['soc'] = results['soc'].astype(float)

    return results


def get_exp():
    exp = pd.DataFrame(columns=['voltage', 'soc'])
    with open('exp.txt') as d:
        e = d.read()
        e = e.rstrip()
        e = e.split(';')

        exp['voltage'] = (e[1::2])
        exp['soc'] = (e[::2])

        exp['voltage'] = exp['voltage'].astype(float)
        exp['soc'] = exp['soc'].astype(float)

    return exp


def calculate_rms(array):
    # Convert the array to a numpy array
    np_array = np.array(array)

    # Calculate the square of each element
    squared_values = np_array ** 2

    # Calculate the mean of the squared values
    mean_squared = np.mean(squared_values)

    # Calculate the square root of the mean squared value
    rms = np.sqrt(mean_squared)

    return rms
