"""Main functions involved in the model

Responsible for objective function definition,
running simulations in COMSOL and doing all the
data handling required in between iterations.

"""

import numpy as np
import pandas as pd
# import time
import mph
import differential

'''.'''


def obj_fun(x, *args, **kwargs):
    """Discharge curve based objective function.
    Runs the simulation(s) and returns the residuals.

    DFO-LS simulates iteratively a single case (x) that is optimized.
    The PSO reads all the different cases (particles) at once
    and then simulates all of them for each objective function call.
    Therefore, it is necessary to write down a cycle
    as each line in x is a particle.

    Parameters
    ----------
    x : ndarray
        Contains all the parameters to be optimized.
    *args : tuple
        The DFO-LS requires a tuple for experimental data.
    **kwargs : dictionary
        The PSO requires a dictionary.

    Returns
    -------
    res : list
        One residual per each particle in the PSO.
        One list with residuals per each evaluated point
        in the DFO-LS case.
    """

    exp = args if args else kwargs

    if isinstance(exp, dict):
        exp = exp['exp']

    if len(np.shape(x)) == 1:
        r = sim(x)
        re = tuple(r[['time', 'voltage', 'current', 'capacity']].apply(tuple, axis=1))

        res = interp(exp, re)
    else:
        res = []
        for row in x:
            r = sim(row)
            re = tuple(r[['time', 'voltage', 'current', 'capacity']].apply(tuple, axis=1))

            residual = interp(exp, re)
            res.append(calculate_rms(residual))

    # print(time.perf_counter())
    return res


def obj_fun_dva(x, *args, **kwargs):
    """Differential Voltage curve based objective function.
    Runs the simulation(s) and returns the residuals.

    DFO-LS simulates iteratively a single case (x) that is optimized.
    The PSO reads all the different cases (particles) at once
    and then simulates all of them for each objective function call.
    Therefore, it is necessary to write down a cycle
    as each line in x is a particle.

    Parameters
    ----------
    x : ndarray
        Contains all the parameters to be optimized.
    *args : tuple
        The DFO-LS requires a tuple for experimental data.
    **kwargs : dictionary
        The PSO requires a dictionary.

    Returns
    -------
    res : list
        One residual per each particle in the PSO.
        One list with residuals per each evaluated point
        in the DFO-LS case.
    """

    exp = args if args else kwargs

    if isinstance(exp, dict):
        exp = exp['exp']

    if len(np.shape(x)) == 1:
        r = sim(x)
        re = tuple(r[['time', 'voltage', 'current', 'capacity']].apply(tuple, axis=1))
        res = interp_dva(exp, re)
    else:
        res = []
        for row in x:
            r = sim(row)
            re = tuple(r[['time', 'voltage', 'current', 'capacity']].apply(tuple, axis=1))
            residual = interp_dva(exp, re)
            res.append(calculate_rms(residual))

    # print(time.perf_counter())
    return res


''''''


def sim(x):
    """Run the simulation in COMSOL.
    Responsible for the simulation independently
    of the optimization algorithm utilized

    Parameters
    ----------
    x : ndarray
        Contains the parameters to be used in the single simulation.

    Returns
    -------
    results : DataFrame
        Results of the simulation ['time', 'voltage', 'current', 'capacity'].
    """
    client = mph.start()
    model = client.load('models/li_battery_2d_Mathilda_1Domain.mph')

    # model.parameter('h1', x[0])              #Share of heterogeneous region 1 [h2=1-h1]
    model.parameter('LI_loss', x[0])       # Loss of lithium inventory
    model.parameter('epss_ia_pos1', x[1])  # Inactive phase volume fraction, pos el, region 1
    # model.parameter('epss_ia_pos2', x[2])  # Inactive phase volume fraction, pos el, region 2
    model.parameter('epss_ia_neg1', x[2])  # Inactive phase volume fraction, neg el, region 1
    # model.parameter('epss_ia_neg2', x[4])  # Inactive phase volume fraction, neg el, region 1

    model.solve('Discharge')
    model.export()
    client.clear()

    results = get_results()
    return results


def interp(e, re):
    """Interpolate output of the model to experimental results.
    This allows to have comparable data and then calculate residuals in each evaluated data point.

    Data is interpolated with respect to exchanged charge of the shorter process between experimental and model.

    Parameters
    ----------
    e : tuple
        Contains the experimental results.
    re : tuple
        Contains the model results.

    Returns
    -------
    residual : list
        Residuals of each data point in the discharge curve.
    """
    u_exp = [element[1] for element in e]
    q_exp = [element[3] for element in e]
    u_res = [element[1] for element in re]
    q_res = [element[3] for element in re]

    if len(q_exp) < len(q_res):
        exp = np.interp(np.array(q_res), np.array(q_exp), np.array(u_exp))
        res = np.interp(np.array(q_res), np.array(q_res), np.array(u_res))
    else:
        exp = np.interp(np.array(q_exp), np.array(q_exp), np.array(u_exp))
        res = np.interp(np.array(q_exp), np.array(q_res), np.array(u_res))

    residual = [(a-b) for a, b in zip(res, exp)]
    return residual


def interp_dva(e, re):
    """Interpolate output of the model to experimental results.
    This allows to have comparable data and then calculate residuals in each evaluated data point.

    Data is interpolated with respect to exchanged charge of the shorter process between experimental and model.

    Parameters
    ----------
    e : tuple
        Contains the experimental results.
    re : tuple
        Contains the model results.

    Returns
    -------
    residual : list
        Residuals of each data point in the discharge curve.
    """

    t_exp = [element[0] for element in e]
    u_exp = [element[1] for element in e]
    c_exp = [element[2] for element in e]
    t_res = [element[0] for element in re]
    u_res = [element[1] for element in re]
    c_res = [element[2] for element in re]

    qs_e, es_e, dvdq_e, dqdv_e = differential.dvdq(u_exp, c_exp, t_exp)
    qs_r, es_r, dvdq_r, dqdv_r = differential.dvdq(u_res, c_res, t_res)

    dqdv_res = np.interp(qs_e, np.array(qs_r), np.array(dqdv_r))

    residual = [(a-b) for a, b in zip(dqdv_res, dqdv_e)]
    return residual


def length(e, re):
    """Used before interp. NOT USED ANYMORE
    It was only comparing the length of the vectors
    """
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
        results = [element[0] for element in re]
        results.extend([0] * (minexp - minres))

    res = [(a-b) for a, b in zip(results, exp)]
    return res


''' Functions responsible for reading and formatting both experimental and model data'''


def get_results():
    """Functions responsible for reading and formatting COMSOL data
    The result is exported as .txt file and is then open and formatted to a DataFrame.

    Returns
    -------
    results : DataFrame
        Results of the simulation ['time', 'voltage', 'current', 'capacity'].
    """

    results = pd.DataFrame(columns=['time', 'voltage', 'current', 'capacity'])
    with open('data/Discharge.txt') as d:
        dch = d.read()
        dch = dch.rstrip()
        dch = dch.split(';')

        results['time'] = (dch[::4])
        results['voltage'] = (dch[1::4])
        results['current'] = (dch[2::4])
        results['capacity'] = (dch[3::4])

        results['time'] = results['time'].astype(float)
        results['voltage'] = results['voltage'].astype(float)
        results['current'] = results['current'].astype(float)
        results['capacity'] = results['capacity'].astype(float)
    return results


def get_exp():  # From COMSOL example
    """Functions responsible for reading and formatting COMSOL data
    The result is exported as .txt file and is then open and formatted to a DataFrame.

    NOT USED ANYMORE. COMSOL OUTPUT WAS USED AS EXPERIMENTAL DATA IN THE TESTING PHASE.

    Returns
    -------
    exp : DataFrame
        Results of the simulation ['time', 'voltage', 'current', 'capacity'].
        """
    exp = pd.DataFrame(columns=['voltage', 'soc'])
    with open('data/exp.txt') as d:
        e = d.read()
        e = e.rstrip()
        e = e.split(';')

        exp['time'] = (e[::4])
        exp['voltage'] = (e[1::4])
        exp['current'] = (e[2::4])
        exp['soc'] = (e[3::4])

        exp['time'] = exp['time'].astype(float)
        exp['voltage'] = exp['voltage'].astype(float)
        exp['current'] = exp['current'].astype(float)
        exp['soc'] = exp['soc'].astype(float)

    return exp


def get_txt():  # Mathilda experimental data, also in her paper with Moritz
    """Functions responsible for reading and formatting M-M data
    The result is exported as .txt file and is then open and formatted to a DataFrame.
    Mathilda experimental data, comes from her paper with Moritz [1].

    Returns
    -------
    exp : DataFrame
        Results of the simulation ['time', 'voltage', 'current', 'capacity'].

    References
    ----------
    [1] https://doi.org/10.1016/j.est.2022.105948.
    """
    exp = pd.DataFrame(columns=['time', 'voltage', 'current', 'capacity'])
    e = pd.read_csv('data/PSb_c20_2_Comsol.txt',
                    sep='\t',
                    header=None,
                    names=['Column1', 'Column2', 'Column3']
                    )

    t = e['Column1'].to_numpy()
    v = e['Column2'].to_numpy()
    c = e['Column3'].to_numpy()
    c = -c

    # soc = []
    # for line in t:
    #    soc_v = (t[-1] - line) / (t[-1] - t[0])
    #    soc.append(soc_v)

    capacity = []
    time = 0
    cap = 0
    for i, j in zip(t, c):
        capacity_v = cap+j*(i-time)/3600
        capacity.append(capacity_v)
        time = i
        cap = capacity_v

    exp['voltage'] = np.array(v)
    exp['time'] = np.array(t)
    exp['current'] = np.array(c)
    exp['capacity'] = np.array(capacity)
    return exp


def calculate_rms(data_list):
    """Calculates the RMS of a list

    Parameters
    ----------
    data_list : list
        Contains an array of data.

    Returns
    -------
    rms : float
        Root mean square of the list.
    """

    l2a = np.array(data_list)
    squared_values = l2a ** 2
    mean_squared = np.mean(squared_values)
    rms = np.sqrt(mean_squared)
    return rms
