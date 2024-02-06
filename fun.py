import mph
import numpy as np
import pandas as pd
import time

'''DFOLS reads a tuple for experimental data and then simulates iteratively the single case that is gradually optimized.
 Tee PSO reads a dictionary and then simulates for all the different cases at once. 
 Therefore it is necessary to write down a cycle as each line in x is a particle.'''


def obj_fun(x, *args, **kwargs):
    exp = args if args else kwargs

    if isinstance(exp, dict):
        exp = exp['exp']

    if len(np.shape(x)) == 1:
        r = sim(x)
        re = tuple(r[['voltage', 'soc']].apply(tuple, axis=1))

        res = interp(exp, re)
        # res = length(exp, re)
    else:
        res = []
        for row in x:
            r = sim(row)
            re = tuple(r[['voltage', 'soc']].apply(tuple, axis=1))

            residual = interp(exp, re)
            # residual = length(exp, re)
            res.append(calculate_rms(residual))

    print(time.perf_counter())
    return res


'''Responsible for the simulation independently from the optimization algorithm utilized'''


def sim(x):
    client = mph.start()
    model = client.load('li_battery_2d_NMC-Gr.mph')

    # model.parameter('h1', x[0])              #Share of heterogeneous region 1 [h2=1-h1]
    model.parameter('LI_loss', x[0])       # Loss of lithium inventory
    model.parameter('epss_ia_pos1', x[1])  # Inactive phase volume fraction, pos el, region 1
    model.parameter('epss_ia_pos2', x[2])  # Inactive phase volume fraction, pos el, region 2
    model.parameter('epss_ia_neg1', x[3])  # Inactive phase volume fraction, neg el, region 1
    model.parameter('epss_ia_neg2', x[4])  # Inactive phase volume fraction, neg el, region 1

    model.solve('Discharge')
    model.export()
    client.clear()

    results = get_results()
    return results


''' Two different ways of manipulating data and obtaining the residuals. 
The First is based on interpolation to a common SOC that ranges from 1 to 0 in n point.
The second relies purely on the length of the vectors.'''


def interp(e, re):
    soc = np.linspace(1, 0, 100)
    u_exp = [element[0] for element in e]
    x_exp = [element[1] for element in e]
    u_res = [element[0] for element in re]
    x_res = [element[1] for element in re]

    exp = np.interp(soc, np.array(x_exp), np.array(u_exp))
    res = np.interp(soc, np.array(x_res), np.array(u_res))

    residual = [(a-b) for a, b in zip(res, exp)]
    return residual


def length(e, re):
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


def get_results():  # From COMSOL txt file
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


def get_exp():  #From COMSOL example
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


def get_txt():  # Mathilda experimental data, also in her paper with Moritz
    exp = pd.DataFrame(columns=['voltage', 'time', 'current', 'soc'])
    e = pd.read_csv('PSb_c20_2_Comsol.txt', sep='\t', header=None, names=['Column1', 'Column2', 'Column3'])

    t = e['Column1'].to_numpy()
    v = e['Column2'].to_numpy()
    c = e['Column3'].to_numpy()

    soc = []
    for line in t:
        soc_v = (t[-1] - line) / (t[-1] - t[0])
        soc.append(soc_v)

    exp['voltage'] = np.array(v)
    exp['time'] = np.array(t)
    exp['current'] = np.array(c)
    exp['soc'] = np.array(soc)
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
