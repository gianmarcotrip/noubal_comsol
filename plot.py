"""


"""

import matplotlib.pyplot as plt
import numpy as np
import fun
import differential


def plot(exp, resid, opt_output):
    rms = fun.calculate_rms(resid)
    rms = "{:.2f}".format(rms * 1000)

    plt.plot(exp.capacity, exp.voltage, label='Experimental')
    plt.plot(opt_output.capacity, opt_output.voltage, label='Model', linestyle='--')
    box = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax = plt.gca()
    plt.figtext(0.05, 0.05, f'RMSE: {rms} mV', bbox=box, ha='left', va='bottom', transform=ax.transAxes)
    plt.legend()
    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title('Discharge Curve')
    plt.grid(True)
    # plt.gca().invert_xaxis()
    plt.show()


def plot_dva(exp, opt_output):
    # exp_v = np.interp(opt_output['time'], exp['time'], exp['voltage'])
    # exp_c = np.interp(opt_output['time'], exp['time'], exp['current'])

    qs_e, es_e, dvdq_e, dqdv_e = differential.dvdq_maria(exp['voltage'], exp['current'], opt_output['time'])
    qs_r, es_r, dvdq_r, dqdv_r = differential.dvdq_maria(opt_output['voltage'], opt_output['current'], opt_output['time'])

    plt.plot(qs_e, dvdq_e, label='Experimental')
    plt.plot(qs_r, dvdq_r, label='Model', linestyle='--')
    # box = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax = plt.gca()
    # plt.figtext(0.05, 0.05, f'RMSE: {rms} mV', bbox=box, ha='left', va='bottom', transform=ax.transAxes)
    plt.legend()
    plt.xlabel('Capacity (Ah)')
    plt.ylabel('E/V')
    plt.title('Differential Voltage')
    plt.grid(True)
    # plt.gca().invert_xaxis()
    plt.show()
