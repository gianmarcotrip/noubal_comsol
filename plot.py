import matplotlib.pyplot as plt


def plot(exp, rms, opt_output):
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


def plot_dva(Qs_e, dvdq_e, Qs_r, dvdq_r):
    plt.plot(Qs_e, dvdq_e, label='Experimental')
    plt.plot(Qs_r, dvdq_r, label='Model', linestyle='--')
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
