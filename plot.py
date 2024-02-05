import matplotlib.pyplot as plt


def plot(exp, rms, opt_output):
    plt.plot(exp.soc, exp.voltage, label='Experimental')
    plt.plot(opt_output.soc, opt_output.voltage, label='Model', linestyle='--')
    box = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax = plt.gca()
    plt.figtext(0.05, 0.05, f'RMSE: {rms} mV', bbox=box, ha='left', va='bottom', transform=ax.transAxes)
    plt.legend()
    plt.xlabel('SOC (-)')
    plt.ylabel('Voltage (V)')
    plt.title('Discharge Curve')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.show()


def plot_dva(Qs, dvdq):
    plt.plot(Qs, dvdq, label='Experimental')
    #plt.plot(opt_output.soc, opt_output.voltage, label='Model', linestyle='--')
    #box = dict(boxstyle='round', facecolor='white', alpha=0.5)
    #ax = plt.gca()
    #plt.figtext(0.05, 0.05, f'RMSE: {rms} mV', bbox=box, ha='left', va='bottom', transform=ax.transAxes)
    plt.legend()
    plt.xlabel('SOC (-)')
    plt.ylabel('Voltage (V)')
    plt.title('Differential Voltage')
    plt.grid(True)
    #plt.gca().invert_xaxis()
    plt.show()
