import fun
import matplotlib.pyplot as plt

#client = mph.start()
#model = client.load('noubal/li_battery_2d_NMC-Gr.mph')

settings = {"balancing_pars": [0.5, 0.4, 0.05, 0.3, 0.05],
            # 'h1', 'LI_loss', 'epss_ia_pos1', 'epss_ia_pos2', 'epss_ia_neg1', 'epss_ia_neg2'
            "bounds": [[0, 0, 0, 0, 0], [1, 0.5, 0.5, 0.5, 0.5]],   # [min], [max]
            }

x, opt_output, resid = fun.opt(settings)

rms = fun.calculate_rms(resid)
rms = "{:.2f}".format(rms*1000)
exp = fun.get_exp()

fig = plt.figure()
plt.plot(exp.soc, exp.voltage, label='Experimental')
plt.plot(opt_output.soc, opt_output.voltage, label='Model', linestyle='--')
box = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax = plt.gca()
plt.figtext(0.05, 0.05, f'RMSE: {rms} mV', bbox=box, ha='left', va='bottom', transform=ax.transAxes)
plt.legend()
plt.xlabel('SOC (-)')
plt.ylabel('Voltage (V)')
plt.title('Discharge curve')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()




