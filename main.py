import fun
import opt
import plot
import differential

settings = {"balancing_pars": [0.01, 0.01, 0.01, 0.01, 0.01],
            # 'h1', 'LI_loss', 'epss_ia_pos1', 'epss_ia_pos2', 'epss_ia_neg1', 'epss_ia_neg2'
            "bounds": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.4, 0.4, 0.4]],   # [min], [max]
            }

PSO = 'False'
if PSO == 'True':
    x, opt_output, resid = opt.pso(settings)
else:
    x, opt_output, resid = opt.dfo(settings)

rms = fun.calculate_rms(resid)
rms = "{:.2f}".format(rms*1000)
exp = fun.get_txt()

Qs, Es, dvdq, dqdv = differential.dvdq_maria(exp['voltage'], -exp['current'], exp['time'])

plot.plot(exp, rms, opt_output)

plot.plot_dva(Qs, dvdq)
