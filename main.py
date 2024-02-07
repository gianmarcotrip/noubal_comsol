import fun
import opt
import plot
import differential

settings = {"balancing_pars": [0.01, 0.01, 0.01, 0.01, 0.01],
            # 'h1', 'LI_loss', 'epss_ia_pos1', 'epss_ia_pos2', 'epss_ia_neg1', 'epss_ia_neg2'
            "bounds": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.4, 0.4, 0.4]],   # [min], [max]
            "dva": 'True', "PSO": 'False'}

# introduce if condition for different input experimental data types through settings information

if settings["PSO"] == 'True':
    x, opt_output, resid = opt.pso(settings)
else:
    x, opt_output, resid = opt.dfo(settings)

rms = fun.calculate_rms(resid)
rms = "{:.2f}".format(rms*1000)
exp = fun.get_exp()

Qs_e, Es_e, dvdq_e, dqdv_e = differential.dvdq_maria(exp['voltage'], exp['current'], exp['time'])
Qs_r, Es_r, dvdq_r, dqdv_r = differential.dvdq_maria(opt_output['voltage'], opt_output['current'], opt_output['time'])

plot.plot(exp, rms, opt_output)

plot.plot_dva(Qs_e, dvdq_e, Qs_r, dvdq_r)
