import fun
import opt
import plot


settings = {"balancing_pars": [0.05, 0.05, 0.05, 0.05, 0.05],
            # 'h1', 'LI_loss', 'epss_ia_pos1', 'epss_ia_pos2', 'epss_ia_neg1', 'epss_ia_neg2'
            "bounds": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.4, 0.4, 0.4]],   # [min], [max]
            }

# x, opt_output, resid = opt.pso(settings)
x, opt_output, resid = opt.dfo(settings)

rms = fun.calculate_rms(resid)
rms = "{:.2f}".format(rms*1000)
exp = fun.get_txt()

plot.plot(exp, rms, opt_output)
