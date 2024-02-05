# noubal_comsol

The following is a parametrization method developed for interpreting heterogenously degraded regions within Li-ion batteries. It consist of a 2D physical model implemented in COMSOL Multyphysics combined with a parameters optimization algorithm DFO-LS (Derivative-Free Optimizer for Least-Squares Minimization) [1] developed in python. The MPh [2] interface is used for running the COMSOL code with python.
An alternative version with a PSO algoritmh can be used by changing the optimizer. This is delveloped with PySwarms [3].




[1] https://numericalalgorithmsgroup.github.io/dfols/build/html/index.html 

[2] https://mph.readthedocs.io/en/stable/index.html

[3] https://pyswarms.readthedocs.io/en/latest/#
