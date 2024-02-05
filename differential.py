import numpy as np
from scipy import ndimage


def dvdq_maria(us, js, ts):
    def smooth(a, WSZ):
        # Mock-up of Matlab smooth function:
        # https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
        # a: NumPy 1-D array containing the data to be smoothed
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
        r = np.arange(1, WSZ - 1, 2)
        try:
            start = np.cumsum(a[:WSZ - 1])[::2] / r
            stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
        except ValueError:
            start = np.cumsum(a[:WSZ - 2])[::2] / r
            stop = (np.cumsum(a[:-WSZ:-1])[:-1:2] / r)[::-1]
        return np.concatenate((start, out0, stop))

    def smooth_matlab(E, prespan):
        Es = smooth(E[:-1], round(len(E) * prespan))
        return Es

    def gaussian_filter(x, y, span):
        c = ndimage.gaussian_filter(y, span, mode="reflect")

        return x, c

    prespan = 0.05
    Es = smooth_matlab(np.array(us), prespan)
    dt = ts[1] - ts[0]
    caps = np.cumsum(js) * dt / 3600
    Qs = smooth_matlab(caps, prespan)

    dV = np.diff(Es)
    dQ = np.diff(Qs)
    Es = Es[0:-1]
    Qs = Qs[0:-1]

    dvdq = dV / dQ[:len(dV)]
    dqdv = dQ[:len(dV)] / dV

    gaussspan = 0.01
    Qs, dvdq = gaussian_filter(Qs, -dvdq, len(Qs) * gaussspan)
    Es, dqdv = gaussian_filter(Es, -dqdv, len(Es) * gaussspan)

    return Qs, Es, dvdq, dqdv