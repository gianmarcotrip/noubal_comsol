import numpy as np
from scipy import ndimage


def smooth(a, wsz):
    # Mock-up of Matlab smooth function:
    # https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(wsz, dtype=int), 'valid') / wsz
    r = np.arange(1, wsz - 1, 2)
    try:
        start = np.cumsum(a[:wsz - 1])[::2] / r
        stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    except ValueError:
        start = np.cumsum(a[:wsz - 2])[::2] / r
        stop = (np.cumsum(a[:-wsz:-1])[:-1:2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def smooth_matlab(e, prespan):
    es = smooth(e[:-1], round(len(e) * prespan))
    return es


def gaussian_filter(x, y, span):
    c = ndimage.gaussian_filter(y, span, mode="reflect")

    return x, c


def dvdq_maria(us, js, ts):
    prespan = 0.05
    es = smooth_matlab(np.array(us), prespan)
    dt = ts[1] - ts[0]
    caps = np.cumsum(js) * dt / 3600
    qs = smooth_matlab(caps, prespan)

    dv = np.diff(es)
    dq = np.diff(qs)
    es = es[0:-1]
    qs = qs[0:-1]

    dvdq = dv / dq[:len(dv)]
    dqdv = dq[:len(dv)] / dv

    gaussspan = 0.01
    qs, dvdq = gaussian_filter(qs, -dvdq, len(qs) * gaussspan)
    es, dqdv = gaussian_filter(es, -dqdv, len(es) * gaussspan)

    return qs, es, dvdq, dqdv
