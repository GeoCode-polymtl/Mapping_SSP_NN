import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from linear_radon import linear_radon

__all__ = ['to_tsquare','hyperbolic_radon','test_hyperbolicradon']

def to_tsquare(d, t, alpha=1.0, adjoint=False, reg=0.0):
    """
    Perform b-spline interpolation of the traces on the time axis from t to t^2

    :param d: Input data with dimensions [ntraces,nt]
    :param t: Input time vector with nt samples
    :param alpha: Number of samples of the output will be nt * alpha
    :param adjoint: If True, goes from t^2 -> t
    :param reg:     A regularization coefficient for the interpolation

    :return:
        The interpolated data [..., alpha * nt]
    """

    t2 = t**2
    n = tf.cast(t.shape[0] * alpha, tf.int32)
    t2i = tf.linspace(t2[0], t2[-1], n)
    t2i = tf.reshape(t2i, [1, -1, 1])
    t2 = tf.reshape(t2, [1, -1, 1])
    t2i = tf.tile(t2i,[d.shape[0],1,1])
    t2 = tf.tile(t2,[d.shape[0],1,1])
    shape = d.shape
    d = tf.expand_dims(d,axis=-1)
    if adjoint:
        tel,ti = t2i,t2
    else:
        tel,ti = t2,t2i

    datai = tfa.image.interpolate_spline(tel,d,ti,2,reg)
    datai = tf.reshape(datai, list(shape[:-1]) + [-1])
    return datai, tf.reshape(t2i, [-1])


def hyperbolic_radon(d, t, x, c, fmax=None, adjoint=False, alpha=1.0, reg=0.0):
    """
    Hyperbolic TFradon transform

    :param d: Data with shape [..., nx, nt] if adjoint is False,
              or [..., nc, nt] if adjoint is True.
    :param t: A 1D array containing the time of the traces
    :param x:  A 1D array containing the full offset of the nx traces
    :param c:  A 1D array containing the nc velocities
    :param fmax: The maximum frequency to include for the transform
    :param adjoint: If True, goes from the velocity-time to offset-time domain,
                    if False, goes from offset-time to velocity-time domain.
    :param alpha: Number of samples of the output will be nt * alpha
    :param reg:  A regularization coefficient for the interpolation
    :return:
    """

    d2, t2 = to_tsquare(d, t, alpha=alpha, adjoint=adjoint, reg=reg)
    dt2 = t2[1] - t2[0]
    radlin2 = linear_radon(d2, dt2, x**2, c**2, adjoint=adjoint)
    radlin, t2 = to_tsquare(radlin2, t, alpha=alpha, adjoint=not adjoint)
    return radlin


def test_hyperbolicradon():

    nx = nt = 100
    d = np.zeros((100, 100), dtype=np.float)
    x = np.linspace(0, 1000, nx)
    t = np.linspace(0, 1, nt)
    dt = t[1] - t[0]
    for ref in [[0.2, 1000], [0.4, 1500], [0.6, 2500]]:
        for ix in range(nx):
            it = int(np.sqrt(ref[0] ** 2 + x[ix]**2 / ref[1]**2)/dt)
            if it < nt:
                d[ix, it] = 1.0

    c = tf.linspace(500, 3000, 100)
    d2, t2 = to_tsquare(d, t)
    d22, x2 = to_tsquare(tf.transpose(d2, [1, 0]), x)

    radon = hyperbolic_radon(d, t, x, c)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(np.transpose(d[:, :]))
    axs[1].imshow(np.transpose(d22[:, :]))
    axs[2].imshow(np.transpose(radon[:, :]))
    plt.show()


