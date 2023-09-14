import tensorflow as tf
import math

__all__ = ['phase_shift_sum_base','phase_shift_sum','linear_radon','test_phase_shift_sum_dotprod',
           'test_phase_shift_sum_gradient','test_linear_radon_dotprod']

def phase_shift_sum_base(d, f, x, c, adjoint=False):
    """
    Apply the phase shift summation to a signal in the frequency domain

    :param d: Data with shape [..., nx, nf] if adjoint is False,
              or [..., nc, nt] if adjoint is True.
    :param f: Frequency vector with nf components
    :param x: A 1D array containing the full offset of the nx traces
    :param c: A 1D array containing the nc velocities
    :param adjoint: If True, goes from the velocity-freq to offset-freq domain,
                    if False, goes from offset-freq to velocity-freq domain.

    :return:
        m: Phase-shifted summation of the signal. If adjoint, the dimensions
           is [..., nx, nf], else is [..., nc, nf]
    """

    nd = tf.rank(d)
    inds = tf.concat([tf.ones(nd-1, dtype=tf.int32), [-1]], axis=0)
    f = tf.cast(tf.reshape(f, inds), d.dtype)

    c = tf.cast(c, d.dtype)
    nc = c.shape[-1]

    inds = tf.concat([tf.ones(nd - 2, dtype=tf.int32), [-1, 1]], axis=0)
    x = tf.cast(x, d.dtype)
    nx = x.shape[-1]

    if adjoint:
        c = tf.reshape(c, inds)
        m = tf.TensorArray(d.dtype, size=nx)
        for ix in tf.range(nx):
            i = tf.cast(tf.complex(0.0, -1.0), d.dtype)
            delta = tf.exp(i * 2 * math.pi * f * x[ix] / c)
            m = m.write(ix, tf.reduce_sum(delta * d, axis=-2))
    else:
        x = tf.reshape(x, inds)
        m = tf.TensorArray(d.dtype, size=nc)
        for ic in tf.range(nc):
            i = tf.cast(tf.complex(0.0, 1.0), d.dtype)
            delta = tf.exp(i * 2 * math.pi * f * x / c[ic])
            m = m.write(ic, tf.reduce_sum(delta * d, axis=-2))

    leading = tf.range(1, nd - 1)
    trailing = tf.constant([-1]) + tf.rank(f)
    new_order = tf.concat([leading, [0], trailing], axis=0)

    return tf.transpose(m.stack(), new_order)


@tf.custom_gradient
def phase_shift_sum(d, f, x, c, adjoint=False):
    """
    Apply the phase shift summation to a signal in the frequency domain,
    and provide a custom gradient.

    :param d: Data with shape [..., nx, nf] if adjoint is False,
              or [..., nc, nt] if adjoint is True.
    :param f: Frequency vector with nf components
    :param x: A 1D array containing the full offset of the nx traces
    :param c: A 1D array containing the nc velocities
    :param adjoint: If True, goes from the velocity-freq to offset-freq domain,
                    if False, goes from offset-freq to velocity-freq domain.

    :return:
        m: Phase-shifted summation of the signal. If adjoint, the dimensions
           is [..., nx, nf], else is [..., nc, nf]
    """
    dout = phase_shift_sum_base(d, f, x, c, adjoint=adjoint)

    def grad(dd):
        ddout = phase_shift_sum_base(dd, f, x, c, adjoint=not adjoint)
        return ddout, None, None, None

    return dout, grad


def linear_radon(d, dt, x, c, fmax=None, adjoint=False):
    """
    Linear TFradon transform.

    :param d: Data with shape [..., nx, nt] if adjoint is False,
              or [..., nc, nt] if adjoint is True.
    :param dt: The time step interval
    :param x:  A 1D array containing the full offset of the nx traces
    :param c:  A 1D array containing the nc velocities
    :param fmax: The maximum frequency to include for the transform
    :param adjoint: If True, goes from the velocity-time to offset-time domain,
                    if False, goes from offset-time to velocity-time domain.
    :return:
        Linear TFradon transformation of the d. If adjoint, the dimensions
           is [..., nx, nt], else is [..., nc, nt]
    """
    nt = d.shape[-1]
    d_fft = tf.signal.rfft(d)
    fnyq = 1.00 / (nt*dt) * (nt//2+1)
    if fmax is None:
        fmax = fnyq
    if fmax > fnyq:
        raise ValueError("fmax=%f is greater than nyquist=%f"
                         % (fmax, 0.5 / dt))
    f = tf.range(fmax, delta=1.00 / (nt*dt))
    nf0 = d_fft.shape[-1]
    nf = f.shape[-1]
    d_fft = d_fft[..., :nf]

    d_fft = phase_shift_sum(d_fft, f, x, c, adjoint=adjoint)

    if not tf.math.equal(nf0, nf):
        fpad = nf0 - nf
        paddings = tf.zeros([tf.rank(d_fft)-1, 2], dtype=tf.int32)
        paddings = tf.concat([paddings, tf.constant([[0, fpad]])], axis=0)
        d_fft = tf.pad(d_fft, paddings)

    return tf.signal.irfft(d_fft)


def test_phase_shift_sum_dotprod():
    """
    Dot product test of test_phase_shift.
    """
    nf = 30
    nx = 20
    nc = 15
    nb = 10
    d = tf.cast(tf.random.uniform([nb, nx, nf], dtype=tf.float32), tf.complex128)
    m = tf.cast(tf.random.uniform([nb, nc, nf], dtype=tf.float32), tf.complex128)
    f = tf.range(nf, dtype=tf.float32) * 0.3
    x = tf.range(1, nx + 1, dtype=tf.float32) * 0.1
    c = tf.range(1, nc + 1, dtype=tf.float32) * 0.2

    dot1 = tf.math.abs(tf.reduce_sum(tf.math.conj(m) * phase_shift_sum(d, f, x, c)))
    dot2 = tf.math.abs(tf.reduce_sum(tf.math.conj(phase_shift_sum(m, f, x, c, True)) * d))
    assert (dot2 - dot1).numpy() < 1e-12


def test_phase_shift_sum_gradient():
    """
    Testing if the custom gradient of test_phase_shift is equal to the automatic
    gradient provided by tensorflow.
    """
    nf = 30
    nx = 20
    nc = 15
    nb = 10
    d = tf.cast(tf.random.uniform([nb, nx, nf], dtype=tf.float32),
                tf.complex128)
    m = tf.cast(tf.random.uniform([nb, nc, nf], dtype=tf.float32),
                tf.complex128)
    f = tf.range(nf, dtype=tf.float32) * 0.3
    x = tf.range(1, nx + 1, dtype=tf.float32) * 0.1
    c = tf.range(1, nc + 1, dtype=tf.float32) * 0.2
    with tf.GradientTape() as tape:
        tape.watch(d)
        dout1 = phase_shift_sum(d, f, x, c)
    grad1 = tape.gradient(dout1, d)

    with tf.GradientTape() as tape:
        tape.watch(d)
        dout2 = phase_shift_sum_base(d, f, x, c)
    grad2 = tape.gradient(dout2, d)

    assert tf.reduce_max(tf.math.abs(grad1-grad2)) < 1e-12

    with tf.GradientTape() as tape:
        tape.watch(m)
        mout1 = phase_shift_sum(m, f, x, c, adjoint=True)
    grad1 = tape.gradient(mout1, m)

    with tf.GradientTape() as tape:
        tape.watch(m)
        mout2 = phase_shift_sum_base(m, f, x, c, adjoint=True)
    grad2 = tape.gradient(mout2, m)

    assert tf.reduce_max(tf.math.abs(grad1-grad2)) < 1e-12


def test_linear_radon_dotprod():
    """
    Dot product test for the linear TFradon function.

    Fails to always pass, probably because of fft and ifft of tensorflow is not
    accurate.

    """
    nt = 30
    nx = 20
    nc = 20
    nb = 10
    dt = 0.5
    fmax = 0.5

    d = tf.random.uniform([nb, nx, nt], dtype=tf.float64)
    m = tf.random.uniform([nb, nc, nt], dtype=tf.float64)
    x = tf.range(1, nx + 1)
    c = tf.range(1, nc + 1)

    def dot(a, b):
        return tf.abs(tf.reduce_sum(tf.math.conj(a) * b))
    dot1 = dot(m, linear_radon(d, dt, x, c, fmax=fmax))
    dot2 = dot(linear_radon(m, dt, x, c, fmax=fmax, adjoint=True), d)

    assert (dot2 - dot1).numpy() < 1e-12  #tf.signal.fft is not accurate in tf