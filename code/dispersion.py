
from linear_radon import phase_shift_sum
import tensorflow as tf

__all__ = ['linear_radon_freq','dispersion']

def linear_radon_freq(d, dt, x, c, fmax=None, norm=False, epsilon=0.001):

    nt = d.shape[-1]
    d_fft = tf.signal.rfft(d)
    if norm: d_fft /= tf.cast(tf.abs(d_fft) + epsilon*tf.math.reduce_max(tf.abs(d_fft)),dtype=tf.complex64)
    fnyq = 1.00 / (nt*dt) * (nt//2+1)
    if fmax is None:
        fmax = fnyq
    if fmax > fnyq:
        raise ValueError("fmax=%f is greater than nyquist=%f"
                         % (fmax, 0.5 / dt))
    f = tf.range(fmax, delta=1.00 / (nt*dt))
    nf = f.shape[-1]

    d_fft = d_fft[..., :nf]

    return phase_shift_sum(d_fft, f, x, c)


def dispersion(d, dt, x, c, fmax=None,epsilon=0.001):
    return tf.abs(linear_radon_freq(d, dt, x, c, fmax=fmax, norm=True,epsilon=epsilon))

