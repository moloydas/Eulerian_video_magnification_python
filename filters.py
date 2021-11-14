#filters.py
import scipy.fftpack
from scipy.signal import butter, lfilter
import numpy as np

def ideal_temporal_filter(pyd, fps, low_f, high_f):
    fft = scipy.fftpack.fft(pyd, axis=0)
    frequencies = scipy.fftpack.fftfreq(pyd.shape[0], d=1.0/fps)
    bound_low = (np.abs(frequencies - low_f)).argmin()
    bound_high = (np.abs(frequencies - high_f)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = scipy.fftpack.ifft(fft, axis=0)
    return np.abs(iff)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

