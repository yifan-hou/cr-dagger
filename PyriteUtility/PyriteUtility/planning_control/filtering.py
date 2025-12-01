import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../'))

import numpy as np
import scipy
from collections import deque

class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        if any(np.isnan(x)):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")
    

class LiveLFilter(LiveFilter):
    def __init__(self, b, a, dim=1):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy.
            a (array-like): denominator coefficients obtained from scipy.
        """
        self.b = b
        self.a = a
        zero = np.zeros(dim)
        self._xs = deque([zero] * len(b), maxlen=len(b))
        self._ys = deque([zero] * (len(a) - 1), maxlen=len(a)-1)
    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y

class LiveLPFilter(LiveLFilter):
    def __init__(self, fs, cutoff, order=4, dim=1):
        """Initialize live lowpass filter.

        Args:
            fs (float): sampling rate.
            cutoff (float): cutoff frequency.
            order (int, optional): order of the filter. Defaults to 4.
        """
        b, a = scipy.signal.butter(order, cutoff, fs = fs)
        super().__init__(b, a, dim=dim)


def test_scalar_data():
    # create time steps and corresponding sine wave with Gaussian noise
    fs = 30  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds

    ys = np.sin(2*np.pi * 1.0 * ts)  # signal @ 1.0 Hz, without noise
    yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    yraw = ys + yerr

    from sklearn.metrics import mean_absolute_error as mae
    import matplotlib.pyplot as plt

    # define lowpass filter with 2.5 Hz cutoff frequency
    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")

    # # butterworth filter with 3rd order, cutoff frequency 0.05
    # b, a = scipy.signal.butter(3, 0.05)

    y_scipy_lfilter = scipy.signal.lfilter(b, a, yraw)

    live_lfilter = LiveLFilter(b, a)
    live_lpfilter = LiveLPFilter(fs, 2.5, order=4)

    # simulate live filter - passing values one by one
    y_live_lfilter = [live_lfilter(y) for y in yraw]
    y_live_lpfilter = [live_lpfilter(y) for y in yraw]


    print(f"lfilter error: {mae(y_scipy_lfilter, y_live_lfilter):.5g}")
    print(f"lpfilter error: {mae(y_scipy_lfilter, y_live_lpfilter):.5g}")

    plt.figure(figsize=[20, 10])
    plt.plot(ts, yraw, label="Noisy signal")
    plt.plot(ts, y_scipy_lfilter, lw=2, label="SciPy lfilter")
    plt.plot(ts, y_live_lfilter, lw=4, ls="dashed", label="LiveLFilter")
    plt.plot(ts, y_live_lpfilter, lw=4, ls="dotted", label="LiveLPFilter")

    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=4,
            fontsize="smaller")
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def test_vector_data():
    # create time steps and corresponding sine wave with Gaussian noise
    fs = 30  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds

    ys = [np.sin(2*np.pi * 1.0 * ts),  # signal @ 1.0 Hz, without noise
          np.cos(2*np.pi * 1.0 * ts)]
    ys = np.array(ys).T

    yerr = 0.5 * np.random.normal(size=ys.shape)  # Gaussian noise
    yraw = ys + yerr

    from sklearn.metrics import mean_absolute_error as mae
    import matplotlib.pyplot as plt

    # define lowpass filter with 2.5 Hz cutoff frequency
    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")

    y_scipy_lfilter = scipy.signal.lfilter(b, a, yraw)

    live_lfilter = LiveLFilter(b, a, dim=2)
    live_lpfilter = LiveLPFilter(fs, 2.5, order=4, dim=2)

    # simulate live filter - passing values one by one
    y_live_lfilter = np.array([live_lfilter(y) for y in yraw])
    y_live_lpfilter = np.array([live_lpfilter(y) for y in yraw])


    print(f"lfilter error: {mae(y_scipy_lfilter, y_live_lfilter):.5g}")
    print(f"lpfilter error: {mae(y_scipy_lfilter, y_live_lpfilter):.5g}")

    fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    axs[0].plot(ts, yraw[:, 0], label="Noisy signal")
    axs[0].plot(ts, y_scipy_lfilter[:, 0], lw=2, label="SciPy lfilter")
    axs[0].plot(ts, y_live_lfilter[:, 0], lw=4, ls="dashed", label="LiveLFilter")
    axs[0].plot(ts, y_live_lpfilter[:, 0], lw=4, ls="dotted", label="LiveLPFilter")

    axs[0].legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=4,
            fontsize="smaller")
    
    axs[1].plot(ts, yraw[:, 1], label="Noisy signal")
    axs[1].plot(ts, y_scipy_lfilter[:, 1], lw=2, label="SciPy lfilter")
    axs[1].plot(ts, y_live_lfilter[:, 1], lw=4, ls="dashed", label="LiveLFilter")
    axs[1].plot(ts, y_live_lpfilter[:, 1], lw=4, ls="dotted", label="LiveLPFilter")

    axs[1].legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=4,
            fontsize="smaller")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vector_data()