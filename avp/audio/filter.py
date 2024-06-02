import numpy as np

import scipy

from avp.audio.signal import ndsignal

import matplotlib.pyplot as plt

from avp.audio.util import _plt_use_style

class iirfilter:
    def __init__(self, sr: int, critical_freq: int | tuple[int, int], order=5, btype="lowpass", plot_fftbins: int = 512):
        Wn = critical_freq if isinstance(critical_freq, int) else list(critical_freq)

        self.sos = scipy.signal.iirfilter(N=order, Wn=Wn, btype=btype, ftype="butter", output="sos", fs=sr, analog=False)

        self.w, self.h = scipy.signal.sosfreqz(sos=self.sos, worN=plot_fftbins, fs=sr)

    
    def plot_frequency_response(self):
        _plt_use_style()

        plt.plot(self.w / np.pi, 20 * np.log10(np.maximum(np.abs(self.h), 1e-5)))

        plt.xlabel("Frequency [Hz]")

        plt.ylabel("Frequency Response [dB]")

    
    def plot_phase_response(self):
        _plt_use_style()

        plt.plot(self.w / np.pi, np.angle(self.h))

        plt.xlabel("Frequency [Hz]")

        plt.ylabel("Phase Response [Rad]")
    

    def __call__(self, signal: ndsignal):
        return ndsignal(samples=scipy.signal.sosfiltfilt(self.sos, signal.ndarray, axis=1), sr=signal.sr)


