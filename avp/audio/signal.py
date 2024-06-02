from numbers import Complex

from types import NoneType

from typing import Self, Any, Tuple

import numpy as np

from numpy.typing import NDArray

from numpy import complex128, ndarray

import matplotlib.pyplot as plt

import scipy

import scipy.signal

import scipy.fft

import librosa

import librosa.feature

import librosa.display

import pydub

from avp.audio.util import _standardize_array, _pretty_time_format, _plt_use_style

class ndsignal(ndarray):
    """this is an array of audio samples in sample-major `(n_samples, n_channels)`.

        this is an `ndarray` derived class with information like the sample rate and duration of an audio array added in.

        this allows for both time (using float indices)- and sample (using integer indices)-based indexing of the samples;

        indexing with `1.0` means the sample at one-second while indexing with `1` means the first sample.

        slicing works the same way: provide floats for time, and integers for sample-numbers.

        when using operations on an `ndsignal` with an `ndarray`, the `ndarray` is expected to be in channel-major `(n_channels, n_samples)`; or `(n_samples,)`. *see NOTE*

        ## NOTE

        outside `ndarray` audio arrays used with this class should be in channel-major: `(n_channels, n_samples)`; or `(n_samples,)`.

        we assume all `ndsignal` instances are in sample-major while all `ndarray` instances (of audio samples) are in channel-major  `(n_channels, n_samples)` or `(n_samples,)`.

        all `ndarray` instances are converted into a the cooperative format needed internally before operating on the provided `ndarray`; so any outside libraries interacting with this should be in channel-major while our arrays are in sample-major.

        this is important to note; this helps us; throughout the library we make the above assumption for convienience and standardization.
    """

    def load(fpath: str, time_start: float = 0., duration: float | None = None, resample_hz = None, to_mono: bool = False):
        samples, sr = librosa.load(fpath, offset=time_start, duration=duration, mono=to_mono, sr=resample_hz)

        return ndsignal(
            samples=samples, 
            sr=sr
        )


    def __new__(cls, samples, sr: int = 22_050, _no_transpose = False):
        """create a new ndsignal; the samples are expected to be in channel-major `(n_samples,)`; or `(n_channels, n_samples)` and will be converted into sample-major, internally -- unless _notranspose is true."""
        obj = np.asarray(samples).view(cls)

        if len(list(obj.shape)) == 1:
            obj = np.expand_dims(obj, 0)

        obj = obj.transpose() if not _no_transpose else obj

        obj._sr = sr

        return obj


    def __array_finalize__(self, obj):
        if obj is None: return

        self._sr = getattr(obj, "_sr", None)

    
    def __getitem__(self, key):
        samples = super().__getitem__(self._key(key))

        if np.isscalar(samples):
            return samples

        elif isinstance(key, tuple):
            if isinstance(key[1], int):
                return ndsignal(samples, self.sr)

            else:
                return ndsignal(samples, self.sr, True)

        else:
            return ndsignal(samples, self.sr, True)
    
    
    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray | list):
            value = _standardize_array(value)

        super().__setitem__(self._key(key), value)


    def __str__(self) -> str:
        duration = _pretty_time_format(time=self.duration, starting_places=12, unit_delim=" ", fullunitname=True)

        period = _pretty_time_format(time=self.T, starting_places=12, unit_delim=" ", fullunitname=True)

        return f"NDSignal(\n Total Duration = {duration},\n Time Per Sample (Period; T) = {period},\n Samples Per Channel (N) = {self.N:,},\n Channels = {self.channels},\n Sampling Rate = {self.sr:,} hertz\n)"
    
    
    def __repr__(self):
        return super().__repr__()

    
    def _index(self, index: float | int):
        return index if isinstance(index, int) else round(index / self.T)


    def _slice(self, slc: slice):
        start = self._index(slc.start) if slc.start else 0

        stop = self._index(slc.stop) if slc.stop else self.N

        step = self._index(slc.step) if slc.step else 1

        return slice(start, stop, step)


    def _key(self, key: tuple | slice | int | float):
        if isinstance(key, float | int):
            return self._index(key)

        elif isinstance(key, slice):
            return self._slice(key)
        
        else:
            modified_key = list()

            for subkey in list(key):
                if isinstance(subkey, float | int):
                    modified_key.append(self._index(subkey))
                
                elif isinstance(subkey, NoneType):
                    modified_key.append(np.newaxis)

                else:
                    modified_key.append(self._slice(subkey))
            
            return tuple(modified_key)

    
    @property
    def sr(self):
        return self._sr
    

    @property
    def channels(self) -> int:
        """the number of channels."""
        return self.shape[1]
    

    @property
    def N(self) -> int:
        """the total number of samples."""
        return self.shape[0]
    

    @property
    def T(self) -> float:
        """the period; the time-elapsed for a single sample."""
        return 1 / self.sr
    
    
    @property
    def duration(self) -> float:
        """the total time elapsed by this signal."""
        return self.T * self.N


    @property
    def ndarray(self) -> np.ndarray:
        return self.transpose().view(np.ndarray)
    

    @property
    def peak(self):
        return np.max(np.abs(self.ndarray))


    @property
    def dynamic_range(self):
        return 20 * np.log10(self.peak / self.floor)


    @property
    def floor(self):
        return np.min(np.abs(self.ndarray))


    @property
    def time_domain(self) -> np.ndarray:
        return np.arange(self.N) / self.sr


    @property
    def frequency_domain(self):
        return librosa.fft_frequencies(sr=self.sr, n_fft=self.N)[1:]
    
    
    @property
    def harmonics(self) -> Tuple[NDArray[complex128] | complex128, NDArray[complex128]]:
        """
            computes the FFT on each channel.

            ### Returns
            `(dc, ffts) -> NDarray[Complex], NDArray[Complex]`: returns the direct-current for each channel (average magnitude; magnitude of 0 Hertz),
            indexed as `dc[nth_channel] -> Complex Valued DC of Channel`;
            and then, the FFTs for each channel; i.e., the index `ffts[nth_channel, nth_frequency_bin]` returns the complex-valued frequency bin on that channel.
            
        """
        # NumPy's FFT
        np_fft = np.fft.fft(self.ndarray, axis=1)

        # first value on all channels; 0 Hz Frequency Bin
        dc = np_fft[:, 0] 

        # all channels; exclude 0 Hz and only include positive Frequency Bins
        ffts = np_fft[:, 1:self.N // 2 + 1] 

        return dc, ffts


    def into_mono(self, interleave: bool = False) -> Self:
        if not interleave:
            samples= np.sum(self, axis=1) / self.channels

            samples = samples[:, np.newaxis]

        else:
            samples = np.zeros((self.N * self.channels, 1))

            index = 0

            for channel in range(self.channels):
                for sample in range(self.N):
                    samples[index, 0] = self[sample, channel]

                    index += 1

        return ndsignal(
            samples=samples,
            sr=self.sr,
            _no_transpose=True
        )
    

    def peak_normalize(self):
        return ndsignal(
            samples=self / self.peak,
            sr=self.sr,
            _no_transpose=True
        )

    
    def normalize(self, bit_width: int | None = None):
        """Full Scale Normalize the audio samples based on bit-width (equivalently, sample width * 8)."""
        bit_width = bit_width - 1 if bit_width else np.dtype(self.dtype).itemsize * 8 - 1

        maxval = 2**bit_width - 1

        factor = maxval / self.peak

        samples = self * factor
        
        return ndsignal(
            samples=samples, 
            sr=self.sr,
            _no_transpose=True
        )
    

    def as_db(self, ref_value: int | float):
        for channel in range(self.channels):
            for sample in range(self.N):
                self[sample, channel] = 20 * np.log10(self[sample, channel] / ref_value)
    

    def as_peak_db(self):
        self.as_db(self.peak)


    def spectra(self, frame_width_ms: float = 2.5, frame_hop_ms: float = 1.0, scaling="spectrum", mono=False):
        window_size = int(frame_width_ms / 1e3 * self.sr)

        hop_size = int(frame_hop_ms / 1e3 * self.sr)

        nfft = 2**np.ceil(np.log2(window_size))

        noverlap = window_size - hop_size

        return scipy.signal.spectrogram(self.ndarray if not mono else self.into_mono().ndarray, self.sr, nperseg=window_size, nfft=nfft, noverlap=noverlap, scaling=scaling, axis=1)


    def plot(self):
        _plt_use_style()

        signal = self.into_mono().peak_normalize()

        plt.plot(signal.time_domain, signal.ndarray[0])

        plt.ylabel("Signal Amplitude")

        plt.xlabel("Time [s]")

        plt.ylim((-1., 1.))
    
    
    def harmonic_plot(self):
        signal = self.into_mono().peak_normalize()

        _, ffts = signal.harmonics

        f = signal.frequency_domain

        db = librosa.amplitude_to_db(S=np.abs(ffts[0]), ref=np.max(np.abs(ffts[0])))

        _plt_use_style()

        plt.plot(f, db)

        plt.xlabel("Frequency [Hz]")

        plt.ylabel("Frequency Amplitude [dB]")
        
    
    def spectral_power_plot(self, frame_width_ms: float = 2.5, frame_hop_ms: float = 1.0):
        f, t, Sxx = self.spectra(frame_width_ms=frame_width_ms, frame_hop_ms=frame_hop_ms, scaling="spectrum", mono=True)

        _plt_use_style()

        plt.pcolormesh(t, f, Sxx[0], cmap="coolwarm", shading="gouraud", vmax=np.max(Sxx[0]), vmin=np.min(Sxx[0]), norm="log")

        plt.ylim((0, self.sr // 2))

        plt.ylabel("Frequency [Hz]")

        plt.xlabel("Time [s]")

        plt.colorbar(label="Spectral Power")


    def spectral_density_plot(self, frame_width_ms: float = 2.5, frame_hop_ms: float = 1.0):
        f, t, Sxx = self.spectra(frame_width_ms=frame_width_ms, frame_hop_ms=frame_hop_ms, scaling="density", mono=True)

        _plt_use_style()

        plt.pcolormesh(t, f, Sxx[0], cmap="coolwarm", shading="gouraud", vmax=np.max(Sxx[0]), vmin=np.min(Sxx[0]), norm="log")

        plt.ylim((0, self.sr // 2))

        plt.ylabel("Frequency [Hz]")

        plt.xlabel("Time [s]")

        plt.colorbar(label="Spectral Density")