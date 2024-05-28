from typing import Any, Self, List, Tuple

import math

from pathlib import Path

from warnings import warn

import numpy as np

from numpy.typing import NDArray

from numpy import full, ndarray

import matplotlib.pyplot as plt

import scipy

import scipy.signal

import scipy.fft

import resampy

import soundfile

import pydub

import pydub.effects

import librosa

import librosa.feature

import librosa.display

import pydub

from avp.audio.util import _standardize_array, _pretty_time_format, _add_channels_as_copy



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
#        if isinstance(value, np.ndarray | list):
#            value = _standardize_array(value)

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
    def peak(self):
        return np.max(np.abs(self.y))


    @property
    def floor(self):
        return np.min(np.abs(self.y))


    @property
    def t(self) -> np.ndarray:
        return np.arange(self.N) / self.sr


    @property
    def y(self) -> np.ndarray:
        return self.transpose().view(np.ndarray)
    
    
    @property
    def Fy(self) -> list[np.ndarray]:
        return [scipy.fft.fft(channel) for channel in self.y]
    
    
    @property
    def Fx(self):
        return scipy.fft.fftfreq(self.N, self.T)
    

    def insert(self, at: int | float, signl: Self | np.ndarray | list) -> None:
        """
            insert an audio signal at the given sample-index (integer provided) or time-index (float provided).
        """
        signl = _standardize_array(signl)

        at = self._index(at)

    
    def split(self, interval: float | int) -> List[np.ndarray]:
        """split the audio signal into an array of smaller ndarrays (in channel-major); the interval is a sample-index when an integer is provided and a time index when a float is provided."""
        interval = self._index(interval)

        lst = list()

        for i in range(0, self.N, interval):
            if i + interval < self.N:
                lst.append(self[i:i + interval, :].transpose().view(np.ndarray))

            else:
                lst.append(self[i:, :].transpose().view(np.ndarray))


    def save(self, fpath: str):
        pass


class signal:
    """
        this is a digital signal.

        the difference between this and `ndsignal` is that this contains an ndsignal; making it a mutable container for the ndsignal;
        this allows resampling without tracking a new variable.

        so view this as a digital-signal-processing context for the audio signals.
    """
    def generate(duration, sr, amp_range, freq_range, phase_range, noise_factors):
        pass


    def __init__(self, audio: ndsignal | None, samplewidth: int | None = None, path: str | Path | None = None, time_start: float | None, duration: float | None, resample_hz: int | None = None, channels: int | None = None):
        # allow users to load at a time-offset. ?
        start = time_start or 0

        # load audio
        # allow users to resample and specify one-channel. ?
        audio = audio or ndsignal.load(path, start, duration, resample_hz, to_mono = True if channels == 1 else False)

        channels = channels or audio.channels 

        # allow the user to increase the number of channels in the _originalinal audio. ?
        # copy the first channel into the number of requested channels.
        # this is when the audio contains less channels than what was specified.
        if channels > audio.channels:
            first_channel_data = audio[:, 0]

            audio = np.repeat(first_channel_data[:, np.newaxis], channels, axis=1)
        
        self._bitdepth = (samplewidth or np.dtype(audio.dtype).itemsize) * 8
        
        self.inner = audio


    def __getitem__(self, key):
        return self.inner.__getitem__(key)
    
    
    def __setitem__(self, key, value):
        self.inner.__setitem__(key, value)
    

    _units = "none"


    _bitdepth = None
    

    _orignal = None


    @property
    def sr(self):
        return self.inner.sr
    
    
    @sr.setter
    def sr(self, new_hz):
        self.resample(new_hz)
    
    
    @property
    def T(self):
        return self.inner.T
    
    
    @T.setter
    def T(self, new_period):
        self.resample(1 / new_period)


    @property
    def rms(self):
        return np.sqrt(np.mean(self.inner ** 2))
    
    
    @property
    def norm(self):
        return self._units


    def into_inner(self) -> ndsignal:
        return self.inner
    
    
    def resample(self, new_hz: int):
        y = resampy.resample([channel for channel in self.inner.y], self.inner.sr, parallel=True)

        self._original = self._original or self.inner.copy()

        self.inner = ndsignal(y, new_hz)
    
    
    def fullscale_norm(self):
        max_a = (2 ** self._bitdepth) - 1

        factor = max_a / self.inner.peak

        self._original = self._original or self.inner.copy()

        self.inner *= factor

        self._units = "0dBFS"

        
    def peak_norm(self):
        max_a = 1.0

        factor = 1.0 / self.inner.peak

        self._original = self._original or self.inner.copy()

        self.inner *= factor

        self._units = "1dBFS"
    
    
    def rms_norm(self, target_dBFS: float = -20.0):
        current_dBFS = self.rms

        factor = 10 ** (target_dBFS / current_dBFS) / current_dBFS

        self._original = self._original or self.inner.copy()

        self.inner *= factor

        self._units = "rms_" + str(target_dBFS) + "dBFS"
    
    
    def distortion(self):
        pass


def distortion(initial: ndsignal, final: ndsignal):
    fft_i = [channel.fft() for channel in initial.y]

    fft_f = [channel.fft() for channel in final.y]

    amplitude_distortion = [np.abs(fft_i[channel]) - np.abs(fft_f[channel]) for channel in range(fft_i.channels)]

    phase_distortion = np.angle(fft_f) - np.angle(fft_i)

    harmonic_i = [np.sum(np.abs(channel[1:]) ** 2) / np.abs(channel[0]) ** 2 for channel in fft_i]

    harmonic_f = [np.sum(np.abs(channel[1:]) ** 2) / np.abs(channel[0]) ** 2 for channel in fft_f]

    harmonic_distortion = harmonic_f - harmonic_i

    return (
        amplitude_distortion,
        phase_distortion,
        harmonic_distortion
    )
