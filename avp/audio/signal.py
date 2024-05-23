from ast import List
from sqlite3 import Time
from typing import Any, Self

import math
from warnings import warn

import numpy as np

from numpy.typing import NDArray

from numpy import ndarray

import matplotlib.pyplot as plt

import scipy

import scipy.signal

import scipy.fft

import resampy

import soundfile

import pydub

import librosa

import librosa.feature

import librosa.display

import pydub


class ndsignal(ndarray):

    def load(fpath: str, time_start: float = 0., duration: float | None = None, resample_hz = None):
        samples, sr = librosa.load(fpath, offset=time_start, duration=duration, mono=False, sr=resample_hz)

        return ndsignal(
            samples=samples, 
            sr=sr
        )


    def __new__(cls, samples, sr: int = 22_050, into_sample_major: bool = True):
        obj = np.asarray(samples).view(cls)

        if len(list(obj.shape)) == 1:
            obj = np.expand_dims(obj, 0)

        if into_sample_major:
            obj = obj.transpose()

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
                return ndsignal(samples, self.sr, False)

        else:
            return ndsignal(samples, self.sr, False)
    
    
    def __setitem__(self, key, value):
        super().__setitem__(self._key(key), value)

    
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
    

    def as_channel_major(self) -> Self:
        """return as an ndarray in channel-major; in the shape: (n_channels, n_samples)."""
        return self.transpose().view(np.ndarray)
    
    
    def save(self, fpath: str):
        pass


class Signal:
    def __init__(self, audio: ndsignal):
        self.inner = audio


    def __getitem__(self, key):
        return self.inner.__getitem__(key)
    
    
    def __setitem__(self, key, value):
        self.inner.__setitem__(key, value)