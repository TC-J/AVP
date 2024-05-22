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


class ndsignal(ndarray):
    def __new__(cls, samples, sr: int = 22_050, into_sample_major: bool = True):
        obj = np.asarray(samples).view(cls)

        if into_sample_major:
            obj = obj.transpose()

        obj._sr = sr

        obj._in_sample_major = into_sample_major

        return obj


    def __array_finalize__(self, obj):
        if obj is None: return

        self._sr = getattr(obj, "_sr", None)

        self._in_sample_major = getattr(obj, "_in_sample_major", True)
    
    
    def __getitem__(self, key):
        return super().__getitem__(self._key(key))
    
    
    def __setitem__(self, key, value):
            super().__setitem__(self._key(key), value)

    
    def _keyshape(self, key):
        if isinstance(key, int | float):
            return (1, self.channels) if self.in_sample_major else (self.channels, 1)
        
        elif isinstance(key, slice):
            modified_slice = self._slice(key)

            samples = (modified_slice.stop - modified_slice.start) // modified_slice.step

            return (samples, self.channels) if self.in_sample_major else (self.channels, samples)
        
        # key is a tuple
        else:
            subkeyshapes = list()

            for subkey in list(key):
                subkeyshapes.append(self._keyshape(subkey))
            
            return (subkeyshapes[0][0] + subkeyshapes[1][0], self.channels) if self.in_sample_major else (self.channels, subkeyshapes[0][1] + subkeyshapes[1][1])


    def _index(self, index: float | int):
        return index if isinstance(index, int) else round(index / self.T)


    def _slice(self, slc: slice):
        slc.start = slc._index(slc.start) if slc.start else 0

        slc.stop = slc._index(slc.stop) if slc.stop else self.N

        slc.step = slc._index(slc.step) if slc.step else 1

        return slc


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
        return self.shape[1] if self.in_sample_major else self.shape[0]
    

    @property
    def N(self) -> int:
        """the total number of samples."""
        return self.shape[0] if self.in_sample_major else self.shape[1]
    

    @property
    def T(self) -> float:
        """the period; the time-elapsed for a single sample."""
        return 1 / self.sr
    
    
    @property
    def duration(self) -> float:
        """the total time elapsed by this signal."""
        return self.T * self.N
    

    @property
    def in_sample_major(self):
        """the shape is (number_of_samples, number_of_channels) when true and (number_of_channels, number_of_samples) when false."""
        return self._in_sample_major
    
    
    @in_sample_major.setter
    def in_sample_major(self, value: bool):
        """the shape is (number_of_samples, number_of_channels) when true and (number_of_channels, number_of_samples) when false."""
        if (not self._in_sample_major and value) or (self._in_sample_major and not value):
            self.resize((self.shape[1], self.shape[0]))

        self._in_sample_major = value
    

    def as_channel_major(self) -> Self:
        """return the ndsignal in channel-major; in the shape: (n_channels, n_samples)."""
        self.in_sample_major = False

        return self
    
    
    def as_sample_major(self) -> Self:
        """return the ndsignal in sample-major; in the shape: (n_samples, n_channels)."""
        self.in_sample_major = True

        return self


class signal:
    def __init__(self, ndsig: ndsignal):
        self.inner = ndsig
    

    def __getitem__(self, key):
        return self.inner.__getitem__(key)
    
    
    def __setitem__(self, key, value):
        self.inner.__setitem__(key, value)