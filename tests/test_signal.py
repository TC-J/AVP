import unittest

import pytest

import librosa

from avp.audio.signal import ndsignal

import scipy as sp

import numpy as np

from pathlib import Path



class NDSignalTestCase(unittest.TestCase):
    def setUp(self):
        self.sample_rates = [1, 2, 10, 100, 22_050, 44_100, 48_000, 88_200, 96_000, 192_000]

        self.test_samples = np.arange(0, 32_000, 1)

        self.test_channel_counts = np.arange(1, 11, 1)

        self.test_ndarrays = list()

        for channels in self.test_channel_counts:
            self.test_ndarrays.append(np.tile(channels, (len(self.test_samples), 1)))

        self.piano_c, self.piano_c_sr = librosa.load("assets/audios/piano_c.wav", sr=None, mono=False)



    def test_creation(self):
        """test creation of signals at the industry-wide-common sample-rates.."""
        for sr in self.sample_rates:
            # create an ndsignal that is in channel-major form
            signal_0 = ndsignal([[1,2,3], [1,2,3]], sr)

            # sample rate is set.
            assert signal_0.sr == sr

            # period is set properly.
            assert signal_0.T == (1 / sr)

            # test signals created with a different sample-input-format but equivalent values..

            # create an ndsignal that is already in sample-major form.
            signal_1 = ndsignal([[1, 1], [2, 2], [3, 3]], sr=sr, into_sample_major=False)

            # signals have the same sample-rates.
            assert signal_1.sr == signal_0.sr

            # signals have the same period.
            assert signal_1.T == signal_0.T


    def test_signal_load(self):
        signal = ndsignal.load("assets/audios/piano_c.wav")

        assert self.piano_c_sr == signal.sr

        assert np.all(self.piano_c.T == signal)


    def test_sample_indexing(self):
        """test the indexing of the ndsignals with sample indices."""
        for test_ndarray in self.test_ndarrays:
            assert np.all(test_ndarray == ndsignal(test_ndarray))
        
        signal = ndsignal(self.piano_c, self.piano_c_sr)

        assert signal[1:5, 0].channels == 1

        assert signal[:, :].channels == 2

        assert signal[:].channels == 2

        assert signal[1:5].N == 4
    

    def test_time_indexing(self):
        piano_signal = ndsignal(self.piano_c, self.piano_c_sr)

        assert piano_signal[0.1:.6, 0].channels == 1

        start_sample = round(0.1 / piano_signal.T)

        stop_sample = round(0.6 / piano_signal.T)

        signal = piano_signal[0.1:0.6, :]

        assert signal.channels == piano_signal.channels

        assert signal.sr == piano_signal.sr

        assert np.all(signal == piano_signal[start_sample: stop_sample, :])

        assert signal.duration == 0.5

        assert signal.N == 0.5 * 44100


    def test_time_indices_setter(self):
        pass