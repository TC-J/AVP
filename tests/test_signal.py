import unittest

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
            self.test_ndarrays.append(np.tile(self.test_samples, (len(self.test_samples), channels)))


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



    def test_sample_indexing(self):
        """test the indexing of the ndsignals with sample indices."""
        for test_ndarray, channels in self.test_ndarrays, self.test_channel_counts:
            assert np.all(test_ndarray == ndsignal(test_ndarray))
    
    
    def test_time_indexing(self):
        pass