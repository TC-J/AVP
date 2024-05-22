from typing import Self

import cv2

from pathlib import Path

import numpy as np

class ndphrame(np.ndarray):
    def __new__(cls, pixels: np.ndarray | list):
        obj = np.asarray(pixels).view(cls)

        return obj


    def __array_finalize__(self, obj):
        if obj is None: return


class phrame:
    def __init__(self, pixels: ndphrame | np.ndarray | list, format: str):
        self.inner = ndphrame(pixels) if isinstance(pixels, np.ndarray | list) else pixels

        self._format = format
    
    