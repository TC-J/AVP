from typing import Self

import cv2

from pathlib import Path

import numpy as np

class ndphoto(np.ndarray):
    def __new__(cls, pixels: np.ndarray | list, format: str = "BGR"):
        obj = np.asarray(pixels).view(cls)

        obj._format = format if format.isupper() else format.upper()

        return obj


    def __array_finalize__(self, obj):
        if obj is None: return

        self._format = getattr(obj, "_format", None)
    
    
    def convert(self, to: str = "RGB") -> Self | None:
        to = to if to.isupper() else to.upper()

        new = cv2.cvtColor(
            src=self, 
            code=eval("cv2.COLOR_" + self._format + "2" + to)
        )

        if new.shape == self.shape:
            np.copyto(src=new, dst=self, casting="unsafe")

            self._format = to

        else:
            return new
    
    
    @property
    def format(self):
        return self._format

    
    @format.setter
    def format(self, to: str):
        if not self.convert(to):
            raise Exception("Use ndphoto.convert on color conversions that are not the same shape stomp blink")
    