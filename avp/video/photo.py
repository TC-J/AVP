from functools import reduce
from typing import Self

import PIL.Image

import cv2

from pathlib import Path

import numpy as np

import re

from matplotlib.pyplot import imshow

import PIL

from PIL import Image

class ndphoto(np.ndarray):
    def load(fpath: str):
        image = PIL.Image.open(fpath)

        return ndphoto(
            pixels=np.array(image), 
            cmode=image.mode
        )
    

    def __new__(cls, pixels: np.ndarray | list, cmode: str = "BGR"):
        obj = np.asarray(pixels).view(cls)

        obj._cmode = cmode if cmode.isupper() else cmode.upper()

        return obj


    def __array_finalize__(self, obj):
        if obj is None: return

        self._cmode = getattr(obj, "_cmode", None)
    

    @property
    def cmode(self):
        if len(self._cmode.split("_")) == 2:
            self._cmode = self._cmode.split("_")[0]

        return self._cmode

    
    @cmode.setter
    def cmode(self, to: str):
        if self.convert_mode(to) is not None:
            raise Exception("Use ndphoto.convert on color conversions that are not the same shape stomp blink")
    

    @property
    def size(self) -> tuple:
        return self.shape[1], self.shape[0]


    def resize(self, width: int | float, height: int | float)-> Self:
        if width == -1:
            width = self.shape[1]
        
        if height == -1:
            height = self.shape[0]
        
        if isinstance(width, float):
            width = self.shape[1] * width
        
        if isinstance(height, float):
            height = self.shape[0] * height

        resized = cv2.resize(self, (width, height), cv2.INTER_LANCZOS4)

        return ndphoto(
            pixels=np.array(resized),
            cmode=self.cmode
        )
    

    def convert_mode(self, to: str = "RGB", try_inplace = True) -> Self | None:
        to = to if to.isupper() else to.upper()

        if len(self._cmode.split("_")) == 2:
            _format = self._cmode.split("_")[0]

        formats = ndphoto.available_cmode_conversions()

        if self._cmode + " -> " + to not in formats:
            to = [name for name in formats if re.search(self._cmode + " -> " + to, name)][0].split(" -> ")[1]

        codename = "cv2.COLOR_" + self._cmode + "2" + to

        new = cv2.cvtColor(
            src=self.view(np.ndarray), 
            code=eval(codename)
        )

        if new.shape == self.shape and try_inplace:
            np.copyto(src=new, dst=self, casting="unsafe")

            self._cmode = to

        else:
            return ndphoto(new, cmode=to)
    
    
    def show(self):
        imshow(self)


    def save(self, fpath: str | Path):
        image = PIL.Image.fromarray(self, mode=self.cmode)

        image.save(fpath)


    def available_cmode_conversions() -> list[str]:
        constants = ["_".join(name.split("_")[1:]).replace("2", " -> ", 1) for name in dir(cv2) if name.isupper() and name.startswith("COLOR_") and not re.search("BAYER", name)]

        return constants

