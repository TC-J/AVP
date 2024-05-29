from functools import reduce
from typing import Self

import cv2

from pathlib import Path

import numpy as np

import re

from matplotlib.pyplot import imshow


class ndphoto(np.ndarray):
    def __new__(cls, pixels: np.ndarray | list, format: str = "BGR"):
        obj = np.asarray(pixels).view(cls)

        obj._format = format if format.isupper() else format.upper()

        return obj


    def __array_finalize__(self, obj):
        if obj is None: return

        self._format = getattr(obj, "_format", None)
    

    @property
    def format(self):
        if len(self._format.split("_")) == 2:
            self._format = self._format.split("_")[0]

        return self._format

    
    @format.setter
    def format(self, to: str):
        if self.convert(to) is not None:
            raise Exception("Use ndphoto.convert on color conversions that are not the same shape stomp blink")
    

    def sift(self, contrast_th = 0.03, edge_th = 10):
        sift = cv2.SIFT_create()

        sift.setContrastThreshold(contrast_th)

        sift.setEdgeThreshold(edge_th)

        gray = self.convert("GRAY", False)

        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return keypoints, descriptors
    

    def convert(self, to: str = "RGB", try_inplace = True) -> Self | None:
        to = to if to.isupper() else to.upper()

        if len(self._format.split("_")) == 2:
            _format = self._format.split("_")[0]

        formats = ndphoto.available_format_conversions()

        if self._format + " -> " + to not in formats:
            to = [name for name in formats if re.search(self._format + " -> " + to, name)][0].split(" -> ")[1]

        codename = "cv2.COLOR_" + self._format + "2" + to

        new = cv2.cvtColor(
            src=self.view(np.ndarray), 
            code=eval(codename)
        )

        if new.shape == self.shape and try_inplace:
            np.copyto(src=new, dst=self, casting="unsafe")

            self._format = to

        else:
            return ndphoto(new, format=to)
    
    
    def show(self):
        imshow(self)
    
    
    def keypoints_show(self, color=(160, 32, 240), contrast_th=0.03, edge_th=10):
        keypoints, descriptors = self.sift(contrast_th, edge_th)

        img = self.copy()

        img = cv2.drawKeypoints(image=copy_self, keypoints=keypoints, outImage=None, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        imshow(img)


    def available_format_conversions() -> list[str]:
        constants = ["_".join(name.split("_")[1:]).replace("2", " -> ", 1) for name in dir(cv2) if name.isupper() and name.startswith("COLOR_") and not re.search("BAYER", name)]

        return constants

