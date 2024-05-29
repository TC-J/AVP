import cv2

from avp.video.photo import ndphoto


class vcam:
    def __init__(self, device: int = 0):
        self.dev = device

        self.inner = cv2.VideoCapture(self.dev)
    
    
    def __str__(self) -> str:
        return f"vcam(\n device = {self.dev},\n resolution = {self.resolution},\n fps = {self.fps},\n brightness = {self.brightness},\n saturation = {self.saturation},\n gamma = {self.gamma},\n gain = {self.gain}\n)"
    

    def __repr__(self):
        return f"<vcam@{id(self)} device={self.dev}>"


    @property
    def resolution(self):
        return self.inner.get(cv2.CAP_PROP_FRAME_WIDTH), self.inner.get(cv2.CAP_PROP_FRAME_HEIGHT)

    
    @resolution.setter
    def resolution(self, width: int, height: int):
        self.inner.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        self.inner.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    
    @property
    def brightness(self):
        return self.inner.get(cv2.CAP_PROP_BRIGHTNESS)

    
    @brightness.setter
    def brightness(self, value: float):
        self.inner.set(cv2.CAP_PROP_BRIGHTNESS, value)

    
    @property
    def contrast(self):
        return self.inner.get(cv2.CAP_PROP_CONTRAST)

    
    @contrast.setter
    def contrast(self, value: float):
        self.inner.set(cv2.CAP_PROP_CONTRAST, value)


    @property
    def saturation(self):
        self.inner.get(cv2.CAP_PROP_SATURATION)


    @saturation.setter
    def saturation(self, value):
        self.inner.set(cv2.CAP_PROP_SATURATION, value)


    @property
    def exposure(self):
        return self.inner.get(cv2.CAP_PROP_EXPOSURE)


    @property
    def fps(self):
        return self.inner.get(cv2.CAP_PROP_FPS)


    @fps.setter
    def fps(self, value: float):
        self.inner.set(cv2.CAP_PROP_FPS, value)
    
    
    @property
    def format(self):
        return self.inner.get(cv2.CAP_PROP_FORMAT)
    

    @property
    def gamma(self):
        return self.inner.get(cv2.CAP_PROP_GAMMA)


    @gamma.setter
    def gamma(self, value: float):
        self.inner.set(cv2.CAP_PROP_GAMMA, value)


    @property
    def gain(self):
        return self.inner.get(cv2.CAP_PROP_GAIN)


    @gain.setter
    def gain(self, value: float):
        self.inner.set(cv2.CAP_PROP_GAIN, value)
    
    
    def photo(self) -> ndphoto:
        success, frame = self.inner.read()

        return ndphoto(frame, format="BGR") if success else None
    
    
    def done(self):
        self.inner.release()


    def cameras():
        n = 0

        while True:
            if not vcam(n).inner.isOpened(): break

            n += 1

        return n

