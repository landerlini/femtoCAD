class Laser:
    def __init__ (self, numerical_aperture):
        self._numerical_aperture = numerical_aperture

    @property
    def numerical_aperture (self):
        return self._numerical_aperture

