import numpy as np
from copy import deepcopy as copy


class Segment:
    """
    Segment class, representing a graphite line
    """
    def __init__ (self, start, stop, side='auto', category=0, radius=0.010):
        if side not in ['auto', 'front', 'back']:
            raise KeyError(f"Unexpected side definition {side}: can be 'front', 'back' or 'auto'")
        self.start = np.asarray(start).astype(np.float64).copy()
        self.stop = np.asarray(stop).astype(np.float64).copy()
        self._side = side
        self.category = category
        self.radius = radius

    @property
    def side(self):
        return self._side

    def points (self, n_points=100, **plot_config):
        "Return the coordinates of segments along the graphite line"
        return np.linspace(self.start, self.stop, n_points)

    def __str__ (self):
        return "\n".join([
            f"Segment {id(self):x}",
            f"Start: [{self.start[0]:5.3f}, {self.start[1]:5.3f}, {self.start[2]:5.3f}]",
            f"Stop : [{self.stop[0]:5.3f}, {self.stop[1]:5.3f}, {self.stop[2]:5.3f}]"
            ])

    def split (self, coord, boundary):
        if isinstance(coord, str) and coord.lower() in ['x', 'y', 'z']:
            coord = 'xyz'.index(coord.lower())

        s1, s2 = copy(self), copy(self)
        s1.stop[coord] = boundary
        s2.start[coord] = boundary
        return s1, s2

    @property
    def x0 (self):
        return self.start[0]

    @property
    def y0 (self):
        return self.start[1]

    @property
    def z0 (self):
        return self.start[2]

    @property
    def x1 (self):
        return self.stop[0]

    @property
    def y1 (self):
        return self.stop[1]

    @property
    def z1 (self):
        return self.stop[2]
