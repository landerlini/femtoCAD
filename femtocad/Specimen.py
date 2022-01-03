import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from io import StringIO

from femtocad.material_tab import material_tab


class Specimen:
    def __init__ (self, depth, x_side=None, y_side=None, material=None, origin=None):
        self.depth = depth
        self.origin = origin if origin is not None else np.array([0., 0., 0.], dtype=np.float64)
        self.x_side = x_side
        self.y_side = y_side
        self.material = material

        self.material_tab = self.load_material_tab()

    @staticmethod
    def load_material_tab ():
        df = pd.read_csv(StringIO(material_tab), delimiter='\s+,\s+', engine='python')
        df.columns = [s.replace(" ","") for s in df.columns]
        df.set_index('Material', inplace=True)
        return df

    def draw (self):
        if self.x_side is None or self.y_side is None:
            return

        vertices = np.stack([self.origin, self.origin + np.array([self.x_side, self.y_side, self.depth])])


        for x0,y0,z0, x1,y1,z1 in product(*[[0,1] for _ in range(6)]):
            if any ([x0 > x1, y0 > y1, z0 > z1]):
                continue

            if np.count_nonzero([x0 == x1, y0 == y1, z0 == z1]) == 2:
                x, y, z = vertices.T
                plt.plot([x[x0], x[x1]], [y[y0], y[y1]], [z[z0], z[z1]], color='#88cccc', alpha=0.8, linewidth=1)

    @property
    def refractive_index (self):
        return float(self.material_tab.loc[self.material])
