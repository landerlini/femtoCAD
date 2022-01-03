import numpy as np
import pandas as pd
from femtocad import Segment
import matplotlib.pyplot as plt


class Collection:
    def __init__ (self, name):
        self.name = name
        self.dataframe = pd.DataFrame(columns=['id', 'segment'])
        self.dataframe.set_index('id')

    def add_segment(self, *args, **kwargs):
        segment = Segment(*args, **kwargs)
        self.dataframe.loc[id(segment)] = segment

    def make_project (self, laser, specimen):
        from femtocad import Project  ## avoid circular import

        project = Project(self.name, laser, specimen)
        project.dataframe = self.dataframe.copy()
        project.validate()
        project.freeze()
        return project


    def draw(self):
        plt.figure(dpi=120)
        plt.subplot(1, 1, 1, projection='3d')

        for s in self.dataframe['segment']:
            c = np.c_[s.start, s.stop]
            plt.plot(c[0], c[1], c[2], color='black')

        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.gca().set_zlabel("depth")
        plt.title(self.name)

        plt.gca().view_init(elev=-150, azim=50)


