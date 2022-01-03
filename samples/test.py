import matplotlib.pyplot as plt

from femtocad import Collection
from femtocad import Specimen
from femtocad import Laser

collection = Collection("test")
laser = Laser(numerical_aperture=1.2)

for i in range(3):
    for j in range(3):
        collection.add_segment([i, j, 0], [i, j, 1])

for i in range(3):
    for j in range(3):
        collection.add_segment([i+0.5, j+0.5, 0.5], [i+0.5, j+0.5, 1])

specimen = Specimen(depth=0.500, x_side=2.5, y_side=2.5, material='scvd', origin=[-0.25, -0.25, 0.25])

project = collection.make_project(laser, specimen)
print (project.dataframe.describe())

#plt.style.use('dark_background')
project.draw()
plt.gca().grid('off')
plt.show()