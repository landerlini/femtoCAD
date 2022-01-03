import matplotlib.pyplot as plt

from femtocad import Collection
from femtocad import Specimen
from femtocad import Laser

animated = True

xstrip = Collection("test")

for iY in range(3):
    y = iY * 0.1 + 0.05
    for iZ in range(4):
        z = iZ * 0.12 + 0.05
        xstrip.add_segment([0., y, z], [0.3, y, z])
        z_next = (1+iZ) * 0.120 + 0.05
        xstrip.add_segment([0., y, z_next], [0.0, y, z])
        xstrip.add_segment([0.3, y, z_next], [0.3, y, z])

for iX in range(3):
    x = iX * 0.1 + 0.05
    for iZ in range(4):
        z = iZ * 0.12 + 0.11
        xstrip.add_segment([x, 0., z], [x, 0.3, z])
        z_prev = (iZ-1) * 0.120 + 0.11
        xstrip.add_segment([x, 0.0, z_prev], [x, 0.0, z])
        xstrip.add_segment([x, 0.3, z_prev], [x, 0.3, z])


laser = Laser(numerical_aperture=1.2)
specimen = Specimen(depth=0.520, x_side=0.5, y_side=0.5, material='scvd', origin=[-0.10, -0.10, 0.])

project = xstrip.make_project(laser, specimen)

project.draw()

if not animated:
    plt.show()
else:
    import imageio
    plt.ion()
    plt.show()
    n_frames = 90
    for i in range(n_frames):
        plt.gca().view_init(elev=-150, azim=50+i*2)
        plt.savefig(f"frame{i:03d}.png")

    with imageio.get_writer('sketch.gif', mode='I') as writer:
        for i in range(n_frames):
            image = imageio.imread(f"frame{i:03d}.png")
            writer.append_data (image)


