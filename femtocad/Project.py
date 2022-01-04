import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from femtocad.Collection import Collection

EPSILON = 0.010 # mm


class Project (Collection):
    def __init__ (self, name, laser, specimen):
        Collection.__init__(self, name)
        self.laser = laser
        self.specimen = specimen
        self._frozen = False
        self._validated = None
        self.warnings = []

    def get_angle(self):
        angle = np.arcsin (self.laser.numerical_aperture / self.specimen.refractive_index)
        return angle

    def freeze(self):
        self._frozen = True

    def __hash__ (self):
        return hash(self.dataframe.to_json)

    def crop_invalid (self):
        while True:
            need_reprocessing = False
            self.dataframe['valid'] = True

            s = self.specimen
            for iCoord, side in [(2, s.depth), (0, s.x_side), (1, s.y_side)]:
                if side is None:
                    continue

                lower = self.specimen.origin[iCoord]
                upper = side+lower

                dropped = []
                split = []

                for idx, (seg, was_valid) in self.dataframe[['segment', 'valid']].iterrows():
                    valid = True
                    valid &= seg.start[iCoord] >= lower
                    valid &= seg.stop[iCoord] >= lower
                    valid &= seg.start[iCoord] <= upper
                    valid &= seg.stop[iCoord] <= upper

                    outside = False
                    outside |= (seg.start[iCoord] <= lower) and (seg.stop[iCoord] <= lower)
                    outside |= (seg.start[iCoord] >= upper) and (seg.stop[iCoord] >= upper)

                    if valid:
                        self.dataframe.loc[idx, 'valid'] = was_valid
                    elif outside:
                        self.dataframe.loc[idx, 'valid'] = False
                    else:
                        dropped.append(idx)
                        boundary = upper if seg.start[iCoord] > upper or seg.stop[iCoord] > upper else lower
                        s1, s2 = seg.split(iCoord, boundary)
                        split += [s1, s2]

                self.dataframe.drop(index=dropped, inplace=True)
                for seg in split:
                    self.dataframe.loc[id(seg)] = seg

                if len(split):
                    need_reprocessing = True

            if not need_reprocessing:
                break

    def assign_side (self):
        self.dataframe['back'] = False
        for idx, (seg,) in self.dataframe.query('valid == True')[['segment']].iterrows():
            if seg.side == 'front':
                self.dataframe.loc[idx, 'back'] = False
            elif seg.side == 'back':
                self.dataframe.loc[idx, 'back'] = True
            else:
                z_front = self.specimen.origin[2]
                z_back = self.specimen.origin[2] + self.specimen.depth

                z1, z0 = seg.stop[2], seg.start[2]
                z_avg = (z1+z0)/2
                z_half = (2*self.specimen.origin[2] + self.specimen.depth)/2

                if z1 <= z_front + EPSILON or z0 <= z_front + EPSILON:
                    self.dataframe.loc[idx, 'back'] = False
                elif z1 >= z_back - EPSILON or z0 >= z_back - EPSILON:
                    self.dataframe.loc[idx, 'back'] = True
                else:
                    self.dataframe.loc[idx, 'back'] = z_avg > z_half

        for idx, (seg, back) in self.dataframe.query('valid == True')[['segment', 'back']].iterrows():
            if (back and seg.start[2] > seg.stop[2]) or (not back and seg.start[2] < seg.stop[2]):
                seg.start, seg.stop = seg.stop, seg.start

    def get_validation_points(self, side):
        "Points on which to validate the sequence. Early points should not project shadows on the later ones"

        if side not in ['front', 'back']:
            raise ValueError(f"side should be either 'front' or 'back', got '{side}'.")

        df = self.dataframe.query(f'valid==True and back=={side=="back"}').copy()
        for k in ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']:
            df[k] = df['segment'].apply(lambda s: getattr(s, k))

        df.sort_values(by=['z1', 'z0', 'x1', 'y1'], inplace=True, ascending=(side == 'back'))

        points = np.empty ([0,3], dtype=np.float32)

        for idx, (seg,) in df.query('valid == True')[['segment']].iterrows():
            points = np.concatenate([points, seg.points()])

        return points

    def _validation (self, validation_points, side, already_existing=None):
        if side not in ['front', 'back']:
            raise ValueError(f"side should be either 'front' or 'back', got '{side}'.")

        if already_existing is None:
            already_existing = []

        valid = np.ones(len(validation_points), dtype=np.bool)
        for iPoint in range(1, len(validation_points)):
            v = np.concatenate(already_existing+[validation_points[:iPoint]]) - validation_points[iPoint]
            rho = np.linalg.norm(v[:,:2],axis=1)
            z = v[:,2] if side == 'back' else -v[:,2]
            theta = np.arctan2(z, rho)

            valid[iPoint] &= np.all(theta < np.pi/2 - self.get_angle())
            if not valid[iPoint]:
                print("Found and invalid point:", iPoint)

        return valid

    def validate(self):
        if self._frozen and hash(self) == self._validated:
            return True

        self.crop_invalid()
        self.assign_side()

        validation_points_front = self.get_validation_points('front')
        valid = self._validation(validation_points_front, 'front')
        if np.count_nonzero(~valid):
            self.warnings.append(
                f"Found {np.count_nonzero(~valid)}/{len(valid)} invalid points in front processing"
            )

        validation_points_back = self.get_validation_points('back')
        valid = self._validation(validation_points_back, 'back', already_existing=[validation_points_front])
        if np.count_nonzero(~valid):
            self.warnings.append(
                f"Found {np.count_nonzero(~valid)}/{len(valid)} invalid points in back processing"
            )

        self._validated = hash(self)

        return True

    def add_segment(self, *args, **kwargs):
        if self._frozen:
            raise RuntimeError("Trying to add structures to a frozen project")

        Collection.add_segment(self, *args, **kwargs)


    def draw(self):
        plt.figure(dpi=150)
        plt.subplot(1, 1, 1, projection='3d')

        self.specimen.draw()

        labels = {
            "Front process": dict(color="black"),
            "Back process": dict(color="blue"),
            "Out of specimen": dict(color="black", linestyle='dotted', alpha=0.2),
            "Shadow effects": dict(color="red", linestyle='solid'),
        }

        for id, (s, valid, back) in self.dataframe[['segment', 'valid', 'back']].iterrows():
            c = np.c_[s.start, s.stop]
            if valid:
                if back:
                    plt.plot(c[0], c[1], c[2], **labels['Back process'])
                    plt.plot(*s.start, 'o', markersize=1.5, **labels['Back process'])
                else:
                    plt.plot(c[0], c[1], c[2], **labels['Front process'])
                    plt.plot(*s.start, 'o', markersize=1.5, **labels['Front process'])
            else:
                plt.plot(c[0], c[1], c[2], **labels['Out of specimen'])

        validation_points_front = self.get_validation_points('front')
        valid_front = self._validation(validation_points_front, 'front')

        validation_points_back = self.get_validation_points('back')
        valid_back = self._validation(validation_points_back, 'back', already_existing=[validation_points_front])

        invalid_pts = np.concatenate([validation_points_front[~valid_front], validation_points_back[~valid_back]])
        plt.plot(invalid_pts[:,0], invalid_pts[:,1], invalid_pts[:,2], 'ro', markersize=1.2)

        if len(invalid_pts) == 0:
            del labels['Shadow effects']

        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.gca().set_zlabel("depth")
        ks = list(labels.keys())
        plt.legend([plt.Line2D([], [], **labels[k]) for k in ks], ks)
        plt.title(self.name)

        plt.tight_layout()
        #plt.axis('off')
        plt.gca().view_init(elev=-150, azim=50)
        plt.gca().w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.6))
        plt.gca().w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.6))
        plt.gca().w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.6))

    def export (self):
        ret = {}
        df = self.dataframe.query(f'valid==True').copy()

        def append (k, v):
            if k in ret.keys():
                ret[k].append(v)
            else:
                ret[k] = [v]

        for seg in df['segment']:
            bary = (seg.start + seg.stop)/2
            len = np.linalg.norm(seg.start - seg.stop)
            orientation = np.argmax(np.abs(seg.stop - seg.start)) + 1
            append('x', bary[0])
            append('y', bary[1])
            append('z', bary[2])
            append('length', len)
            append('radius', seg.radius)
            append('orientation', orientation)
            append('category', seg.category)

        return pd.DataFrame(ret)


