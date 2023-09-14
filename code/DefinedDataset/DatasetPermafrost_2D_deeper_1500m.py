import numpy as np

from NN_permafrost.GeoFlow import GeoDataset, Acquisition
from NN_permafrost.GeoFlow import Vpdepth, Vsdepth, Qdepth, ShotGather, Dispersion
from NN_permafrost.GeoFlow import EarthModel
from ModelGenerator import (Property, Lithology, Deformation, Sequence,
                            Stratigraphy)
from NN_permafrost.GeoFlow import QTAU

class PermafrostModel_2D_deeper_1500m(EarthModel):
    def __init__(self):
        super().__init__()

    def build_stratigraphy(self):
        lithologies = {}

        name = "Water"
        vp = Property("vp", vmin=1430, vmax=1430)
        vpvs = Property("vpvs", vmin=0, vmax=0)
        rho = Property("rho", vmin=1000, vmax=1000)
        q = Property("q", vmin=1000, vmax=1000)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen sediments"  # Buckingham 1996, Fig 11.
        vp = Property("vp", vmin=1700 - 200, vmax=1700 + 200, texture=200)
        vpvs = Property("vpvs", vmin=4.25 - 1.25, vmax=4.25 + .45, texture=1.52)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=45, vmax=55, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen sediments bottom"
        vp = Property("vp", vmin=1700, vmax=1700 + 400, texture=200)
        vpvs = Property("vpvs", vmin=1.8, vmax=2.2, texture=0.25)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=45, vmax=55, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Sands"  # Matsushima 2016, fig 13c @ 6C Dou 2016.
        vp = Property("vp", vmin=4000 - 500, vmax=4500 + 300, texture=200)
        vpvs = Property("vpvs", vmin=2.31 - .3, vmax=2.31 + .5, texture=0.42)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=55, vmax=65, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Partially Frozen Sands"  # Matsushima 2016, fig 13c @ 3C.
        vp = Property("vp", vmin=3700 - 700, vmax=3700, texture=200)
        vpvs = Property("vpvs", vmin=2.78 - .5, vmax=2.78 + .5, texture=0.28)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=9, vmax=11, texture=3.5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Silts"  # Dou 2016, Fig 9, # Buckingham 1996, Fig 11.
        vp = Property("vp", vmin=3400 - 400, vmax=3400 + 200, texture=200)  # texture=300
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8 + .5, texture=0.29)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=40, vmax=50, texture=31.5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Dou 2016, Fig 9; Buckingham 1996, Fig 11.
        name = "Partially Frozen Silts"
        vp = Property("vp", vmin=2200 - 100, vmax=2200 + 700, texture=450)  # texture=450
        vpvs = Property("vpvs", vmin=2.78 - .5, vmax=2.78 + .5, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=18, vmax=22, texture=10)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Dou 2016, Fig 9; Buckingham 1996, Fig 11.
        name = "Partially Frozen Silts2"
        vp = Property("vp", vmin=1950, vmax=1950 + 500, texture=200)  # texture=550
        vpvs = Property("vpvs", vmin=3 - .5, vmax=3 + .5, texture=1.3)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=22, vmax=28, texture=5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Shale"  # Bellefleur 2007, Figure 3 zone 2.
        # IOE Taglu D-43.
        vp = Property("vp", vmin=3000 - 400, vmax=3500 + 600, texture=350)  # texture=950
        # vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.87)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8 + .5, texture=0.3)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # King, 1976.
        q = Property("q", vmin=90, vmax=110, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Iperk"  # Bellefleur 2007, Figure 3 zone 2.
        # IOE Taglu D-43.
        # vp = Property("vp", vmin=4000, vmax=4000, texture=1500)
        vp = Property("vp", vmin=3800 - 200, vmax=3800 + 200, texture=600)  # texture=800
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8 + .5, texture=0.7)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # King, 1976.
        q = Property("q", vmin=90, vmax=110, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen Shale"  # Bellefleur 2007, Figure 3 zone 2.
        # IOE Taglu D-43
        vp = Property("vp", vmin=2200 - 100, vmax=2200 + 300, texture=200)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8 + .5, texture=0.3)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # King, 1976.
        q = Property("q", vmin=70, vmax=100, texture=20)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Modified from Matsushima 2016, fig 13c @ 6C Dou 2016.
        name = "Frozen Sands2"
        vp = Property("vp", vmin=2600 - 100, vmax=2600 + 400, texture=300)
        vpvs = Property("vpvs", vmin=2.6 - .5, vmax=2.6 + .5, texture=0.6)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=22, vmax=28, texture=10)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Modified from Partially frozen Silts Dou 2016, Fig 9.
        # Buckingham 1996, Fig 11.
        name = "Hydrates"
        vp = Property("vp", vmin=2200 - 100, vmax=2200 + 800, texture=150)  # texture=450
        vpvs = Property("vpvs", vmin=2.78 - .5, vmax=2.78 + .5, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=18, vmax=22, texture=5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        deform = Deformation(max_deform_freq=0.02,
                             min_deform_freq=0.0001,
                             amp_max=30,
                             max_deform_nfreq=40,
                             prob_deform_change=.6)

        water = Sequence(lithologies=[lithologies["Water"]],
                         thick_min=50, thick_max=160, deform=deform)
        unfrozen1 = Sequence(lithologies=[lithologies["Unfrozen sediments"],
                                          lithologies["Partially Frozen Silts2"]],
                             deform=deform, thick_min=20,thick_max=160)
        lithologies_permafrost = [lithologies["Partially Frozen Sands"],
                                  lithologies["Frozen Shale"],
                                  lithologies["Frozen Sands"],
                                  lithologies["Partially Frozen Silts"]]
        permafrost = Sequence(lithologies=lithologies_permafrost,
                              ordered=False, deform=deform,
                              thick_min=20, thick_max=320,skip_prob=.5)
        hydrates = Sequence(lithologies=[lithologies["Unfrozen sediments"],
                                         lithologies["Hydrates"]],
                            deform=deform, thick_min=20, thick_max=120,skip_prob=0.5)
        lithologies_unfrozen2 = [lithologies["Unfrozen sediments"]]
        unfrozen2 = Sequence(lithologies=lithologies_unfrozen2,
                             deform=deform, thick_min=20, thick_max=160,skip_prob=0.5)
        lithologies_unfrozen3 = [lithologies["Unfrozen sediments bottom"]]
        unfrozen3 = Sequence(lithologies=lithologies_unfrozen3,
                             deform=deform, thick_min=20, thick_max=160,skip_prob=0.5)
        unfrozen4 = Sequence(lithologies=[lithologies["Unfrozen Shale"]],
                             deform=deform, thick_min=20)
        sequences = [water, unfrozen1, permafrost, hydrates,unfrozen2, unfrozen3, unfrozen4]
        strati = Stratigraphy(sequences=sequences)

        # Including texture to the properties summary (adapted from Stratigraphy.properties)
        properties = {p.name: [9999, 0]
                 for p in strati.sequences[0].lithologies[0]}
        for seq in strati.sequences:
            for lith in seq:
                for p in lith.properties:
                    if properties[p.name][0] > p.min - p.texture:
                        properties[p.name][0] = p.min - p.texture
                    if properties[p.name][1] < p.max + p.texture:
                        properties[p.name][1] = p.max + p.texture

        vmin = 99999
        vmax = 0
        for seq in sequences:
            for lith in seq:
                if lith.vpvs.max == 0:
                    vmin = 0
                elif vmin > lith.vp.min / (lith.vpvs.max + lith.vpvs.texture):
                    vmin = lith.vp.min / (lith.vpvs.max + lith.vpvs.texture)
                if lith.vpvs.min != 0 and vmax < lith.vp.max / (lith.vpvs.min - lith.vpvs.texture):
                    vmax = lith.vp.max / (lith.vpvs.min - lith.vpvs.texture)
        properties["vs"] = [vmin, vmax]

        return strati, properties

    def generate_model(self, seed=None):

        props2d, layerids, layers = super().generate_model(seed=seed)
        tempvpvs = props2d["vpvs"]
        tempvp = props2d["vp"]
        tempvs = tempvp*0
        mask = tempvpvs != 0
        tempvs[mask] = tempvp[mask] / tempvpvs[mask]
        props2d["vs"] = tempvs

        FL = np.array([14.543601567117028,85.73266793608842,378.648471011289])
        qtau = QTAU(7, 1000, 15, 80, FL)
        props2d["taup"] = qtau.Q2tau(props2d["q"])
        props2d["taus"] = props2d["taup"]

        return props2d, layerids, layers

class AcquisitionPermafrost_2D_deeper_1500m(Acquisition):
    """
    Fix the survey geometry for the ARAC05 survey on the Beaufort Sea.
    """
    def __init__(self,model):
        super().__init__(model)
        self.singleshot = False
        self.dh = 2.5
        self.ds = 50

    def set_rec_src(self):
        # Source and receiver positions.
        offmin = 85  # In meters.
        offmax = offmin + 120*self.dg
        if self.singleshot:
            # Add just one source at the right (offmax).
            sx = np.arange(self.Npad + offmax, 1 + self.Npad + offmax)
        else:
            # Compute several sources.
            l1 = self.Npad*self.dh + offmax + 1
            l2 = self.model.NX*self.dh - (self.Npad*self.dh)
            sx = np.arange(l1, l2, self.ds) #* self.model.dh
        sz = np.full_like(sx, self.source_depth)
        sid = np.arange(0, sx.shape[0])

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)],
                           axis=0)

        gx0 = np.arange(offmin, offmax, self.dg)
        gx = np.concatenate([s - gx0 for s in sx], axis=0)

        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        gz = np.full_like(gx, self.receiver_depth)
        gid = np.arange(0, len(gx))

        rec_pos = np.stack([gx,
                            np.zeros_like(gx),
                            gz,
                            gsid,
                            gid,
                            np.full_like(gx, 2),
                            np.zeros_like(gx),
                            np.zeros_like(gx)],
                           axis=0)

        return src_pos, rec_pos

class DatasetPermafrost_2D_deeper_1500m(GeoDataset):
    name = "DatasetPermafrost_2D_deeper_1500m"

    def __init__(self, noise=0):
        if noise == 1:
            self.name = self.name + "_noise"
        super().__init__()
        if noise == 1:
            for name in self.inputs:
                self.inputs[name].random_static = True
                self.inputs[name].random_static_max = 1
                self.inputs[name].random_noise = True
                self.inputs[name].random_noise_max = 0.02

    def set_dataset(self):
        self.trainsize = 5
        self.validatesize = 0
        self.testsize = 0

        model = PermafrostModel_2D_deeper_1500m()

        model.dh = dh = 2.5
        # nshots = 1
        # dshots = 50
        # length = nshots*dshots + 1682
        # z = 1500
        length, depth = 5000, 1500
        model.NX = int(length / dh)
        model.NZ = int(depth / dh)
        # model.texture_xrange = 3
        # model.texture_zrange = 1.95 * model.NZ / 2

        model.dip_0 = False
        model.dip_max = 15*np.pi/180
        model.ddip_max = 5

        model.layer_num_min = 12 #5 Define it as the NZ/min(thick_max) from the sequences (usually water layer)
        model.layer_dh_min = 20

        acquire = AcquisitionPermafrost_2D_deeper_1500m(model=model)
        acquire.peak_freq = 40
        acquire.dt = dt = 2e-4
        acquire.NT = int(2 / dt)
        acquire.dg = 12.5  # 5 * dh = 12.5 m.
        acquire.ds = 50
        acquire.fs = True
        acquire.source_depth = 7.5 #12.5
        acquire.receiver_depth = 7.5 #12.5
        acquire.singleshot = False

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire,
                                              random_static=False, random_noise=False, random_noise_max=.005,
                                              random_stat=True, compensate_2dto3d=True)}
        outputs = {Vsdepth.name: Vsdepth(model=model, acquire=acquire, twoD=True),
                   Vpdepth.name: Vpdepth(model=model, acquire=acquire, twoD=True),
                   Qdepth.name: Qdepth(model=model,acquire=acquire, twoD=True)}

        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True
            outputs[name].identify_direct = False

        return model, acquire, inputs, outputs

# model = PermafrostModel_2D_30dhmin_1500m()
# model.dh = dh = 2.5

# model.NX = int(length/dh)
# model.NY = int(depth/dh)

# model.dip_0 = False
# model.dip_max = 15*np.pi/180
# model.ddip_max = 5

# model.layer_num_min = 8 #5 Define it as the NZ/min(thick_max) from the sequences (usually water layer)
# model.layer_dh_min = 30

if __name__ == "__main__":
    dataset = DatasetPermafrost_2D_deeper_1500m()
    dataset.model.animated_dataset()
    temp = dataset.acquire.set_rec_src()