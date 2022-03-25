import mpcl
import h5py
import os
dirname = os.path.dirname(__file__)


class cfg:
    filename = os.path.join(dirname, "modelnet10_sample.h5")


with h5py.File(cfg.filename, "r") as f:
    sizes = {}
    for grp in list(f.keys()):
        pc = mpcl.pointcloud(f[grp]["coords"])
        print(grp)
        for k in range(3, 8):
            features, neighbors = pc.extractKnnTensorsAndNeighbors(k)
            print(k, features.shape, neighbors.shape)
