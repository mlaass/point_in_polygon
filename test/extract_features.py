#!/bin/python3
import io
import sys
import os
import glob
import h5py
from tqdm import tqdm
import numpy as np

import mpcl
config = {
    "datapath": "/data/ssd/moritz/tf/pointclouds/data/"
}

dirname = os.path.dirname(__file__)


files = glob.glob(f'{config["datapath"]}**.h5', recursive=False)
print(files)
files = [os.path.join(dirname, "modelnet10_sample.h5")]

for file in files[:1]:
    outfile = f'{file[:-3]}_features.h5'
    print("******* extract: ", file, "->", outfile)

    with h5py.File(file, "r") as f:
        for grp in tqdm(list(f.keys())):
            pc = mpcl.pointcloud(f[grp]["coords"])
            g = f.require_group(grp)
            print(grp, list(g.keys()), f[grp]["coords"].shape)
            # for k in (range(6, 13)):
            k = 6
            features, neighbors = pc.extractKnnTensorsAndNeighbors(k)
            print(k, neighbors.shape, neighbors.dtype)
            print(neighbors[0])
            nfeat = f"features_k{k}"
            nneigh = f"neighbors_k{k}"
            # if nfeat in g:
            #     del g[nfeat]
            # if nneigh in g:
            #     del g[nneigh]

            # with h5py.File(outfile, "a") as fout:
            #           neighbors.shape, neighbors.dtype)
            #     gout = fout.require_group(grp)
            #     #gout.create_dataset(nfeat, data=features)
            #     gout.create_dataset(nneigh, data=neighbors)
