#!/bin/python3
import point_in_polygon as pip
import h5py
import os
dirname = os.path.dirname(__file__)


class cfg:
    polyfile = os.path.join(dirname, "eea_europe_wei_plus.h5")
    pointfile = os.path.join(dirname, "twitter_1mio_coords.h5")


if __name__ == "__main__":
    print("start")
    polys = h5py.File(cfg.polyfile, "r")
    points = h5py.File(cfg.pointfile, "r")

    print("build zk")
    zk = pip.ZonalKey(polys["polygons"][:], polys["coords"][:])
    print(zk.stats())
    print("test zk")
    results = zk.test(points["coords"][:])
    print(zk.stats())
