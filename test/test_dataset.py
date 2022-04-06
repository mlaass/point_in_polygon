#!/bin/python3
import point_in_polygon as pip
import h5py
import os
dirname = os.path.dirname(__file__)


class cfg:
    polyfile = os.path.join(dirname, "GlobalLSIB_Polygons_Detailed_300p.h5")
    pointfile = os.path.join(dirname, "twitter_1mio_coords.h5")


def stats_convert(stats):
    for s in list(stats.keys()):
        if s.endswith("_ns"):
            stats[f"{s[:-3]}_s"] = stats[s]*1e-9
    return stats


if __name__ == "__main__":
    print("start")
    polys = h5py.File(cfg.polyfile, "r")
    points = h5py.File(cfg.pointfile, "r")

    print("build zk")
    zk = pip.ZonalKey(polys["polygons"][:200], polys["coords"][:])
    print(stats_convert(zk.stats()))

    print("test zk")
    results = zk.test(points["coords"][:])
    print(stats_convert(zk.stats()))
    # results = [(i, s)
    #            for i, s in zip(range(len(results)), results) if len(s) != 0]

    print("test_para zk")
    results_para = zk.test_para(points["coords"][:])
    print(stats_convert(zk.stats()))

    # results_para = [(i, s)
    #                 for i, s in zip(range(len(results_para)), results_para) if len(s) != 0]

    rd = dict(results)
    rdp = dict(results_para)
    print(len(results), len(results_para))
    for r in rd:
        if(not r in rdp):
            print("not in rdp: ", r)
        else:
            if(rd[r] != rdp[r]):
                print(r, rd[r], rdp[r])
