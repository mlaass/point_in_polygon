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


def print_times(stats):
    stats = stats_convert(stats)
    stats = dict(filter(lambda e: e[0].endswith("_s"), stats.items()))
    stats = dict(sorted(stats.items(), key=lambda e: e[1]))
    for s in list(stats.keys()):
        if s.endswith("_s"):
            print(f"{s}: ", stats[s])


if __name__ == "__main__":
    poly_count = 200
    print("start")
    polys = h5py.File(cfg.polyfile, "r")
    points = h5py.File(cfg.pointfile, "r")

    print("build rt")
    rt = pip.PolyRTree(polys["polygons"][:poly_count], polys["coords"][:])

    # print("test rt")
    # rt_results = rt.test(points["coords"][:])
    rt_results = []
    rt_rd = dict(rt_results)

    print("test_para rt")
    rt_results_para = rt.test_para(points["coords"][:])
    rt_rdp = dict(rt_results_para)
    print(len(rt_results), len(rt_results_para))
    print("equal results: ", rt_rd == rt_rdp)

    print("***********************")

    print("build box list")
    bl = pip.PolyBoxList(polys["polygons"][:poly_count], polys["coords"][:])

    print("test bl crossing")
    bl_results_crossing = bl.test_crossing(points["coords"][:])
    bl_rcr = dict(bl_results_crossing)

    print("test bl crossing rt")
    bl.build_rtree()
    bl_results_crossing_rt = bl.test_crossing_rt(points["coords"][:])
    bl_rcr_rt = dict(bl_results_crossing_rt)

    print("test bl crossing rt para")
    bl_results_crossing_rt_para = bl.test_crossing_rt_para(points["coords"][:])
    bl_rcr_rtp = dict(bl_results_crossing_rt_para)

    print("test bl crossing para")
    bl_results_crossing_para = bl.test_crossing_para(points["coords"][:])
    bl_rcrp = dict(bl_results_crossing_para)

    print("test bl crossing para2")
    bl_results_crossing_para2 = bl.test_crossing_para2(points["coords"][:])
    bl_rcrp2 = dict(bl_results_crossing_para2)

    # print("test bl winding")
    # results_winding = bl.test_winding(points["coords"][:])
    # bl_rwd = dict(results_winding)

    # print(stats_convert(rt.stats()))
    # print(stats_convert(bl.stats()))

    print(len(bl_results_crossing), len(
        rt_results_para), len(bl_results_crossing_para), len(bl_results_crossing_para2))
    print("equal results: ", bl_rcr == rt_rdp,
          bl_rcr == bl_rcrp,  bl_rcrp == bl_rcrp2)

    print("\n***********************")

    print("build opencl setup")
    ocl = pip.OpenCLImpl(polys["polygons"][:poly_count], polys["coords"][:])

    print("test ocl")
    ocl_results_naive = ocl.test_naive(points["coords"][:])
    ocl_rn = dict(ocl_results_naive)
    print(len(bl_results_crossing), len(ocl_results_naive))
    print("equal results: ", bl_rcr == ocl_rn)

    # print(stats_convert(ocl.stats()))
    print("===> boost polygon\n")
    print_times(rt.stats())
    print("===> simple polygon\n")
    print_times(bl.stats())
    print("===> opencl\n")
    print_times(ocl.stats())
