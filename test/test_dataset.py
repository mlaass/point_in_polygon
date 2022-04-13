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
    print("poly_count: ", poly_count)
    polys = h5py.File(cfg.polyfile, "r")
    points = h5py.File(cfg.pointfile, "r")

    print("build boost polys")
    bt = pip.PolyBoost(polys["polygons"][:poly_count], polys["coords"][:])

    bt_results = []
    # too slow oterhwise
    if(poly_count < 120):
        print("test bt")
        bt_results = bt.test(points["coords"][:])
    bt_rd = dict(bt_results)

    print("test_para bt")
    bt_results_para = bt.test_para(points["coords"][:])
    bt_rdp = dict(bt_results_para)
    print(len(bt_results), len(bt_results_para))
    print("equal results: ", bt_rd == bt_rdp)

    print("\n***********************\n")

    print("build edge list polys")
    el = pip.PolyEdgeList(polys["polygons"][:poly_count], polys["coords"][:])

    print("test el")
    el_results = el.test(points["coords"][:])
    el_rcr = dict(el_results)

    print("test el rt")
    el.build_rtree()
    el_results_rt = el.test_rt(points["coords"][:])
    el_rcr_rt = dict(el_results_rt)

    print("test el rt para")
    el_results_rt_para = el.test_rt_para(points["coords"][:])
    el_rcr_rtp = dict(el_results_rt_para)

    print("test el para")
    el_results_para = el.test_para(points["coords"][:])
    el_rcrp = dict(el_results_para)

    print("test el para2")
    el_results_para2 = el.test_para2(points["coords"][:])
    el_rcrp2 = dict(el_results_para2)

    # print("test el winding")
    # results_winding = el.test_winding(points["coords"][:])
    # el_rwd = dict(results_winding)

    # print(stats_convert(bt.stats()))
    # print(stats_convert(el.stats()))

    print(len(el_results), len(
        bt_results_para), len(el_results_para), len(el_results_para2))
    print("equal results: ", el_rcr == bt_rdp,
          el_rcr == el_rcrp,  el_rcrp == el_rcrp2)

    print("\n***********************")

    print("build opencl polys")
    ocl = pip.OpenCLImpl(polys["polygons"][:poly_count], polys["coords"][:])

    print("test ocl")
    ocl_results_naive = ocl.test_naive(points["coords"][:])
    ocl_rn = dict(ocl_results_naive)
    print(len(el_results), len(ocl_results_naive))
    print("equal results: ", el_rcr == ocl_rn)

    # print(stats_convert(ocl.stats()))
    print("===> boost polygon\n")
    print_times(bt.stats())
    print("===> edge list polygon\n")
    print_times(el.stats())
    print("===> opencl\n")
    print_times(ocl.stats())
