#!/bin/python3
import point_in_polygon as pip
import h5py
import os
dirname = os.path.dirname(__file__)


if __name__ == "__main__":
    print("test_opencl: ")
    print(pip.test_opencl())
