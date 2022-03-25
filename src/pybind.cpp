#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <functional>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// This holds the actual implementation. Copy this header to your projects (and
// a hasher, for example murmur.hpp)

#include "mpcl.hpp"

// Wrap 2D C++ array (given as pointer) to a numpy object.
template <typename T>
static py::array_t<T> wrap2D(T *data, size_t h, size_t w) {

  auto shape = {h, w};
  auto strides = std::vector<size_t>({sizeof(T) * w, sizeof(T)});
  auto caps = py::capsule(
      data, [](void *v) { /*delete reinterpret_cast<double *>(v);*/ });

  return py::array_t<T, py::array::forcecast | py::array::c_style>(
      shape, strides, data, caps);
}

// This template is a handy tool to call a function f(i,j,value) for each entry
// of a 2D matrix self. template<typename func> static void
void map_matrix(const py::array_t<double> &self,
                std::function<void(int, int, double)> f) {
  if (self.ndim() != 2)
    throw(std::runtime_error("2D array expected"));
  auto s1 = self.strides(0);
  auto s2 = self.strides(1);
  const char *data = reinterpret_cast<const char *>(self.data());

  for (int i1 = 0; i1 < self.shape(0); i1++) {
    for (int i2 = 0; i2 < self.shape(1); i2++) {
      size_t offset = i1 * s1 + i2 * s2;
      // std::cout <<"("<< offset<<", "<<i1 <<", "<<i2 <<"), ";
      const double *d = reinterpret_cast<const double *>(data + offset);
      f(i1, i2, *d);
    }
  }
  std::cout << std::endl;
}
template <typename T>
void map_pointcloud(const py::array_t<T> &self,
                    std::function<void(int, int)> f) {
  if (self.ndim() != 2)
    throw(std::runtime_error("2D array expected"));
  auto s1 = self.strides(0);
  auto s2 = self.strides(1);
  const char *data = reinterpret_cast<const char *>(self.data());

  for (int i1 = 0; i1 < self.shape(0); i1++) {
    size_t offset0 = i1 * s1;
    size_t offset1 = i1 * s1 + s2;
    // std::cout <<"("<< offset<<", "<<i1 <<", "<<i2 <<"), ";
    const double *d0 = reinterpret_cast<const double *>(data + offset0);
    const double *d1 = reinterpret_cast<const double *>(data + offset1);
    f((int)*d0, (int)*d1);
  }
  std::cout << std::endl;
}

// The module begins
PYBIND11_MODULE(mpcl, m) {
  m.doc() = "This is a Python binding of Martin's Pointcloud Library (MPCL)";
  py::class_<mpcl::pointcloud>(m, "pointcloud")
      .def(py::init([](py::array_t<_Float32> points) {
        auto self = new mpcl::pointcloud();
        auto r = points.unchecked<2>();
        if (r.shape(1) != 3) {
          throw std::runtime_error("Input shape must be (n, 3)");
        }
        for (py::ssize_t i = 0; i < r.shape(0); i++)
          self->coords.push_back(point(r(i, 0), r(i, 1), r(i, 2)));
        self->buildIndex();
        return self;
      }))
      .def(
          "extractKnnTensors",
          +[](mpcl::pointcloud &self, size_t k) {
            std::vector<std::vector<double>> features;
            self.extractKnnTensors(k, features);
            return wrap2D<double>((double *)&features[0], features.size(),
                                  features[0].size());
          })
      .def(
          "extractKnnTensorsAndNeighbors",
          +[](mpcl::pointcloud &self, size_t k) {
            std::vector<std::vector<double>> features;
            std::vector<std::vector<double>> neighbors;
            self.extractKnnTensorsAndNeighbors(k, features, neighbors);
            return std::make_tuple(
                std::move(wrap2D<double>((double *)&features[0],
                                         features.size(), features[0].size())),
                std::move(wrap2D<_Float32>((_Float32 *)&neighbors[0],
                                           neighbors.size(),
                                           neighbors[0].size())));
          })

      ;
}