#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <functional>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// This holds the actual implementation. Copy this header to your projects (and
// a hasher, for example murmur.hpp)

#include "pip.hpp"
#include "timer.hpp"

// Wrap 2D C++ array (given as pointer) to a numpy object.
template <typename T>
static py::array_t<T> wrap2D(T *data, size_t h, size_t w) {

  auto shape = {h, w};
  auto strides = std::vector<size_t>({sizeof(T) * w, sizeof(T)});
  auto caps = py::capsule(
      data, [](void *v) { /*delete reinterpret_cast<double *>(v);*/ });

  return py::array_t<T, py::array::forcecast | py::array::c_style>(
      shape, strides, data);
}

template <class T>
py::array_t<T> create_matrix(size_t width, size_t height,
                             T *data_ptr = nullptr) {
  return py::array_t<T>(py::buffer_info(
      data_ptr,
      sizeof(T), // itemsize
      py::format_descriptor<T>::format(),
      2,                                                 // ndim
      std::vector<size_t>{width, height},                // shape
      std::vector<size_t>{height * sizeof(T), sizeof(T)} // strides
      ));
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

// The module begins
PYBIND11_MODULE(point_in_polygon, m) {
  m.doc() = "PIP is point in polygon tests";
  py::class_<PIP::ZonalKey<uint32_t>>(m, "ZonalKey")
      .def(py::init(
          [](py::array_t<uint32_t> polygons, py::array_t<_Float32> coords) {
            auto self = new PIP::ZonalKey<uint32_t>();
            auto p = polygons.unchecked<2>();
            auto c = coords.unchecked<2>();
            if (p.shape(1) != 2) {
              throw std::runtime_error("polygons input shape must be (n, 2)");
            }
            if (c.shape(1) != 2) {
              throw std::runtime_error("coords input shape must be (n, 2)");
            }
            timer::clock clock{};

            for (auto i = 0; i < p.shape(0); i++) {

              PIP::polygon2 poly;
              std::vector<PIP::point2> points;
              size_t start = p(i, 0);
              for (size_t j = 0; j < p(i, 1); ++j) {
                points.push_back(PIP::point2(c(start + j, 0), c(start + j, 1)));
              }
              bg::assign_points(poly, points);
              self->addPolygon((uint32_t)i, poly);
            }
            const auto t1{clock.reset()};

            self->buildTree();
            const auto t2{clock.elapsed()};

            self->stats["construct_polygons_ns"] = t1;
            self->stats["polygon_count"] = p.shape(0);
            self->stats["coord_count"] = c.shape(0);
            self->stats["build_rtree_ns"] = t2;
            return self;
          }))
      .def(
          "test",
          +[](PIP::ZonalKey<uint32_t> &self, py::array_t<_Float32> coords) {
            auto c = coords.unchecked<2>();
            if (c.shape(1) < 2) {
              throw std::runtime_error(
                  "coords input shape must be at least (n, 2)");
            }
            std::vector<std::set<uint32_t>> res;

            timer::clock clock{};

            for (auto i = 0; i < c.shape(0); ++i) {
              auto t = self.test(PIP::point2(c(i, 0), c(i, 1)));
              clock.pause();
              std::set<uint32_t> set(t.begin(), t.end());
              res.push_back(set);
              clock.resume();
            }
            const auto t1{clock.elapsed()};
            self.stats["test_points_count"] = c.shape(0);
            self.stats["test_points_ns"] = t1;

            return res;
          })
      .def(
          "stats", +[](PIP::ZonalKey<uint32_t> &self) { return self.stats; })

      ;
}