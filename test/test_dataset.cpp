#include "mpcl.hpp"
#include <highfive/H5File.hpp>
#include <iostream>
#include <tqdm.hpp>

namespace h5 = HighFive;

const std::string filename = "../test/modelnet10_sample.h5";

int main() {

  auto file = h5::File(filename, h5::File::ReadWrite);

  for (auto g : file.listObjectNames()) {
    std::cout << "read group: " << g << std::endl;
    // we get the dataset
    mpcl::pointcloud pc;
    std::vector<std::vector<double>> read_data;
    h5::DataSet coords = file.getGroup(g).getDataSet("coords");

    auto shape = coords.getDimensions();
    std::cout << "shape = (" << shape[0] << ", " << shape[1] << ")"
              << std::endl;

    std::vector<std::vector<double>> result;
    coords.select({0, 0}, {shape[0], 3}).read(result);
    for (auto p : result) {
      pc.coords.push_back(point(p[0], p[1], p[2]));
    }
    pc.buildIndex();
    size_t k = 6;
    std::cout << "extract knn for k = " << k << std::endl;
    std::vector<double> features;
    std::vector<double> neighbors;
    pc.extractKnnTensorsAndNeighbors(k, features, neighbors);
    size_t row_size = (k + 1) * 3;
    std::cout << k << " (" << features.size() / 8 << ","
              << features.size() / shape[0] << ") ("
              << neighbors.size() / row_size << "," << row_size << ")"
              << std::endl;
    std::cout << "[ ";
    for (auto i = 0; i < row_size; i++) {

      std::cout << neighbors[i] << " ";
    }
    std::cout << " ] " << std::endl;
  }

  std::cout << "ok" << std::endl;
  return 0;
}
