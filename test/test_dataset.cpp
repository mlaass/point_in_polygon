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
    for (auto k : tq::range(3, 8)) {
      std::cout << "extract knn for k = " << k << std::endl;
      std::vector<std::vector<double>> features;
      std::vector<std::vector<double>> neighbors;
      pc.extractKnnTensorsAndNeighbors(k, features, neighbors);
      std::cout << "features = (" << features.size() << ","
                << features[0].size() << ")"
                << " neighbors: (" << neighbors.size() << ","
                << neighbors[0].size() << ")" << std::endl;
    }
  }

  std::cout << "ok" << std::endl;
  return 0;
}
