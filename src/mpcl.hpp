#ifndef MPCL_INCLUDE
#define MPCL_INCLUDE
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/random.hpp>
#include <boost/graph/reverse_graph.hpp>

using namespace boost;

typedef float cost;
typedef adjacency_list<vecS, vecS, bidirectionalS>
    graph_t; // property<vertex_color_t,default_color_type>,
             // property<edge_weight_t, cost, property<edge_index_t,size_t>>

// typedef property_map<graph_t, edge_weight_t>::type WeightMap;
// typedef property_map<graph_t, vertex_color_t>::type ColorMap;
// typedef color_traits<property_traits<ColorMap>::value_type> Color;
// typedef property_map<graph_t, edge_index_t>::type IndexMap;
typedef graph_t::vertex_descriptor vertex_descriptor;
typedef graph_t::edge_descriptor edge_descriptor;
typedef graph_t::vertex_iterator vertex_iterator;

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/geometry/index/rtree.hpp>

// Boost.Range
#include <boost/range.hpp>
// adaptors
#include <boost/function_output_iterator.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/transformed.hpp>

// related to X11 #define clashing with Eigen's enum value Success
#ifdef Success
#undef Success
#endif
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using Eigen::MatrixXd;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

struct flat_point3 {
  float x, y, z;
  flat_point3() {}
  flat_point3(double _x, double _y, double _z) : x(_x), y(_y), z(_z){};
  flat_point3(double _x, double _y) : x(_x), y(_y), z(0){};
  template <typename Q> flat_point3(Q &p) {
    x = p.x;
    y = p.y;
    z = p.z;
  }
};

/*struct flat_point2{
float x,y;
flat_point2(){}
flat_point2(double _x,double  _y):x(_x),y(_y){};
};*/
struct flat_point2 {
  float x, y, z;
  flat_point2() {}
  flat_point2(double _x, double _y) : x(_x), y(_y){};
  template <typename Q> flat_point2(Q &p) {
    x = p.x;
    y = p.y;
  }
};

BOOST_GEOMETRY_REGISTER_POINT_3D(flat_point3, float, bg::cs::cartesian, x, y,
                                 z);
BOOST_GEOMETRY_REGISTER_POINT_2D(flat_point2, float, bg::cs::cartesian, x, y);
// typedef bg::model::point<double, 3, bg::cs::cartesian>  point;
// //boost::geometry::model::d2::point_xy<double>
typedef flat_point3 point3;
typedef flat_point2 point2;
typedef point3 point;

typedef bg::model::multi_point<point3>
    multipoint; // boost::geometry::model::d2::point_xy<double>
typedef bg::model::box<point3> box3;
typedef bg::model::box<point2> box2;
typedef bg::model::polygon<point3> polygon;               // ccw, open polygon
typedef bg::model::polygon<point2> polygon2;              // ccw, open polygon
typedef bg::model::multi_polygon<polygon> multipolygon;   // ccw, open polygon
typedef bg::model::multi_polygon<polygon2> multipolygon2; // ccw, open polygon
typedef bg::model::linestring<point3> linestring;
typedef std::pair<box3, uint32_t> value3;

typedef bgi::rtree<value3, bgi::rstar<16, 4>> rtree3;

// a functional turning points into values

struct value_maker3 {
  template <typename T> inline value3 operator()(T const &v) const {
    box3 b(v.value(), v.value()); // make point to box
    return value3(b, v.index());
  }
};
/*struct value_maker2
{
    template<typename T>
    inline value2 operator()(T const& v) const
    {
        box2 b(v.value(), v.value()); // make point to box
        return value2(b, v.index());
    }
};*/

std::ostream &operator<<(std::ostream &os, point3 const &p) {
  return os << "(" << bg::get<0>(p) << ";" << bg::get<1>(p) << ";"
            << bg::get<2>(p) << ")";
}
std::ostream &operator<<(std::ostream &os, point2 const &p) {
  return os << "(" << bg::get<0>(p) << ";" << bg::get<1>(p) << ")";
}

namespace mpcl {
namespace util {

class ZonalKey {
  std::map<std::string, polygon2> zones;
  typedef std::pair<box2, std::string> zk_value;
  typedef bgi::rtree<zk_value, bgi::rstar<16, 4>> zk_rtree;
  zk_rtree rt;

public:
  ZonalKey(std::string filename) {
    std::vector<zk_value> values;
    // a zone definition file contains
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line)) {
      auto pos = line.find(" ");
      if (pos == std::string::npos)
        continue;
      std::string key = line.substr(0, pos);
      std::string wkt = line.substr(pos + 1);
      polygon2 p;
      bg::read_wkt(wkt, p);

      bg::correct(p);
      zones[key] = p;
      box2 b;
      bg::envelope(p, b);
#ifdef DEBUG_ZONAL
      std::cout << "adding a polygon" << std::endl;
      std::cout << "WKT: " << wkt << std::endl;
      std::cout << "Parsed: " << bg::wkt(p) << std::endl;
      std::cout << "BBox:" << bg::wkt(b) << std::endl;
      std::cout << "Key: " << key << std::endl;
#endif
      values.push_back(std::make_pair(b, key));
    }
    std::cout << "Using " << values.size() << " polygons. " << std::endl;
    // build an R tree for this dataset
    rt = zk_rtree(values.begin(), values.end());
  }
  // get spatial key
  std::vector<std::string> operator()(point2 p) {
#ifdef DEBUG_ZONAL
    std::cout << "Intersecting point " << p << std::endl;
#endif
    std::vector<std::string> out;
    std::vector<zk_value> result;
    rt.query(bgi::intersects(p),
             boost::make_function_output_iterator([&](zk_value const &val) {
               if (bg::within(p, zones[val.second])) {
#ifdef DEBUG_ZONAL
                 std::cout << "Found it to be part of " << val.second
                           << std::endl;
#endif

                 out.push_back(val.second);
               }
             }));
    return out;
  }
};

class ZonalKeyMulti {
  std::multimap<std::string, polygon2> zones;
  typedef std::pair<box2, std::string> zk_value;
  typedef bgi::rtree<zk_value, bgi::rstar<16, 4>> zk_rtree;
  zk_rtree rt;

public:
  ZonalKeyMulti(std::string filename) {
    std::vector<zk_value> values;
    // a zone definition file contains
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line)) {
      auto pos = line.find(" ");
      if (pos == std::string::npos)
        continue;
      std::string key = line.substr(0, pos);
      std::string wkt = line.substr(pos + 1);
      polygon2 rp;
      multipolygon2 mp;
      try {
        bg::read_wkt(wkt, rp);
        mp.push_back(rp);
      } catch (...) {
        bg::read_wkt(wkt, mp);
      }
      for (auto &p : mp) {
        bg::correct(p);
        //	       zones[key] = p;
        zones.insert(std::make_pair(key, p));
        box2 b;
        bg::envelope(p, b);
#ifdef DEBUG_ZONAL
        std::cout << "adding a polygon" << std::endl;
        std::cout << "WKT: " << wkt << std::endl;
        std::cout << "Parsed: " << bg::wkt(p) << std::endl;
        std::cout << "BBox:" << bg::wkt(b) << std::endl;
        std::cout << "Key: " << key << std::endl;
#endif
        values.push_back(std::make_pair(b, key));
      }
    }
    std::cout << "Using " << values.size() << " polygons. " << std::endl;
    // build an R tree for this dataset
    rt = zk_rtree(values.begin(), values.end());
  }
  // get spatial key
  std::vector<std::string> operator()(point2 p) {
#ifdef DEBUG_ZONAL
    std::cout << "Intersecting point " << p << std::endl;
#endif
    std::vector<std::string> out;
    std::vector<zk_value> result;
    rt.query(bgi::intersects(p),
             boost::make_function_output_iterator([&](zk_value const &val) {
               auto r = zones.equal_range(val.second);
               polygon2 poly;
               std::string key;
               for (; r.first != r.second; r.first++) {

                 if (bg::within(p, r.first->second)) {
#ifdef DEBUG_ZONAL
                   std::cout << "Found it to be part of " << r->first
                             << std::endl;
#endif

                   out.push_back(r.first->first);
                 }
               }

               /*	   if (bg::within(p,zones[val.second])){
               #ifdef DEBUG_ZONAL
                          std::cout << "Found it to be part of "<< val.second <<
               std::endl; #endif

                           out.push_back(val.second);
                           }*/
             }));
    return out;
  }
};

} // namespace util

struct tensor_features {
  static const std::vector<std::string> names;
  template <typename vec3> static std::vector<double> calc(const vec3 &l) {
    return {
        (l[0] - l[1]) / l[0],     // linearity
        (l[1] - l[2]) / l[0],     // planarity
        l[2] / l[0],              // scattering
        cbrt(l[0] * l[1] * l[2]), // omnivariance
        (l[0] - l[2]) / l[0],     // anisotropy
        -l[0] * log2(l[0]) - l[1] * log2(l[1]) -
            l[2] * log2(l[2]),       // eigenentropy
        (l[0] + l[1] + l[2]) / l[0], // trace_by_l0
        l[2] / (l[0] + l[1] + l[2])  // chg_of_curv
    };
  };
};
const std::vector<std::string> tensor_features::names = {
    "linearity",  "planarity",    "scattering",  "omnivariance",
    "anisotropy", "eigenentropy", "trace_by_l0", "chg_of_curv"};

class pointcloud {
public:
  std::vector<point> coords;
  graph_t g;
  rtree3 rt;

  size_t size() { return coords.size(); };

  //   bool loadh5(std::string filename, bool index = true) {
  // #ifdef HAVE_HDF5
  //     hsize_t dims[2]; /* dataset and chunk dimensions*/
  //     hsize_t chunk_dims[2];
  //     hsize_t col_dims[1];
  //     hsize_t count[2];
  //     hsize_t offset[2];
  //     int status;
  //     auto file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  //     auto coords_ds = H5Dopen2(file_id, "/coords", H5P_DEFAULT);
  //     auto filespace = H5Dget_space(coords_ds); /* Get filespace handle
  //     first. */ auto rank = H5Sget_simple_extent_ndims(filespace); status =
  //     H5Sget_simple_extent_dims(filespace, dims, NULL); if (status < 0)
  //       throw(std::runtime_error("get_simple_extent failed"));
  //     printf("dataset rank %d, dimensions %lu x %lu\n", rank,
  //     (uint64_t)(dims[0]),
  //            (uint64_t)(dims[1]));

  //     coords.clear();
  //     coords.reserve(dims[0]);

  //     // we only support chunked datasets
  //     auto cparms =
  //         H5Dget_create_plist(coords_ds); /* Get properties handle first. */
  //     if (H5D_CHUNKED != H5Pget_layout(cparms))
  //       throw(std::runtime_error("we expect a chunked dataset"));
  //     auto rank_chunk = H5Pget_chunk(cparms, 2, chunk_dims);
  //     printf("chunk rank %d, dimensions %lu x %lu\n", rank_chunk,
  //            (uint64_t)(chunk_dims[0]), (uint64_t)(chunk_dims[1]));

  //     chunk_dims[0] *= 4096; // read 1024 chunk at a time.

  //     auto memspace = H5Screate_simple(rank_chunk, chunk_dims, NULL);

  //     offset[0] = 0;
  //     offset[1] = 0;
  //     count[0] = chunk_dims[0];
  //     count[1] = chunk_dims[1];

  //     // loop through the file
  //     float *chunk_out = new float[chunk_dims[0] * chunk_dims[1]];
  //     for (size_t offset0 = 0; offset0 < dims[0]; offset0 += chunk_dims[0]) {
  //       offset[0] = offset0;
  //       if (offset0 + count[0] > dims[0]) {
  //         count[0] = dims[0] - offset0;
  //         //	      std::cout << "Last slab heigh: " << count[0] << std::endl;
  //         status = H5Sclose(memspace);
  //         memspace = H5Screate_simple(rank_chunk, count, NULL);
  //       }
  //       //	    std::cout << "Reading slab from offset[0]=" << offset[0] <<
  //       // std::endl;

  //       status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
  //                                    count, NULL);

  //       float *p = chunk_out;
  //       //	    std::cout << "Read from " << offset[0] <<  " for " <<
  //       // count[0] << " until " << offset[0] + count[0] << std::endl;
  //       status = H5Dread(coords_ds, H5T_NATIVE_FLOAT, memspace, filespace,
  //                        H5P_DEFAULT, chunk_out);

  //       for (size_t j = 0; j < count[0]; j++) {
  //         auto x = *p++;
  //         auto y = *p++;
  //         float z = 0;
  //         if (dims[1] == 3)
  //           z = *p++;

  //         coords.push_back(point(x, y, z));
  //         // classes.push_back(0);
  //       }
  //     } // for

  //     delete chunk_out;

  //     status = H5Sclose(memspace);
  //     status = H5Dclose(coords_ds);
  //     status = H5Fclose(file_id);

  //     std::cout << "Have " << coords.size() << " points" << std::endl;
  //     // Compute center
  //     multipoint &mp = (multipoint &)coords;
  //     bg::centroid(mp, center);
  //     std::cout << "Center: " << bg::get<0>(center) << ", " <<
  //     bg::get<1>(center)
  //               << ", " << bg::get<2>(center) << std::endl;

  //     return true;

  // #else
  //     throw(std::runtime_error(
  //         "HDF5 support has not been compiled into this executable."));
  // #endif
  //   }

  //   bool saveh5_coords(std::string filename, size_t dim = 3) {
  //     hid_t file_id, dset_id, space_id, dcpl_id;
  //     hsize_t chunk_dims[2] = {1024, dim};

  //     hsize_t dset_dims[2] = {coords.size(), dim};
  //     //	int     buffer[12][12];
  //     /* Create the file */
  //     file_id =
  //         H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
  //         H5P_DEFAULT);
  //     dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  //     H5Pset_chunk(dcpl_id, 2, chunk_dims);

  //     /* Create the dataspace and the chunked dataset */
  //     space_id = H5Screate_simple(2, dset_dims, NULL);
  //     dset_id = H5Dcreate(file_id, "/coords", H5T_NATIVE_FLOAT, space_id,
  //                         H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  //     if (dim == 2)
  //       throw(std::runtime_error("unable to write 2d point clouds now"));
  //     /* Write to the dataset */
  //     H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
  //              &coords[0]);

  //     /* Close */
  //     H5Dclose(dset_id);
  //     H5Sclose(space_id);
  //     H5Pclose(dcpl_id);
  //     H5Fclose(file_id);
  //     return 0;
  //   }

  // bool load(std::string filename) {
  //   std::cout << "Loading " << filename << std::endl;
  //   if (boost::algorithm::ends_with(filename, "h5"))
  //     return loadh5(filename);
  //   std::ifstream ifs;
  //   ifs.open(filename);
  //   if (!ifs.good())
  //     return false;
  //   std::string line;
  //   while (std::getline(ifs, line)) {
  //     //	    std::cout << line << std::endl;
  //     line.erase(line.begin(),
  //                std::find_if(line.begin(), line.end(),
  //                             std::bind1st(std::not_equal_to<char>(), ' ')));
  //     if (line[0] == '#')
  //       continue;
  //     // tokenize
  //     std::stringstream strstr(line);
  //     std::istream_iterator<std::string> it(strstr);
  //     std::istream_iterator<std::string> end;
  //     std::vector<std::string> tokens(it, end);

  //     if (tokens.size() != 5)
  //       throw(std::runtime_error("need five tokens on each non-comment
  //       line"));

  //     double x = std::stod(tokens[0]);
  //     double y = std::stod(tokens[1]);
  //     double z = std::stod(tokens[2]);
  //     size_t cl = std::stoi(tokens[3]);
  //     double certain = std::stod(tokens[4]);
  //     coords.push_back(point(x, y, z));
  //     classes.push_back(cl);
  //   }
  //   std::cout << "Have " << coords.size() << " points" << std::endl;
  //   // Compute center
  //   multipoint &mp = (multipoint &)coords;
  //   bg::centroid(mp, center);
  //   std::cout << "Center: " << bg::get<0>(center) << ", " <<
  //   bg::get<1>(center)
  //             << ", " << bg::get<2>(center) << std::endl;

  //   return true;
  // }

  void buildIndex() {
    rt = rtree3(coords | boost::adaptors::indexed() |
                boost::adaptors::transformed(value_maker3()));

    //	rt = rtree(coords.begin(), coords.end());
  }

  size_t buildGraph(size_t k = 7) {
    // will get parameters
    g = graph_t(coords.size()); // create a fresh empty graph
#pragma omp parallel for
    for (size_t i = 0; i < coords.size(); i++) {
      const auto &p = coords[i];
      rt.query(bgi::nearest(p, k),
               boost::make_function_output_iterator([&](value3 const &v) {
                 //	        #neighbors.push_back(v.first.min_corner());
                 if (i == v.second) {
                   return;
                 }
#pragma omp critical
                 add_edge(i, v.second, g); // add an edge from i to second
               }));
    }
    return num_edges(g);
  }

  void createVertexIndexArray(std::vector<uint32_t> &indices) {
    auto c = num_edges(g);
    if (2 * c > indices.size())
      throw(std::runtime_error(
          "vertex index array buffer smaller than max elements"));

    graph_traits<graph_t>::edge_iterator ei, ei_end;
    uint32_t *p = &indices[0];
    for (tie(ei, ei_end) = edges(g); ei != ei_end; ++ei) {
      // modification for mutual kNN
      auto s = source(*ei, g);
      auto e = target(*ei, g);
      if (edge(e, s, g).second) { // mutual kNN
        *p = s;
        p++;
        *p = e;
        p++;
      }
    }
  }

  void applyKnnDistanceToZ(size_t k = 6) {
#pragma omp parallel for
    for (size_t i = 0; i < coords.size(); i++) {
      const auto &p = coords[i];
      double d = 0;
      rt.query(bgi::nearest(p, k),
               boost::make_function_output_iterator([&](value3 const &v) {
                 //	        #neighbors.push_back(v.first.min_corner());
                 if (i == v.second) {
                   return;
                 }
                 double _d = bg::distance(coords[i], coords[v.second]);
#pragma omp critical
                 if (_d > d) {
                   d = _d;
                 }
               }));
      // coords[i].z = -d;
      bg::set<2>(coords[i], -d);
    }
  }

  // TODO extract knn and calculate tensors seperately
  // knn.shape = (n,3,k) or (n, k, 3)?
  // TODO encode knn into 3d globimap
  void knn(size_t k, std::function<void(const multipoint &)> fn) {

    // our first basic extractor: for each point, extract kNN

    // TODO tqdm here
    for (size_t i = 0; i < coords.size(); i++) {
      // if (i % 1000 == 0)
      //   std::cout << i << "/" << coords.size() << std::endl;
      const point &p = coords[i];
      //	  std::cout << "Extract " << p << std::endl;
      multipoint neighbors;
      neighbors.push_back(p);
      rt.query(
          bgi::nearest(p, k),
          boost::make_function_output_iterator([&neighbors](value3 const &v) {
            neighbors.push_back(v.first.min_corner());
          }));
      fn(neighbors);
    }
  }
  static std::vector<double> neighborsToTF(size_t k,
                                           const multipoint &neighbors) {
    point neighbors_centroid;
    bg::centroid(neighbors, neighbors_centroid);

    // for (auto &p: neighbors)
    //   std::cout << "Neighbor: " << p<<std::endl;
    // std::cout << "Center: " << neighbors_centroid << std::endl;

    MatrixXd mp(k, 3);
    for (size_t i = 0; i < k; i++) // assert coords.size() > k
    {
      mp(i, 0) = bg::get<0>(neighbors[i]) - bg::get<0>(neighbors_centroid);
      mp(i, 1) = bg::get<1>(neighbors[i]) - bg::get<1>(neighbors_centroid);
      mp(i, 2) = bg::get<2>(neighbors[i]) - bg::get<2>(neighbors_centroid);
    }
    // std::cout << mp << std::endl;
    mp = mp.transpose() * mp;
    // std::cout << "--" << std::endl;
    // std::cout << mp << std::endl;
    // std::cout << "--" << std::endl;

    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(mp, false);
    auto ev = solver.eigenvalues().reverse();
    ev = ev / ev.sum();
    return tensor_features::calc(ev);
  }
  void extractKnnTensors(size_t k, std::vector<std::vector<double>> &features) {
    this->knn(k, [&](const multipoint &neighbors) {
      features.push_back(pointcloud::neighborsToTF(k, neighbors));
    });
  }
  void extractKnnTensorsAndNeighbors(
      size_t k, std::vector<std::vector<double>> &features,
      std::vector<std::vector<double>> &neighbors_out) {
    this->knn(k, [&](const multipoint &neighbors) {
      features.push_back(pointcloud::neighborsToTF(k, neighbors));
      std::vector<double> ng;
      bg::for_each_point(neighbors, [&](point const &p) {
        ng.push_back(bg::get<0>(p));
        ng.push_back(bg::get<1>(p));
        ng.push_back(bg::get<2>(p));
      });
      neighbors_out.push_back(ng);
    });
  }
};

}; // namespace mpcl
#endif
