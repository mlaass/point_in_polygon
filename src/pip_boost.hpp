#ifndef PIP_BOOST_INCLUDE
#define PIP_BOOST_INCLUDE
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/astar_search.hpp>
// #include <boost/graph/dijkstra_shortest_paths.hpp>
// #include <boost/graph/graph_utility.hpp>
// #include <boost/graph/random.hpp>
// #include <boost/graph/reverse_graph.hpp>

// using namespace boost;

// typedef float cost;
// typedef adjacency_list<vecS, vecS, bidirectionalS>
//     graph_t; // property<vertex_color_t,default_color_type>,
//              // property<edge_weight_t, cost, property<edge_index_t,size_t>>

// // typedef property_map<graph_t, edge_weight_t>::type WeightMap;
// // typedef property_map<graph_t, vertex_color_t>::type ColorMap;
// // typedef color_traits<property_traits<ColorMap>::value_type> Color;
// // typedef property_map<graph_t, edge_index_t>::type IndexMap;
// typedef graph_t::vertex_descriptor vertex_descriptor;
// typedef graph_t::edge_descriptor edge_descriptor;
// typedef graph_t::vertex_iterator vertex_iterator;

#include "pip_points.hpp"

// Boost.Range
#include <boost/range.hpp>
// adaptors
#include <boost/iterator/function_output_iterator.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/transformed.hpp>

namespace PIP {

// a functional turning points into values

struct value_maker3 {
  template <typename T> inline value3 operator()(T const &v) const {
    box3 b(v.value(), v.value()); // make point to box
    return value3(b, v.index());
  }
};
struct value_maker2 {
  template <typename T> inline value2 operator()(T const &v) const {
    box2 b(v.value(), v.value()); // make point to box
    return value2(b, v.index());
  }
};

std::ostream &operator<<(std::ostream &os, point3 const &p) {
  return os << "(" << bg::get<0>(p) << ";" << bg::get<1>(p) << ";"
            << bg::get<2>(p) << ")";
}
std::ostream &operator<<(std::ostream &os, point2 const &p) {
  return os << "(" << bg::get<0>(p) << ";" << bg::get<1>(p) << ")";
}
template <typename KEY = std::string> class PolyBoost {
  typedef std::pair<box2, KEY> value;
  typedef bgi::rtree<value, bgi::rstar<16, 4>> rtree;

  std::map<KEY, polygon2> zones;
  rtree rt;
  std::vector<value> values;

public:
  std::map<std::string, int64_t> stats;
  void addPolygon(KEY key, polygon2 poly) {
    bg::correct(poly);
    zones[key] = poly;
    box2 b;
    bg::envelope(poly, b);
#ifdef DEBUG_ZONAL
    std::cout << "adding a polygon" << std::endl;
    std::cout << "polygon: " << bg::wkt(poly) << std::endl;
    std::cout << "BBox:" << bg::wkt(b) << std::endl;
    std::cout << "Key: " << key << std::endl;
#endif
    values.push_back(std::make_pair(b, key));
  }

  void buildTree() {
    // build an R tree for this dataset
    rt = rtree(values.begin(), values.end());
  }

  void readWKT(std::string filename) {
    values.clear();
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
    rt = rtree(values.begin(), values.end());
  }

  // get spatial key
  std::vector<KEY> test(point2 p) {
#ifdef DEBUG_ZONAL
    std::cout << "Intersecting point " << p << std::endl;
#endif
    std::vector<KEY> out;
    std::vector<value> result;
    rt.query(bgi::intersects(p),
             boost::make_function_output_iterator([&](value const &val) {
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

template <typename KEY = std::string> class MultiPolyRTree {
  std::multimap<KEY, polygon2> zones;
  typedef std::pair<box2, KEY> value;
  typedef bgi::rtree<value, bgi::rstar<16, 4>> rtree;
  rtree rt;

public:
  void readWKT(std::string filename) {
    std::vector<value> values;
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
    rt = rtree(values.begin(), values.end());
  }
  // get spatial key
  std::vector<KEY> test(point2 p) {
#ifdef DEBUG_ZONAL
    std::cout << "Intersecting point " << p << std::endl;
#endif
    std::vector<std::string> out;
    std::vector<value> result;
    rt.query(bgi::intersects(p),
             boost::make_function_output_iterator([&](value const &val) {
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
             }));
    return out;
  }
};

}; // namespace PIP
#endif
