#ifndef PIP_INCLUDE
#define PIP_INCLUDE
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

namespace PIP {

typedef flat_point3 point3;
typedef flat_point2 point2;

typedef bg::model::multi_point<point3> multipoint;
typedef bg::model::box<point3> box3;
typedef bg::model::box<point2> box2;
typedef bg::model::polygon<point3> polygon;               // ccw, open polygon
typedef bg::model::polygon<point2> polygon2;              // ccw, open polygon
typedef bg::model::multi_polygon<polygon> multipolygon;   // ccw, open polygon
typedef bg::model::multi_polygon<polygon2> multipolygon2; // ccw, open polygon
typedef bg::model::linestring<point3> linestring;
typedef std::pair<box3, uint32_t> value3;
typedef std::pair<box2, uint32_t> value2;

typedef bgi::rtree<value3, bgi::rstar<16, 4>> rtree3;
typedef bgi::rtree<value2, bgi::rstar<16, 4>> rtree2;

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
template <typename KEY = std::string> class PolyRTree {
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
