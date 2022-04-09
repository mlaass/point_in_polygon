#ifndef PIP_POINTS_INCLUDE
#define PIP_POINTS_INCLUDE

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
#include <boost/iterator/function_output_iterator.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/transformed.hpp>

struct flat_point3 {
  float x, y, z;
  flat_point3() {}
  flat_point3(double _x, double _y, double _z) : x(_x), y(_y), z(_z){};
  flat_point3(float _x, float _y, float _z) : x(_x), y(_y), z(_z){};
  flat_point3(double _x, double _y) : x(_x), y(_y), z(0){};
  flat_point3(float _x, float _y) : x(_x), y(_y), z(0){};
  template <typename Q> flat_point3(Q &p) {
    x = p.x;
    y = p.y;
    z = p.z;
  }
};

struct flat_point2 {
  float x, y, z;
  flat_point2() {}
  flat_point2(double _x, double _y) : x(_x), y(_y){};
  flat_point2(float _x, float _y) : x(_x), y(_y){};
  template <typename Q> flat_point2(Q &p) {
    x = p.x;
    y = p.y;
  }
};

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

BOOST_GEOMETRY_REGISTER_POINT_3D(flat_point3, float,
                                 boost::geometry::cs::cartesian, x, y, z);
BOOST_GEOMETRY_REGISTER_POINT_2D(flat_point2, float,
                                 boost::geometry::cs::cartesian, x, y);

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
} // namespace PIP

#endif