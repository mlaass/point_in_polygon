
#ifndef PIP_SIMPLE_INCLUDE
#define PIP_SIMPLE_INCLUDE

// Boost.Range
#include <boost/range.hpp>
// adaptors
#include <boost/iterator/function_output_iterator.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include "pip_points.hpp"

namespace PIP {
typedef std::pair<point2, point2> edge;
typedef std::vector<edge> simple_polygon;

// https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html

inline int isLeft(const point2 &P0, const point2 &P1, const point2 &P2) {
  return ((P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y));
}

template <typename KEY = std::string> struct BoxList {
  std::vector<box2> boxes;
  std::vector<simple_polygon> polygons;
  std::map<size_t, KEY> keys;

  static bool ptest_crossing(const point2 &p, const simple_polygon &poly) {
    int cn = 0; // the  crossing number counter

    // loop through all edges of the polygon
    for (auto &e : poly) { // edge from e.first  to e.first
      if (((e.first.y <= p.y) && (e.second.y > p.y))     // an upward crossing
          || ((e.first.y > p.y) && e.second.y <= p.y)) { // a downward crossing
        // compute  the actual edge-ray intersect x-coordinate
        float vt = (float)(p.y - e.first.y) / (e.second.y - e.first.y);
        if (p.x < e.first.x + vt * (e.second.x - e.first.x)) // p.x < intersect
          ++cn; // a valid crossing of y=p.y right of p.x
      }
    }
    return (cn & 1); // 0 if even (out), and 1 if  odd (in)
  }
  static bool ptest_crossing_para(const point2 &p, const simple_polygon &poly) {
    int cn = 0;
#pragma omp parallel for
    for (auto i = 0; i < poly.size(); ++i) {
      if (((poly[i].first.y <= p.y) && (poly[i].second.y > p.y)) ||
          ((poly[i].first.y > p.y) && poly[i].second.y <= p.y)) {

        float vt = (float)(p.y - poly[i].first.y) /
                   (poly[i].second.y - poly[i].first.y);
        if (p.x < poly[i].first.x + vt * (poly[i].second.x - poly[i].first.x)) {
#pragma omp critical
          ++cn;
        }
      }
    }
    return (cn & 1); // 0 if even (out), and 1 if  odd (in)
  }
  static bool ptest_winding(const point2 &p, const simple_polygon &poly) {
    int wn = 0; // the  winding number counter

    // loop through all edges of the polygon
    for (auto &e : poly) {    // edge from e.first to  e.second
      if (e.first.y <= p.y) { // start y <= p.y
        if (e.second.y > p.y) // an upward crossing
          if (isLeft(e.first, e.second, p) > 0) // p left of  edge
            ++wn;                               // have  a valid up intersect
      } else {                 // start y > p.y (no test needed)
        if (e.second.y <= p.y) // a downward crossing
          if (isLeft(e.first, e.second, p) < 0) // p right of  edge
            --wn;                               // have  a valid down intersect
      }
    }
    return wn;
  }

  std::map<std::string, int64_t> stats;

  void addPolygon(const KEY &key, const std::vector<point2> &p) {
    point2 mn{FLT_MAX, FLT_MAX}, mx{FLT_MIN, FLT_MIN};
    simple_polygon poly;
    for (size_t i = 0; i < p.size(); ++i) {
      mn.x = std::min(mn.x, p[i].x);
      mn.y = std::min(mn.y, p[i].y);
      mx.x = std::max(mx.x, p[i].x);
      mx.y = std::max(mx.y, p[i].y);
      if (i != (p.size() - 1)) {
        poly.push_back(std::make_pair(p[i], p[i + 1]));
      } else {
        poly.push_back(std::make_pair(p[i], p[0]));
      }
    }
    keys[polygons.size()] = key;
    polygons.push_back(poly);
    boxes.push_back(box2(mn, mx));
  }
  std::vector<KEY>
  test_fn(const point2 &p,
          std::function<bool(const point2 &, const simple_polygon &)> fn) {
    std::vector<KEY> res;
    for (size_t i = 0; i < polygons.size(); i++) {
      if (bg::within(p, boxes[i])) { // check box first

        if (fn(p, polygons[i])) { // check polygon
          res.push_back(keys[i]);
        }
      }
    }
    return res;
  }
  std::vector<KEY>
  test_fn_para(const point2 &p,
               std::function<bool(const point2 &, const simple_polygon &)> fn) {
    std::vector<KEY> res;
#pragma omp parallel for
    for (size_t i = 0; i < polygons.size(); i++) {
      if (bg::within(p, boxes[i])) { // check box first

        if (fn(p, polygons[i])) { // check polygon
#pragma omp critical
          res.push_back(keys[i]);
        }
      }
    }
    return res;
  }
  std::vector<KEY> test_crossing(const point2 &p) {
    return test_fn(p, ptest_crossing);
  }
  std::vector<KEY> test_crossing_para(const point2 &p) {
    return test_fn_para(p, ptest_crossing);
  }
  std::vector<KEY> test_winding(const point2 &p) {
    return test_fn(p, ptest_winding);
  }
};

} // namespace PIP

#endif