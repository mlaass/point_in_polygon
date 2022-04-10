
#ifndef PIP_OPENCL_INCLUDE
#define PIP_OPENCL_INCLUDE

#define __CL_ENABLE_EXCEPTIONS

#include "cl_common/cl.hpp"

#include "cl_common/util.hpp" // utility library

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include "cl_common/err_code.h"

#include "pip_points.hpp"

namespace PIP {
template <typename KEY = std::string> struct OpenCLImpl {

  std::vector<cl_uint2> polys;
  std::vector<cl_float4> edges;
  std::vector<cl_float4> boxes;
  std::map<size_t, KEY> keys;
  std::map<std::string, int64_t> stats;
  cl::Program program;
  cl::Context context;
  cl::CommandQueue queue;
  OpenCLImpl() : context(CL_DEVICE_TYPE_DEFAULT) {

    try {
      // Create a context
      ;
      std::cout << "load program.." << std::endl;
      // Load in kernel source, creating a program object for the context
      program =
          cl::Program(context, util::loadProgram("src/pip_opencl.cl"), true);
      // Get the command queue
      queue = cl::CommandQueue(context);
    } catch (cl::Error err) {
      std::cout << "Exception\n";
      std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
                << std::endl;
    }
  }

  void addPolygon(const KEY &key, const std::vector<point2> &p) {
    point2 mn{FLT_MAX, FLT_MAX}, mx{FLT_MIN, FLT_MIN};
    keys[polys.size()] = key;
    polys.push_back({edges.size(), p.size()});
    for (size_t i = 0; i < p.size(); ++i) {
      mn.x = std::min(mn.x, p[i].x);
      mn.y = std::min(mn.y, p[i].y);
      mx.x = std::max(mx.x, p[i].x);
      mx.y = std::max(mx.y, p[i].y);
      if (i != (p.size() - 1)) {
        edges.push_back({p[i].x, p[i].y, p[i + 1].x, p[i + 1].y});
      } else {
        edges.push_back({p[i].x, p[i].y, p[0].x, p[0].y});
      }
    }

    boxes.push_back({mn.x, mn.y, mx.x, mx.y});
  }

  std::vector<std::tuple<uint32_t, std::set<KEY>>> test_naive(float *points,
                                                              size_t n_points) {
    std::vector<std::tuple<uint32_t, std::set<KEY>>> res;
    std::vector<int> result(n_points, -1);

    cl::Buffer d_points;
    cl::Buffer d_polys;
    cl::Buffer d_boxes;
    cl::Buffer d_edges;
    cl::Buffer d_result;
    try {

      // Create the kernel functor
      cl::make_kernel<uint, uint, cl::Buffer, cl::Buffer, cl::Buffer,
                      cl::Buffer, cl::Buffer>
          pip_naive(program, "pip_naive");
      std::cout << "create buffers.. ";
      d_polys = cl::Buffer(context, polys.begin(), polys.end(), true);
      std::cout << "1.. ";
      d_edges = cl::Buffer(context, edges.begin(), edges.end(), true);
      d_boxes = cl::Buffer(context, boxes.begin(), boxes.end(), true);
      std::cout << "2.. ";
      d_result = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            result.size() * sizeof(int), &result[0]);
      std::cout << "3.. " << result[0];
      d_points = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            n_points * sizeof(float) * 2, points);
      std::cout << " done.." << std::endl;

      std::cout << "enqueue args.." << std::endl;
      auto args = cl::EnqueueArgs(queue, cl::NDRange(n_points, polys.size()));
      std::cout << "run.." << std::endl;
      pip_naive(args, n_points, polys.size(), d_points, d_boxes, d_polys,
                d_edges, d_result);

      std::cout << "copy.." << std::endl;
      cl::copy(queue, d_result, result.begin(), result.end());

      std::cout << "make result sets.." << std::endl;
      for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] >= 0) {
          std::set<KEY> set;
          set.emplace(keys[(size_t)result[i]]);
          res.push_back(std::make_tuple(i, set));
        }
      }
      return res;
    } catch (cl::Error err) {
      std::cout << "Exception\n";
      std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
                << std::endl;
      return res;
    }
  }
};

#define TOL (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024) // length of vectors a, b, and c
std::string kernelsource = "__kernel void vadd(    \n"
                           " __global float* a,    \n"
                           " __global float* b,    \n"
                           " __global float* c,    \n"
                           " __global float* d,    \n"
                           " const unsigned int count)      \n"
                           " {                              \n"
                           "   int i = get_global_id(0);    \n"
                           "   if (i < count) {             \n"
                           "     d[i] = a[i] + b[i] + c[i]; \n"
                           "   }               \n"
                           " }\n";

int test_opencl(void) {
  std::vector<float> h_a(LENGTH);             // a vector
  std::vector<float> h_b(LENGTH);             // b vector
  std::vector<float> h_c(LENGTH);             // c vector
  std::vector<float> h_d(LENGTH, 0xdeadbeef); // d vector (result)

  cl::Buffer d_a; // device memory used for the input  a vector
  cl::Buffer d_b; // device memory used for the input  b vector
  cl::Buffer d_c; // device memory used for the input c vector
  cl::Buffer d_d; // device memory used for the output d vector

  // Fill vectors a and b with random float values
  int count = LENGTH;
  for (int i = 0; i < count; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
    h_c[i] = rand() / (float)RAND_MAX;
  }

  try {
    // Create a context
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);

    // Load in kernel source, creating a program object for the context

    cl::Program program(context, kernelsource, true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functor

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(
        program, "vadd");

    d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b = cl::Buffer(context, h_b.begin(), h_b.end(), true);
    d_c = cl::Buffer(context, h_c.begin(), h_c.end(), true);

    d_d = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

    vadd(cl::EnqueueArgs(queue, cl::NDRange(count)), d_a, d_b, d_c, d_d, count);

    cl::copy(queue, d_d, h_d.begin(), h_d.end());

    // Test the results
    int correct = 0;
    float tmp;
    for (int i = 0; i < count; i++) {
      tmp = h_a[i] + h_b[i] + h_c[i]; // assign element i of a+b+c to tmp
      tmp -= h_d[i]; // compute deviation of expected and output result
      if (tmp * tmp < TOL * TOL) // correct if square deviation is less than
                                 // tolerance squared
        correct++;
      else {
        printf(" tmp %f h_a %f h_b %f h_c %f h_d %f\n", tmp, h_a[i], h_b[i],
               h_c[i], h_d[i]);
      }
    }

    // summarize results
    printf("D = A+B+C:  %d out of %d results were correct.\n", correct, count);
    return count;
  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
              << std::endl;
    return -1;
  }
  return 0;
}
} // namespace PIP

#endif