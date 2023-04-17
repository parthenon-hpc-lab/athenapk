// Path: tst/unit/tests.cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "../../src/interp.hpp"

double interpolate_line(double x) {
  // f(x) = x
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

double interpolate_broken_line(double x) {
  // f(x) = x
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

double interpolate_parabola(double x) {
  // f(x) = x^2
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0};
  MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

double interpolate_cubic(double x) {
  // f(x) = x^3
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 8.0, 27.0, 64.0, 125.0};
  MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

TEST_CASE("interpolation of monotone profiles", "[interpolate_line]") {
  REQUIRE(interpolate_line(1.0) == 1.0);
  REQUIRE(interpolate_line(2.5) == 2.5);
}

TEST_CASE("interpolation of monotone profiles", "[interpolate_parabola]") {
  REQUIRE(interpolate_parabola(1.0) == 1.0);
  REQUIRE(interpolate_parabola(2.5) == 2.5*2.5);
  REQUIRE(interpolate_parabola(4.0) == 16.0);
}

TEST_CASE("interpolation of monotone profiles", "[interpolate_cubic]") {
  REQUIRE(interpolate_cubic(1.0) == 1.0);
  REQUIRE(interpolate_cubic(2.5) == 2.5*2.5*2.5);
  REQUIRE(interpolate_cubic(4.0) == 64.0);
}

TEST_CASE("monotone interpolation of non-monotone profiles", "[interpolate_line_nonmonotone]") {
  // ...
}
