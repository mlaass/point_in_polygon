#ifndef TIMER_HPP_INCLUDED__
#define TIMER_HPP_INCLUDED__

#include <chrono>
#include <cstdint>
#include <utility>

namespace timer {
// shorthand type names of the clocks
using hi_res =
    ::std::chrono::high_resolution_clock; // clock with the shortest tick period
using steady = ::std::chrono::steady_clock; // monotonic clock
using wall = ::std::chrono::system_clock;   // system-wide realtime clock
// The default template argument for `timer::clock` instantiations is the steady
// clock with the highest resolution.
using apt = ::std::conditional_t<hi_res::is_steady, hi_res, steady>;

// shorthand unit and value types of the time interval returned
using ns_i64 = ::std::chrono::duration<::std::int64_t, ::std::nano>;
using us_i64 = ::std::chrono::duration<::std::int64_t, ::std::micro>;
using ms_i64 = ::std::chrono::duration<::std::int64_t, ::std::milli>;
using s_i64 = ::std::chrono::duration<::std::int64_t>;
using m_i64 = ::std::chrono::duration<::std::int64_t, ::std::ratio<60>>;
using h_i64 = ::std::chrono::duration<::std::int64_t, ::std::ratio<3600>>;
using ns_dbl = ::std::chrono::duration<double, ::std::nano>;
using us_dbl = ::std::chrono::duration<double, ::std::micro>;
using ms_dbl = ::std::chrono::duration<double, ::std::milli>;
using s_dbl = ::std::chrono::duration<double>;
using m_dbl = ::std::chrono::duration<double, ::std::ratio<60>>;
using h_dbl = ::std::chrono::duration<double, ::std::ratio<3600>>;

// The `timer::clock` class provides methods to measure the elapsed time, to
// reset the start time, and to suspend time measuring.
template <typename clockT = apt> struct clock final {
  /** \brief Return the time elapsed between the creation of the clock object or
   * the latest call of `reset` and either the current time point or the
   * timepoint when the clock object was paused. Time spans wherein the clock
   * was paused are subtracted from the return value. \return Time elapsed in
   * the unit and value type specified by the `unitT` template parameter.
   *         Nanoseconds as uint64_t is the default setting. */
  template <typename unitT = ns_i64>
  [[nodiscard]] auto elapsed() const noexcept {
    return ::std::chrono::duration_cast<unitT>(
               (m_break == typename clockT::time_point{} ? clockT::now()
                                                         : m_break) -
               m_start)
        .count();
  }

  /** \brief Reset the start time to the current time, reset the status to not
   * paused. Return the time elapsed between the creation of the clock object or
   * the latest call of `reset` and either the current time point or the
   * timepoint when the clock object was paused. Time spans wherein the clock
   * was paused are subtracted from the return value. \return Time elapsed in
   * the unit and value type specified by the `unitT` template parameter.
   *         Nanoseconds as uint64_t is the default setting. */
  template <typename unitT = ns_i64> auto reset() noexcept {
    resume();
    const auto tmp{std::move(m_start)};
    m_start = clockT::now();
    return ::std::chrono::duration_cast<unitT>(m_start - tmp).count();
  }

  /** \brief Set the clock into a paused status where time measuring is
   * suspended. The time elapsed between calls of `pause` and `resume` is
   * excluded from the time that `elapsed` or `reset` returns. Use `pause` to
   * exclude processes from time measuring which would overlay the process times
   * you are interested in. Or use `pause` to bridge the period between the end
   * of the process of interest and the point in your code where you want to
   * process the time measured. */
  void pause() noexcept {
    if (m_break == typename clockT::time_point{})
      m_break = clockT::now();
  }

  /** \brief End a paused period and continue time measuring. Reset the status
   * to not paused. */
  void resume() noexcept {
    if (m_break != typename clockT::time_point{}) {
      m_start += clockT::now() - m_break;
      m_break = typename clockT::time_point{};
    }
  }

private:
  // Start time of the period to be measured. (Might be advanced by `resume` or
  // `reset` to address the paused time.)
  typename clockT::time_point m_start{clockT::now()};
  // Either the start time of the suspended period, or the begin of the clock
  // epoch to indicate a not paused status.
  typename clockT::time_point m_break{};
};

clock()->clock<apt>; // Deduction guide for the default constructor.
} // namespace timer

#endif // TIMER_HPP_INCLUDED__