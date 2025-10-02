#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

// Provide a compatibility shim for Kokkos releases prior to 4.4.1 which do not
// expose the SequentialHostInit allocation property. The Parthenon submodule
// expects the symbol to exist, so we define a no-op placeholder when building
// against an older Kokkos revision.

#if defined(KOKKOS_VERSION) && (KOKKOS_VERSION < 40401)

namespace Kokkos {

struct SequentialHostInitTag {};
inline constexpr SequentialHostInitTag SequentialHostInit{};

namespace Impl {

template <>
struct is_view_ctor_property<Kokkos::SequentialHostInitTag> : std::true_type {};

// Treat SequentialHostInit like a regular allocation property so it can be
// forwarded through view allocation helpers. The placeholder does not modify
// allocation behaviour for legacy Kokkos versions.
template <typename P>
struct ViewCtorProp<
    std::enable_if_t<std::is_same<std::decay_t<P>, Kokkos::SequentialHostInitTag>::value>,
    P> {
  using type = Kokkos::SequentialHostInitTag;

  KOKKOS_FUNCTION ViewCtorProp() = default;
  KOKKOS_FUNCTION ViewCtorProp(const ViewCtorProp &) = default;
  KOKKOS_FUNCTION ViewCtorProp &operator=(const ViewCtorProp &) = default;

  KOKKOS_FUNCTION explicit ViewCtorProp(const type &) {}

  type value = type{};
};

} // namespace Impl
} // namespace Kokkos

#endif // defined(KOKKOS_VERSION) && (KOKKOS_VERSION < 40401)
