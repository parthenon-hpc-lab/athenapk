Include(FetchContent)

if (NOT TARGET catch2_define)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.3.2 # or a later release
  )

  FetchContent_MakeAvailable(Catch2)
endif()

# These tests can use the Catch2-provided main
#add_executable(tests tests.cpp)
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
#target_link_libraries(tests PRIVATE parthenon)

#list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
#include(CTest)
#include(Catch)
#catch_discover_tests(tests)
