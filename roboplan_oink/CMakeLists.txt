cmake_minimum_required(VERSION 3.22)
project(roboplan_oink VERSION 0.0.0)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies.
find_package(roboplan REQUIRED)

# Try to find OsqpEigen, build from source if not found or version < 1.0.0
find_package(OsqpEigen QUIET)
set(BUILD_OSQPEIGEN_FROM_SOURCE FALSE)

if(NOT OsqpEigen_FOUND)
  message(STATUS "OsqpEigen not found. Building from source in external/osqp-eigen")
  set(BUILD_OSQPEIGEN_FROM_SOURCE TRUE)
elseif(OsqpEigen_VERSION VERSION_LESS "1.0.0")
  if(DEFINED OsqpEigen_VERSION)
    message(STATUS "Found OsqpEigen ${OsqpEigen_VERSION} but version < 1.0.0. Building newer version from source")
  else()
    message(STATUS "Found OsqpEigen with unknown version. Building from source to ensure v1.0.0+")
  endif()
  set(BUILD_OSQPEIGEN_FROM_SOURCE TRUE)
endif()

if(BUILD_OSQPEIGEN_FROM_SOURCE)
  # When building OsqpEigen from source, we need OSQP >= 1.0.0
  find_package(osqp QUIET)
  set(BUILD_OSQP_FROM_SOURCE FALSE)

  if(NOT osqp_FOUND)
    set(BUILD_OSQP_FROM_SOURCE TRUE)
  elseif(osqp_VERSION VERSION_LESS "1.0.0")
    set(BUILD_OSQP_FROM_SOURCE TRUE)
    # Clear cache variables
    unset(osqp_FOUND)
    unset(osqp_DIR CACHE)
    unset(osqp_INCLUDE_DIR CACHE)
    unset(osqp_INCLUDE_DIRS CACHE)
    unset(osqp_LIBRARIES CACHE)
    unset(osqp_LIBRARY CACHE)
  else()
    message(STATUS "Found OSQP ${osqp_VERSION}: ${osqp_DIR}")
  endif()

  # Build OSQP from source if needed
  if(BUILD_OSQP_FROM_SOURCE)
    message(STATUS "Building OSQP v1.0.0 from source (external/osqp)")

    # Disable OSQP tests and demos
    set(OSQP_BUILD_DEMO OFF CACHE INTERNAL "Disable OSQP demo")
    set(OSQP_BUILD_UNITTESTS OFF CACHE INTERNAL "Disable OSQP tests")

    # Save and disable BUILD_TESTING to prevent OSQP from building tests
    set(BUILD_TESTING_OSQP ${BUILD_TESTING})
    set(BUILD_TESTING OFF)

    # Add OSQP from submodule
    # Note: Not using EXCLUDE_FROM_ALL so it will be installed for downstream packages
    add_subdirectory(external/osqp)

    # Restore BUILD_TESTING
    set(BUILD_TESTING ${BUILD_TESTING_OSQP})
    unset(BUILD_TESTING_OSQP)

    # OSQP creates non-namespaced targets when built via add_subdirectory
    # Create namespaced aliases so OsqpEigen can find them
    if(TARGET osqp AND NOT TARGET osqp::osqp)
      add_library(osqp::osqp ALIAS osqp)
    endif()
    if(TARGET osqpstatic AND NOT TARGET osqp::osqpstatic)
      add_library(osqp::osqpstatic ALIAS osqpstatic)
    endif()

    # Mark osqp as found and set variables so find_package(osqp) succeeds
    set(osqp_FOUND TRUE CACHE BOOL "OSQP was built from source" FORCE)
    set(osqp_VERSION "1.0.0" CACHE STRING "OSQP version" FORCE)
    set(osqp_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/osqp" CACHE PATH "Path to OSQP build" FORCE)

    # Set OSQP version detection variables to skip OsqpEigen's try_compile tests
    # (try_compile doesn't work with ALIAS targets from add_subdirectory)
    set(OSQP_IS_V1 TRUE CACHE BOOL "OSQP is version 1.0+" FORCE)
    set(OSQP_IS_V1_FINAL TRUE CACHE BOOL "OSQP is version 1.0.0 final" FORCE)
  endif()

  # Save current BUILD_TESTING value and disable testing for OsqpEigen
  set(BUILD_TESTING_OSQPEIGEN ${BUILD_TESTING})
  set(BUILD_TESTING OFF)
  set(OSQPEIGEN_COMPILE_tests OFF CACHE INTERNAL "Disable OsqpEigen tests")

  # Add the external osqp-eigen project
  # Note: Not using EXCLUDE_FROM_ALL so it can be installed for downstream packages
  add_subdirectory(external/osqp-eigen)

  # Restore the original BUILD_TESTING value
  set(BUILD_TESTING ${BUILD_TESTING_OSQPEIGEN})
  unset(BUILD_TESTING_OSQPEIGEN)
endif()

# Ament CMake is not immediately required, but we include it at the top so that
# if it is found we can use it in build testing before requiring a
# find_package(ament_cmake_test REQUIRED). Otherwise this can cause infinite recursion
# issues with symlink installs.
find_package(ament_cmake QUIET)

# Add libraries
add_library(roboplan_oink SHARED
    src/optimal_ik.cpp
    src/tasks/frame.cpp
    src/constraints/position_limit.cpp
    src/constraints/velocity_limit.cpp
)
target_include_directories(roboplan_oink PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(roboplan_oink
    roboplan::roboplan
    OsqpEigen::OsqpEigen
)
set_property(TARGET roboplan_oink PROPERTY CXX_STANDARD 20)

install(
  TARGETS
  roboplan_oink
  EXPORT roboplan_oink-targets
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Generate the package configuration file
configure_file(
    "cmake/roboplan_oinkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/roboplan_oinkConfig.cmake"
    @ONLY
)

# Create and install the configuration files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/roboplan_oinkConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/roboplan_oinkConfig.cmake
    INSTALL_DESTINATION lib/cmake/roboplan_oink
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/roboplan_oinkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/roboplan_oinkConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/roboplan_oinkConfigVersion.cmake"
    DESTINATION
        "lib/cmake/roboplan_oink"
)

# Install the headers and targets
install(DIRECTORY
            include/roboplan_oink
        DESTINATION lib/cmake/)

install(EXPORT roboplan_oink-targets
        FILE roboplan_oinkTargets.cmake
        NAMESPACE roboplan_oink::
        DESTINATION lib/cmake/roboplan_oink)

install(TARGETS roboplan_oink
        EXPORT roboplan_oink
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include
        )

# Add the unit tests, if enabled.
option(BUILD_TESTING "" ON)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# If ament_cmake is detected, export the package to be visible in ROS 2.
if ( ament_cmake_FOUND )
    message(STATUS "ament_cmake found. Exporting for ROS 2.")
    ament_package()
endif()
