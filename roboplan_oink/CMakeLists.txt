cmake_minimum_required(VERSION 3.22)
project(roboplan_oink VERSION 0.0.0)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies.
find_package(roboplan REQUIRED)
find_package(OsqpEigen REQUIRED)

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
