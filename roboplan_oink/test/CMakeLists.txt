find_package(GTest REQUIRED)
find_package(roboplan_example_models REQUIRED)

# Build Oink integration tests
add_executable(test_oink test_oink.cpp)
set_property(TARGET test_oink PROPERTY CXX_STANDARD 20)
target_link_libraries(
  test_oink
  roboplan::roboplan
  roboplan_oink
  GTest::gmock_main
  GTest::gtest_main
  roboplan_example_models::roboplan_example_models
)

# Build FrameTask unit tests
add_executable(test_frame_task tasks/test_frame_task.cpp)
set_property(TARGET test_frame_task PROPERTY CXX_STANDARD 20)
target_link_libraries(
  test_frame_task
  roboplan::roboplan
  roboplan_oink
  GTest::gmock_main
  GTest::gtest_main
  roboplan_example_models::roboplan_example_models
)

# Build VelocityLimit unit tests
add_executable(test_velocity_limit constraints/test_velocity_limit.cpp)
set_property(TARGET test_velocity_limit PROPERTY CXX_STANDARD 20)
target_link_libraries(
  test_velocity_limit
  roboplan::roboplan
  roboplan_oink
  GTest::gmock_main
  GTest::gtest_main
  roboplan_example_models::roboplan_example_models
)

# Build PositionLimit unit tests
add_executable(test_position_limit constraints/test_position_limit.cpp)
set_property(TARGET test_position_limit PROPERTY CXX_STANDARD 20)
target_link_libraries(
  test_position_limit
  roboplan::roboplan
  roboplan_oink
  GTest::gmock_main
  GTest::gtest_main
  roboplan_example_models::roboplan_example_models
)

# Build FrameContainmentBarrier unit tests
add_executable(test_frame_containment_barrier barriers/test_frame_containment_barrier.cpp)
set_property(TARGET test_frame_containment_barrier PROPERTY CXX_STANDARD 20)
target_link_libraries(
  test_frame_containment_barrier
  roboplan::roboplan
  roboplan_oink
  GTest::gmock_main
  GTest::gtest_main
  roboplan_example_models::roboplan_example_models
)

# If ament_gtest is detected, make the tests visible in ROS 2.
find_package(ament_cmake_gtest QUIET)
if ( ament_cmake_gtest_FOUND )
    ament_add_gtest_test(test_oink)
    ament_add_gtest_test(test_frame_task)
    ament_add_gtest_test(test_velocity_limit)
    ament_add_gtest_test(test_position_limit)
    ament_add_gtest_test(test_frame_containment_barrier)
else()
    include(GoogleTest)
    gtest_discover_tests(test_oink)
    gtest_discover_tests(test_frame_task)
    gtest_discover_tests(test_velocity_limit)
    gtest_discover_tests(test_position_limit)
    gtest_discover_tests(test_frame_containment_barrier)
endif()
