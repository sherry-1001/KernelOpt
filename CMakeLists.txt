cmake_minimum_required(VERSION 3.12.4)

project(KernelOpt VERSION 1.0
        DESCRIPTION "A personal practice AI kernel optimization"
        LANGUAGES CXX C)
        
set(OPT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mfma -mrdseed -mrdrnd -mclwb -mclflushopt")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer")
elseif(CMAKE_BUILD_TYPE STREQUAL "FastDebug")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O2 -march=native")
    set(CMAKE_C_FLAGS "-Wall -Wextra -g -O2 -march=native")
else()
    message(FATAL_ERROR "Invalid build type!")
endif()

# add 3rdparty onednn
set(DNNL_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(DNNL_ARCH_OPT_FLAGS "" CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE INTERNAL "" FORCE)
# add_subdirectory(3rdparty/oneDNN)

set(SOURCE, graph_visit.cpp)

add_executable(omp_matmul omp_matmul.cpp)
add_executable(naive_matmul naive_matmul.cpp)
# add_executable(ref_matmul ref_matmul.cpp)
# target_link_libraries(ref_matmul PUBLIC dnnl)
add_executable(test_matmul test_matmul.cpp)
add_executable(test_opt_matmul test_opt_matmul.cpp)

add_executable(test_graph test_graph.cpp ${SOURCE})

