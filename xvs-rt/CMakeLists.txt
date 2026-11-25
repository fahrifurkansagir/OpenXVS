# OpenXVS v2.0 — Production CMakeLists.txt for `xvs-rt` (CI / Jetson / Linux with Corrosion)
cmake_minimum_required(VERSION 3.22)
project(xvs_rt VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(XVS_RT_USE_CORROSION "Import xvs-ffi via Corrosion when available" ON)
option(XVS_RT_ENABLE_MALLOC_WRAP "Link -Wl,--wrap=malloc zero-alloc audit traps on Linux" ON)

set(XVS_RT_STRICT_FLAGS
    -fno-exceptions
    -fno-rtti
    -Wall
    -Wextra
    -Werror
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wdouble-promotion
)

if(XVS_RT_USE_CORROSION)
    find_package(Corrosion QUIET)
    if(Corrosion_FOUND)
        corrosion_import_crate(MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../xvs-core/Cargo.toml" CRATES xvs-ffi)
        set(XVS_FFI_TARGET xvs_core)
    endif()
endif()

if(NOT DEFINED XVS_FFI_TARGET)
    add_library(xvs_core_static STATIC IMPORTED GLOBAL)
    set_target_properties(xvs_core_static PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../xvs-core/target/release/libxvs_core.a"
    )
    set(XVS_FFI_TARGET xvs_core_static)
endif()

add_library(xvs_daidalus_boundary STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/daidalus_boundary/xvs_daidalus_bridge.cpp"
)
target_include_directories(xvs_daidalus_boundary PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/daidalus_boundary"
)
target_compile_options(xvs_daidalus_boundary PRIVATE ${XVS_RT_STRICT_FLAGS})

add_library(xvs_rt_core STATIC
    src/mh_vit.cpp
    src/ecbf_qp.cpp
    src/rt_pipeline.cpp
)
target_include_directories(xvs_rt_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../xvs-core/crates/xvs-ffi/include"
)
target_compile_options(xvs_rt_core PRIVATE ${XVS_RT_STRICT_FLAGS})
target_link_libraries(xvs_rt_core PUBLIC xvs_daidalus_boundary ${XVS_FFI_TARGET} m dl pthread)

add_executable(test_xvs_rt tests/test_xvs_rt.cpp)
target_compile_options(test_xvs_rt PRIVATE ${XVS_RT_STRICT_FLAGS})
target_link_libraries(test_xvs_rt PRIVATE xvs_rt_core)

add_executable(xvs_rt_bench src/main_bench.cpp src/malloc_audit.cpp)
target_compile_options(xvs_rt_bench PRIVATE ${XVS_RT_STRICT_FLAGS})
target_link_libraries(xvs_rt_bench PRIVATE xvs_rt_core)
if(XVS_RT_ENABLE_MALLOC_WRAP AND UNIX AND NOT APPLE)
    target_link_options(xvs_rt_bench PRIVATE
        -Wl,--wrap=malloc
        -Wl,--wrap=calloc
        -Wl,--wrap=realloc
        -Wl,--wrap=free
    )
endif()

enable_testing()
add_test(NAME xvs_rt_unit_tests COMMAND test_xvs_rt)
