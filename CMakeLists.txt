cmake_minimum_required(VERSION 3.10)
project(SimdSortBenchmark LANGUAGES CXX)

# --------------------------
# 1) Basic config
# --------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common compile flags
add_compile_options(-O3 -march=native -pthread -Wall -Wextra)

# Use folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# --------------------------
# 2) Include roots
# --------------------------
set(PARLAY_ORIG_INC  ${CMAKE_SOURCE_DIR}/extern/parlaylib/include)
set(PARLAY_NUMA_INC  ${CMAKE_SOURCE_DIR}/extern/parlaylib_numa/include)
set(SIMDSORT_INC     ${CMAKE_SOURCE_DIR}/extern/x86-smid-sort/src)

# Your benchmark sources (current dir only)
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cpp")

if (NOT BENCH_SOURCES)
  message(FATAL_ERROR "No .cpp files found in ${CMAKE_SOURCE_DIR}")
endif()

# --------------------------
# 3) Helper function to build two variants per source
# --------------------------
function(add_dual_parlay_executables SRC)
  get_filename_component(BASE ${SRC} NAME_WE)

  # -------------------- orig variant --------------------
  set(TGT_ORIG "${BASE}_orig")
  add_executable(${TGT_ORIG} ${SRC})

  target_include_directories(${TGT_ORIG} PRIVATE
    ${PARLAY_ORIG_INC}
    ${SIMDSORT_INC}
    ${CMAKE_SOURCE_DIR}        # so you can include local helper headers
  )

  target_link_libraries(${TGT_ORIG} PRIVATE pthread)

  target_compile_definitions(${TGT_ORIG} PRIVATE
    PARLAY_BENCH_VARIANT_ORIG=1
  )

  set_target_properties(${TGT_ORIG} PROPERTIES
    OUTPUT_NAME "${TGT_ORIG}"
    FOLDER "orig"
  )

  # -------------------- numa variant --------------------
  set(TGT_NUMA "${BASE}_numa")
  add_executable(${TGT_NUMA} ${SRC})

  target_include_directories(${TGT_NUMA} PRIVATE
    ${PARLAY_NUMA_INC}
    ${SIMDSORT_INC}
    ${CMAKE_SOURCE_DIR}
  )

  # numa-aware variant typically needs libnuma on Linux
  find_library(NUMA_LIB numa)
  if (NUMA_LIB)
    target_link_libraries(${TGT_NUMA} PRIVATE pthread ${NUMA_LIB})
  else()
    target_link_libraries(${TGT_NUMA} PRIVATE pthread)
    message(WARNING "libnuma not found. ${TGT_NUMA} may fail to link if it uses NUMA APIs.")
  endif()

  target_compile_definitions(${TGT_NUMA} PRIVATE
    PARLAY_BENCH_VARIANT_NUMA=1
  )

  set_target_properties(${TGT_NUMA} PROPERTIES
    OUTPUT_NAME "${TGT_NUMA}"
    FOLDER "numa"
  )
endfunction()

# --------------------------
# 4) Add all benchmarks
# --------------------------
foreach(SRC ${BENCH_SOURCES})
  message(STATUS "Adding dual executables for: ${SRC}")
  add_dual_parlay_executables(${SRC})
endforeach()
