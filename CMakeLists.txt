cmake_minimum_required(VERSION 3.10)
project(SimdSortBenchmark LANGUAGES CXX C) 

# ==========================================
# 1. 基础配置
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 通用编译选项
add_compile_options(-O3 -march=native -pthread -Wall -Wextra)

# 让 IDE 显示文件夹结构
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# 定义路径
set(PARLAY_ORIG_INC  ${CMAKE_SOURCE_DIR}/extern/parlaylib/include)
set(PARLAY_NUMA_INC  ${CMAKE_SOURCE_DIR}/extern/parlaylib_numa/include)
set(SIMDSORT_INC     ${CMAKE_SOURCE_DIR}/extern/x86-smid-sort/src)

# ==========================================
# 2. 自动编译本地 numactl
# ==========================================
include(ExternalProject)

# 检查 submodule 是否存在
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/numactl/autogen.sh")
    message(FATAL_ERROR "numactl submodule missing! Please run: git submodule update --init --recursive")
endif()

# 定义 numactl 的安装路径
set(NUMA_INSTALL_DIR "${CMAKE_BINARY_DIR}/numactl_build")
set(NUMA_LIB_PATH    "${NUMA_INSTALL_DIR}/lib/libnuma.so") 
set(NUMA_INC_PATH    "${NUMA_INSTALL_DIR}/include")        

# 【关键修复 1】强制先创建这个目录，防止 CMake 报错说目录不存在
file(MAKE_DIRECTORY ${NUMA_INC_PATH})

# 创建外部构建任务
ExternalProject_Add(build_local_numa
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/extern/numactl"
    PREFIX "${NUMA_INSTALL_DIR}"
    INSTALL_DIR "${NUMA_INSTALL_DIR}"

    # 【关键修复 2】必须开启源码内构建！
    # 因为 autogen.sh 生成的 Makefile 在源码目录，而不是 build 目录
    # 如果不加这行，会报错 "No targets specified and no makefile found"
    BUILD_IN_SOURCE 1

    # 配置命令：生成 configure 并运行
    CONFIGURE_COMMAND ./autogen.sh && ./configure --prefix=${NUMA_INSTALL_DIR} --silent

    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${NUMA_LIB_PATH}
)

# 创建一个“假”的 CMake 目标
add_library(libnuma_local SHARED IMPORTED GLOBAL)
set_target_properties(libnuma_local PROPERTIES
    IMPORTED_LOCATION "${NUMA_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INC_PATH}"
)
add_dependencies(libnuma_local build_local_numa)

# ==========================================
# 3. 查找源码
# ==========================================
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cpp")

if (NOT BENCH_SOURCES)
  message(FATAL_ERROR "No .cpp files found in ${CMAKE_SOURCE_DIR}")
endif()

# ==========================================
# 4. 定义构建函数
# ==========================================
function(add_dual_parlay_executables SRC)
  get_filename_component(BASE ${SRC} NAME_WE)

  # -------------------- orig variant --------------------
  set(TGT_ORIG "${BASE}_orig")
  add_executable(${TGT_ORIG} ${SRC})

  target_include_directories(${TGT_ORIG} PRIVATE
    ${PARLAY_ORIG_INC}
    ${SIMDSORT_INC}
    ${CMAKE_SOURCE_DIR}
    # 【关键修复 3】即便不链接库，也加上头文件路径，解决 _orig 编译时的 fatal error: numa.h missing
    ${NUMA_INC_PATH} 
  )

  target_link_libraries(${TGT_ORIG} PRIVATE pthread)

  target_compile_definitions(${TGT_ORIG} PRIVATE
    PARLAY_BENCH_VARIANT_ORIG=1
  )

  set_target_properties(${TGT_ORIG} PROPERTIES OUTPUT_NAME "${TGT_ORIG}" FOLDER "orig")

  # -------------------- numa variant --------------------
  set(TGT_NUMA "${BASE}_numa")
  add_executable(${TGT_NUMA} ${SRC})

  target_include_directories(${TGT_NUMA} PRIVATE
    ${PARLAY_NUMA_INC}
    ${SIMDSORT_INC}
    ${CMAKE_SOURCE_DIR}
    # 这里链接 libnuma_local 会自动带入 NUMA_INC_PATH，所以不用显式写
  )

  # 链接我们自定义的 libnuma_local
  target_link_libraries(${TGT_NUMA} PRIVATE pthread libnuma_local)

  target_compile_definitions(${TGT_NUMA} PRIVATE
    PARLAY_BENCH_VARIANT_NUMA=1
  )

  set_target_properties(${TGT_NUMA} PROPERTIES OUTPUT_NAME "${TGT_NUMA}" FOLDER "numa")

endfunction()

# ==========================================
# 5. 添加所有 benchmark
# ==========================================
foreach(SRC ${BENCH_SOURCES})
  message(STATUS "Adding dual executables for: ${SRC}")
  add_dual_parlay_executables(${SRC})
endforeach()