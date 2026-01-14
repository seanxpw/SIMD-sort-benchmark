cmake_minimum_required(VERSION 3.10)

project(SimdSortBenchmark LANGUAGES CXX)

# ==========================================
# 1. 基础配置
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
# -O3: 最高优化
# -march=native: 针对本机 CPU (Cascade Lake) 开启 AVX512
# -pthread: 开启多线程支持
add_compile_options(-O3 -march=native -pthread -Wall)

# ==========================================
# 2. 包含路径
# ==========================================
# 包含 ParlayLib 头文件
include_directories(${CMAKE_SOURCE_DIR}/extern/parlaylib/include)
include_directories(${CMAKE_SOURCE_DIR}/extern/x86-smid-sort/src)
# ==========================================
# 3. 自动发现并编译所有 CPP 文件
# ==========================================

# 查找当前目录下所有的 .cpp 文件，存入 SOURCE_FILES 变量
file(GLOB SOURCE_FILES "*.cpp")

# 遍历每一个找到的源文件
foreach(SOURCE_FILE ${SOURCE_FILES})
    # 获取文件名（不带扩展名），例如 "bench_bw.cpp" -> "bench_bw"
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)

    # 打印消息，让你知道哪些文件被编译了
    message(STATUS "Adding executable: ${EXE_NAME} from source ${SOURCE_FILE}")

    # 创建可执行文件
    add_executable(${EXE_NAME} ${SOURCE_FILE})

    # 链接必要的库 (pthread)
    target_link_libraries(${EXE_NAME} pthread)
endforeach()