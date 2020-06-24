#cmake最低版本要求
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#目标生成项目
project(example-app)
#寻找库文件包
find_package(OpenCV REQUIRED)
#输出信息
# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 将所有文件(.cpp和.h文件)合并保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

# 指定生成目标
add_executable(example-app ${DIR_SRCS})

#连接外部依赖库
target_link_libraries(example-app "${OpenCV_LIBS}")

#设置配置
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)