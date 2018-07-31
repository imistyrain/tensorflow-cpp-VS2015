cmake_minimum_required(VERSION 3.10)
project(demo)
set(CMAKE_CXX_STANDARD 11)

# 配置.so文件存在的路径，默认会去/usr/local/lib下找，
# 放在其他地方的.so文件路径就要用下面的代码添加
link_directories(~/anaconda2/envs/tf/lib)

# 配置.h头文件路径，默认会去/usr/local/include下找,
# 放在其他地方的.h文件就要用以下代码添加进来
include_directories(
        ~/tensorflow
        ~/tensorflow/bazel-genfiles
        ~/tensorflow/bazel-bin/tensorflow
        ~/tools/tf-C/eigen3
)

# 配置生成的可执行文件名为hello,要用到的源文件为hello.cpp
add_executable(keras2cpp keras2cpp.cpp)

# 以下使能OpenCV的路径查找
find_package(OpenCV REQUIRED)

# 以下是将可执行文件与一些.so文件建立动态链接关系，
# 用到的有libtensorflow_cc.so，libtensorflow_framework.so,以及opencv相关的so
target_link_libraries(keras2cpp tensorflow_cc tensorflow_framework ${OpenCV_LIBS})