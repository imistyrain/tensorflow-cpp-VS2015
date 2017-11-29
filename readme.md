# tensorflow C++ VS2015

![](https://i.imgur.com/cMydLg6.jpg)

## 1.编译安装tensorflow

	git clone https://github.com/tensorflow/tensorflow
	cd tensorflow
	mkdir build
	cmake ../tensorflow/contrib/cmake

勾选tensorflow_BUILD_SHARED_LIB和tensorflow_ENABLE_GPU
去掉tensorflow_ENABLE_GRPC_SUPPORT

编译生成好的项目，最后别忘了在INSTALL工程上右键生成，以便把所需的文件拷贝至相应的路径。

## 2.编译本工程

打开tensorflow-cpp.sln编译即可

## 3.新建项目

### 1.注意在Inlcude文件夹选项下加入:

	C:\Program Files\tensorflow\include
	D:\Anaconda\envs\Python35\Lib\site-packages\tensorflow\include
	D:\CNN\tensorflow\build\external\nsync\public

### 2.在Lib文件夹下选项加入：

	D:\CNN\tensorflow\build\

### 3.连接器->输入：

	png\install\lib\png12.lib
	sqlite\install\lib\sqlite.lib
	zlib\install\lib\zlibstatic.lib
	gif\install\lib\giflib.lib
	jpeg\install\lib\libjpeg.lib
	lmdb\install\lib\lmdb.lib
	farmhash\install\lib\farmhash.lib
	fft2d\\src\lib\fft2d.lib
	highwayhash\install\lib\highwayhash.lib
	nsync\install\lib\nsync.lib
	jsoncpp\src\jsoncpp\src\lib_json\$(Configuration)\jsoncpp.lib
	protobuf\src\protobuf\$(Configuration)\libprotobuf.lib
	snappy\src\snappy\$(Configuration)\snappy.lib
	tf_cc_while_loop.dir\$(Configuration)\tf_cc_while_loop.lib
	tf_stream_executor.dir\$(Configuration)\tf_stream_executor.lib
	$(Configuration)\tf_protos_cc.lib
	$(Configuration)\tf_core_gpu_kernels.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cudart_static.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cuda.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cublas.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cublas_device.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cufft.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\curand.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\libx64\cupti.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cusolver.lib
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\cudnn.lib

### 4.连接器->命令行选项：

	/WHOLEARCHIVE:tf_core_lib.dir\$(Configuration)\tf_core_lib.lib
	/WHOLEARCHIVE:tf_core_cpu.dir\$(Configuration)\tf_core_cpu.lib
	/WHOLEARCHIVE:tf_core_framework.dir\$(Configuration)\tf_core_framework.lib
	/WHOLEARCHIVE:tf_core_kernels.dir\$(Configuration)\tf_core_kernels.lib
	/WHOLEARCHIVE:tf_cc_framework.dir\$(Configuration)\tf_cc_framework.lib
	/WHOLEARCHIVE:tf_cc.dir\$(Configuration)\tf_cc_ops.lib 
	/WHOLEARCHIVE:tf_core_direct_session.dir\$(Configuration)\tf_core_direct_session.lib 
	/WHOLEARCHIVE:tf_core_ops.dir\$(Configuration)\tf_core_ops.lib   
	/WHOLEARCHIVE:tf_stream_executor.dir\$(Configuration)\tf_stream_executor.lib
	/WHOLEARCHIVE:tf_cc.dir\$(Configuration)\tf_cc.lib 
	/WHOLEARCHIVE:tf_cc_ops.dir\$(Configuration)\tf_cc_ops.lib

## Note:

本机测试环境为VS2015、CUDA8.0、CUDNN7.0

## 参考：

* [Building a standalone C++ Tensorflow program on Windows](https://joe-antognini.github.io/machine-learning/windows-tf-project)
* [Tensorflow C++ 编译和调用图模型](http://blog.csdn.net/rockingdingo/article/details/75452711)
* [使用TensorFlow C++ API构建线上预测服务](http://mathmach.com/2017/10/09/tensorflow_c++_api_prediction_first/)