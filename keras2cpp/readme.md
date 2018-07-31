# [将Keras训练的模型部署到C++平台上的可行方案](https://blog.csdn.net/qq_25109263/article/details/81285952)

## 1.编译tensorflow

Windows下使用CMake编译，ubuntu使用bazel编译，时间很长

## 2.keras训练

```
python keras_train.py
python load_h5_test.py
```
## 3.转换为pb模型

```
python h5_to_pb.py
python load_pb_test.py
```

## 4.C++代码使用

见keras2cpp.cpp