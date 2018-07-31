import os
import cv2
import numpy as np
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""---------载入已经训练好的模型---------"""
new_model = load_model('my_model_ep20.h5')

"""---------用opencv载入一张待测图片-----"""
# 载入图片
src = cv2.imread('images/6.png')
cv2.imshow("test picture", src)

# 将图片转化为28*28的灰度图
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.resize(src, (28, 28))
dst=dst.astype(np.float32)

# 将灰度图转化为1*784的能够输入的网络的数组
picture=1-dst/255
picture=np.reshape(picture,(1,28,28,1))

# 用模型进行预测
y = new_model.predict(picture)
print("softmax:")
for i,prob in enumerate(y[0]):
    print("class{},Prob:{}".format(i,prob))
result = np.argmax(y)
print("你写的数字是：", result)
print("对应的概率是：",np.max(y[0]))
cv2.waitKey(20170731)