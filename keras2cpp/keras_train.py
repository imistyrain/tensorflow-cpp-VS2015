from tensorflow.examples.tutorials.mnist import *
from keras.models import *
from keras.layers import *
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载数据集
mnist=input_data.read_data_sets("/tmp/mnist",one_hot=True)
train_X=mnist.train.images
train_Y=mnist.train.labels
test_X=mnist.test.images
test_Y=mnist.test.labels

train_X=train_X.reshape((55000,28,28,1))
test_X=test_X.reshape((test_X.shape[0],28,28,1))

print("type of train_X:",type(train_X))
print("size of train_X:",np.shape(train_X))
print("train_X:",train_X)

print("type of train_Y:",type(train_Y))
print("size of train_Y:",np.shape(train_Y))
print("train_Y:",train_Y)

print("num of test:",test_X.shape[0])


# 配置模型结构
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1),padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))


model.add(Conv2D(64, (3, 3), activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(625,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

# 训练模型
epochs=20
model.fit(train_X, train_Y, batch_size=32, epochs=epochs)

# 用测试集去评估模型的准确度
accuracy=model.evaluate(test_X,test_Y,batch_size=20)
print('\nTest accuracy:',accuracy[1])

save_model(model,'my_model_ep{}.h5'.format(epochs))