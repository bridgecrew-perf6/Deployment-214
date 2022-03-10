import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_test = x_test.reshape(x_test.shape[0],1,28,28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Conv2D(8,kernel_size = 3 , padding = 'same', activation = 'relu',input_shape=(1,28,28)))
model.add(MaxPool2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(64,kernel_size = 3 , padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(128,kernel_size = 3 , padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(10,kernel_size = 3 , padding = 'same', activation = 'softmax'))
model.add(MaxPool2D(pool_size = (4,4),padding='same'))
model.add(Flatten())

model.compile(loss = 'categorical_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=60000,
          epochs = 75,
          steps_per_epoch = 1,
          validation_data=(x_test,y_test),
          verbose = 2,#
          validation_steps=1)

model.load_weights("model.h5")

#reshaping input for model
#img = np.expand_dims(img,axis=0)
#img_class = model.predict_classes(img)
#prediction = img_class[0]
#classname = img_class[0]
#print("Class: ",classname)
#model.save('./model.h5')

#deploy_model = load_model('./model.h5',compile=True)
