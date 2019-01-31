from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import Model
import matplotlib.pylab as plt
import numpy as np
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_set = np.concatenate((x_train,x_test),axis = 0)
y_set = np.concatenate((y_train,y_test),axis = 0)
x_train = x_set[0:35000]
y_train = y_set[0:35000]
x_test = x_set[35000:75000]
y_test = y_set[35000:75000]


# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


 #convert class vectors to binary class matrices - this is for use in the
 #categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax',name = "dense"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

get_maxPool1_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
maxP1_output = get_maxPool1_layer_output([x_train])[0]

get_maxPool2_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
maxP2_output = get_maxPool2_layer_output([x_train])[0]

get_last_layer_output = K.function([model.layers[0].input],
                                  [model.layers[6].output])
layer_output = get_last_layer_output([x_train])[0]

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig, ax = plt.subplots(
    nrows=1,
    ncols=2,
    sharex=True,
    sharey=True, )
 
ax = ax.flatten()
#img0 = maxP1_output.reshape(12,12)
#img0 = maxP2_output[0,:,:,0].reshape(4,4)
img0 = layer_output[0].reshape(1,10)
ax[0].imshow(img0, cmap='Greys', interpolation='nearest')
ax[0].imshow(img0,cmap='Greys', interpolation='nearest')
#ax[2].imshow(img2,cmap='Greys', interpolation='nearest')
 
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#plt.plot(range(1, 11), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
