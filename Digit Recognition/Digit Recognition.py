# 載入 MNIST 的 Keras datasets
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

# Data normalization
x_train = x_train / 255
x_valid = x_valid / 255


# 對標籤進行分類編碼
import tensorflow.keras as keras
num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

# 具現化模型
from tensorflow.keras.models import Sequential
model = Sequential()

# construct input Layer
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# First convolution layer (32個3x3的卷積核， 步長為1)
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape = (28, 28, 1)))

# buld hidden layer
# First pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2)))

# Second convolution layer（64個5x5的卷積核， 步長為1，使用填充）
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# Second pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2)))

# subsequent convolution layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))

# The last pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2)))

# Fully Connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compiling and summary
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

# training model
history = model.fit(
    x_train, y_train, epochs = 5, verbose = 1, validation_data = (x_valid, y_valid)
)

# check accuracy
from matplotlib import pyplot as plt
history = history.history
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'], loc = 'lower right')
plt.title('Accuracy')
