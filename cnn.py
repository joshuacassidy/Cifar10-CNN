import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.preprocessing.image import ImageDataGenerator


def list_files(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

classes = 10
train_dir = 'cifar10/train'
test_dir = 'cifar10/test'
validation_dir = 'cifar10/validation'

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(32, 32),
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    class_mode='categorical'
)


history = model.fit_generator(
    train_generator, 
    steps_per_epoch=100, 
    epochs=40, 
    validation_data=validation_generator,
    validation_steps=10,
    workers=12,
    max_queue_size=20, 
    callbacks=[keras.callbacks.EarlyStopping(monitor='acc', mode='max', patience=2, restore_best_weights=True)]

)

test_loss, test_acc = model.evaluate_generator(
    test_generator, 
    steps=10,
    workers=5
)

keras.backend.clear_session()
print(test_loss, test_acc)
