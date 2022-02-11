from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers,layers
from tensorflow.keras.layers import Flatten, Dense,Dropout, MaxPooling2D,Activation, Dense, BatchNormalization, GlobalAveragePooling2D, Input, Conv2D
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

train_dir='/home/ibw1953/Downloads/FP/castle/train'
test_dir='/home/ibw1953/Downloads/FP/castle/test/'
validation_dir='/home/ibw1953/Downloads/FP/castle/valid/'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
 shear_range=0.1, width_shift_range=0.1,
 height_shift_range=0.1, zoom_range=0.1,
 horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
 train_dir,
 target_size=(256, 256),
 batch_size=20,
 class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
 test_dir,
 target_size=(256, 256),
 batch_size=20,
 class_mode='categorical', shuffle=False)
validation_generator = validation_datagen.flow_from_directory(
 validation_dir,
 target_size=(256, 256),
 batch_size=20,
 class_mode='categorical')


input_shape = [256, 256, 3] # as a shape of image
num_classes = 3
model = models.Sequential()
conv_base = VGG16(weights='imagenet', include_top=False,
 input_shape=input_shape)

conv_base.trainable =False

model.add(conv_base)
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
starttime=time.time()
num_epochs = 50
history = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=24, validation_data=validation_generator, validation_steps=8)
# saving the model
model.save('m1epochs50.h5')

train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_acspc:', train_acc)
print('train_loss:',train_loss)
print('test_acc:', test_acc)
print('test_loss:',test_loss)
print("elapsed time (in sec): ", time.time()-starttime)

def plot_acc(h, title="accuracy"):
 plt.plot(h.history['acc'])
 plt.plot(h.history['val_acc'])
 plt.title(title)
 plt.ylabel('Accuracy')
 plt.xlabel('Epoch')
 plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
 plt.plot(h.history ['loss'])
 plt.plot(h.history ['val_loss'])
 plt.title(title)
 plt.ylabel('Loss')
 plt.xlabel('Epoch')
 plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('VGGex1_loss.png')
plt.clf()
plot_acc(history)
plt.savefig('VGGex2_accuracy.png')