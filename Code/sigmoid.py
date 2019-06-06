from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf  
import keras.backend.tensorflow_backend as KTF  
from PIL import ImageFile
import matplotlib.pyplot as plt

# Settings
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Params
MODEL_NAME = 'sigmoid'
SIZE = (448, 448)
SHAPE = (448, 448, 3)
PATIENCE = 5

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))  

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../PornImage/train', target_size=SIZE, batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
testing_set = test_datagen.flow_from_directory('../PornImage/test', target_size=SIZE, batch_size=32, class_mode='binary')

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=SHAPE, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()


early_stopping_callback = EarlyStopping(monitor='val_loss', patience=PATIENCE)
checkpoint_callback = ModelCheckpoint(MODEL_NAME+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
train_history = classifier.fit_generator(training_set, steps_per_epoch = 50, epochs = 1000, validation_data = testing_set, validation_steps = 10, callbacks=[early_stopping_callback, checkpoint_callback])

epoch_count = range(1, len(train_history.history['loss']) + 1)
# summarize history for accuracy
plt.plot(epoch_count, train_history.history['acc'])
plt.plot(epoch_count, train_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')

# summarize history for loss
plt.cla()
plt.plot(epoch_count, train_history.history['loss'])
plt.plot(epoch_count, train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

predict = classifier.predict_generator(testing_set)

from sklearn.metrics import confusion_matrix
import numpy as np

y_true = np.array([0] * 300 + [1] * 300)
y_pred = predict > 0.5

cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.cla()
fig = plt.figure()
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig('confusion_matrix.png')