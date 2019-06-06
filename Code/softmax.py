#載入資料集
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers import ZeroPadding2D,Activation

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 建立分類模型 (MLP) : 平坦層 + 隱藏層 (1024 神經元) + 輸出層 (10 神經元)

model.add(Flatten())
model.add(Dropout(0.25)) 
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)

# train_history=model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=128,verbose=2)


# #評估預測準確率
# scores=model.evaluate(x_test_normalize, y_test_onehot)
# print("Accuracy=", scores[1])
# prediction=model.predict_classes(x_test_normalize)
# print(prediction)

# #預測測試集圖片
# prediction=model.predict_classes(x_test_normalize)
# print(prediction)