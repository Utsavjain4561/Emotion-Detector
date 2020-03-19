import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D,MaxPooling2D,Input,Flatten,Dense,Activation,Dropout
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
from keras.optimizers import  Adam
from keras import regularizers
from keras.regularizers import l1


num_classes = 5
batch_size = 8

train_data_dir = 'data/train'
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=30,
shear_range=0.3,
zoom_range=0.3,
horizontal_flip=True,
fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(48,48),
	batch_size=batch_size,
	class_mode='categorical'
)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,3)))
# model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# # model.add(BatchNormalization())

model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())

model.add(Flatten())
#model.add(Dense(1024,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))
#model.add(Activation("softmax"))

model.summary()

# model=VGG16(weights=None,classes=5,input_shape=(720,1280,3))
# print(model.summary())
model.compile(optimizer="adam",loss="categorical_crossentropy")



# # output_model = model(input_frame)
# # # x = Flatten(name='flatten')(output_model)
# # # x = Dense(4096, activation='relu', name='fc1')(x)
# # # x = Dense(4096, activation='relu', name='fc2')(x)
# # x = Dense(5, activation='softmax', name='predictions')(output_model)
# # my_model = Model(input=input_frame,output=x)

# # my_model.summary()

filepath = os.path.join("./model/model_v6.hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min')
tbCallback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
callbacks = [checkpoint,tbCallback]

nb_train_samples = 298
epochs = 150
model_info  = model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples//batch_size,
	epochs = epochs,
	callbacks = callbacks,
	)