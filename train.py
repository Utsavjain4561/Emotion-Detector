import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D,MaxPooling2D,Input,Flatten,Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import  Adam


num_classes = 5
batch_size = 16

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
	target_size=(224,224),
	batch_size=batch_size,
	class_mode='categorical'
)

model=VGG16(weights=None,include_top=False)
print(model.summary())
input_frame = Input(shape=(720,1280,3),name='image_input')

output_model = model(input_frame)
# x = Flatten(name='flatten')(output_model)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(5, activation='softmax', name='predictions')(output_model)
my_model = Model(input=input_frame,output=x)

my_model.summary()

filepath = os.path.join("./emotion_detector_models/model_v6_{epoch}.hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_acc',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
callbacks = [checkpoint]
my_model.compile(loss='categorical_crossentropy',
	optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])

nb_train_samples = 298
epochs = 150
model_info  = my_model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples//batch_size,
	epochs = epochs,
	callbacks = callbacks,
	)