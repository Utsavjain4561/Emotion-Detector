import cv2
import os
from keras.models import load_model
import numpy as np
import pandas as pd
model = load_model("/home/uj/Desktop/Project/Emotion Detection/Dataset/model/model_v6.hdf5")

file_dir= '/home/uj/Desktop/Project/Emotion Detection/Dataset/test_data'
output = 'output.csv'
images=[]
clabels=[]
for filename in os.listdir(file_dir):
	#print(filename)
	img = cv2.imread(file_dir+'/'+filename)
	img = np.reshape(img,[48,48,3])
	images.append(img)

images = np.asarray(images)
classes = model.predict_classes(images)
print(classes)
print(len(classes))
for label in classes:
	if(label == 0):
		clabels.append("Unknown")
	elif(label == 1):
		clabels.append("angry")
	elif(label == 2):
		clabels.append("happy")
	elif(label == 3):
		clabels.append("sad")
	elif(label == 4):
		clabels.append("surprised")

print(len(clabels))
my_df = pd.DataFrame(clabels)
my_df.to_csv(output,index=False,header=False)