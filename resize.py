import cv2
import os
file_dir = '/home/uj/Desktop/Project/Emotion Detection/Dataset/data/train/surprised'
new_shape = (48, 48)
for filename in os.listdir(file_dir):
	print(filename)
	img = cv2.imread(file_dir+'/'+filename)
	img = cv2.resize(img,new_shape,interpolation=cv2.INTER_AREA)
	print('Resized  shape ',img.shape)
	cv2.imwrite(file_dir+'/'+filename,img)
