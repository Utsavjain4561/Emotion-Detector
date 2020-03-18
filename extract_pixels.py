import cv2
import os
import shutil
with open('Train.csv') as file:
	line = file.readlines()

#path = '/home/uj/Desktop/Project/Emotion Detection/Dataset/data/train'
for path in line:
	modifiedPath = path.split(',')
	name = modifiedPath[0]
	
	img = cv2.imread(name)
	
	
	
	subfolderType = modifiedPath[1].split()[0]
	target = subfolderType+'/'+name
	# print(target)
	# print(cv2.imwrite(target,img))
	# cv2.waitKey(0)
	shutil.copy(name,target)

cv2.destroyAllWindows()
