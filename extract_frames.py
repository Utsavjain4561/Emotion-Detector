import cv2
import os
import math

FPS=10
video = cv2.VideoCapture("Train Tom and jerry.mp4")
video.set(cv2.CAP_PROP_FPS,FPS)
fps = int(video.get(5))
print('FPs is ='+str(fps))
currentFrame = 0

while(video.isOpened()) :
	frameId = video.get(1)
	ret,frame = video.read()
	if (ret!=True):
		break
	if(frameId%math.floor(fps)==0):
		frameName = 'frame'+str(int(currentFrame/fps))+'.jpg'
		print('Creating fram '+frameName)
		cv2.imwrite(frameName,frame)
	currentFrame+=1
video.release()
cv2.destroyAllWindows()
