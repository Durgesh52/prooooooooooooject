# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf
my_model=tf.keras.models.load_model("keras_model.h5")



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		#resize
		resize_frame=cv2.resize(frame,(224,224))
		resize_frame=np.expand_dims(resize_frame,axis=0)
		resize_frame=resize_frame/255
		prediction=my_model.predict(resize_frame)
		
		rock=int(prediction[0][2]*100)
		paper=int(prediction[0][0]*100)
		scissor=int(prediction[0][1]*100)
		
		print(f"Rock: {rock}% , Paper:{paper}% , Scissor:{scissor}% ")
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
