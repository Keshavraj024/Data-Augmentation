"""
Program for Data Augmentation
"""
# Import the necessary modules
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

class data_augmentation:
	def __init__(self,batch,img,num_images):
		self.path = os.getcwd()
		self.i = 1
		self.batch = batch
		self.img = img
		self.num_images = num_images

	def augmentaion(self):
		# Load the image
		img = load_img(self.img)
		img = img_to_array(img)
		img = np.expand_dims(img,0)
		
		# Create a datagen object
		datagen = ImageDataGenerator(vertical_flip=True,
							horizontal_flip = True,
							brightness_range=[0.7,1.0],
							rotation_range = 135,
							zoom_range=[0.2,1.0],
							width_shift_range=0.2,
							height_shift_range=0.2
							)

		#Loop and save the image
		for batch in tqdm(datagen.flow(img,batch_size = self.batch,save_to_dir=self.path+'/Augmented_data',
									save_prefix='image',save_format='jpeg')):
			if self.i > self.num_images:
				break
			self.i += 1

if __name__ == "__main__":
	obj = data_augmentation(1,'Image/dog.jpg',20)
	obj.augmentaion()
			
		

	


