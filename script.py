from skimage import data, io, filters
import os
from scipy.misc import imresize
import random

def create_data(directory, ext):
	for f in os.listdir(directory): 
		if f.endswith(ext):
			image = data.imread(os.path.join(directory,f))
			image_hr = imresize(image, [512,512], interp='bicubic', mode=None)
			image_lr = imresize(image, [128,128], interp='bicubic',mode=None)
			image_lr = imresize(image_lr, [512,512], interp='bicubic',mode=None)
			io.imsave('./DIV2K_512/'+f,image_hr)
			io.imsave('./DIV2K_128/'+f,image_lr)
create_data('./DIV2K','.png')