import sewar, math
from libsvm import svmutil
from brisque import *

def psnr(image_sr,image_hr):
	return sewar.psnr(image_hr,image_sr)

def mse(image_sr,image_hr):
	return sewar.mse(image_hr,image_sr)

def rmse(image_sr,image_hr):
	return math.sqrt( mse(image_sr,image_hr) )

def ssim(image_sr,image_hr):
	return sewar.ssim(image_hr,image_sr)[0]

def uqi(image_sr,image_hr):
	return sewar.uqi(image_hr,image_sr)

def d_lambda(image_sr,image_bi,path=None):
    # Params: Bicubic and Super-Resolved Images     
    return sewar.no_ref.d_lambda(image_bi,image_sr)

def brisque(image_sr,image_bi,path=None):
    bris = BRISQUE()
    return bris.get_score(path)
