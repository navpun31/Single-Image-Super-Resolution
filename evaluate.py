import sewar

def psnr(image_sr,image_hr):
	return sewar.psnr(image_hr,image_sr)

def mse(image_sr,image_hr):
	return sewar.mse(image_hr,image_sr)

def ssim(image_sr,image_hr):
	return sewar.ssim(image_hr,image_sr)[0]

def uqi(image_sr,image_hr):
	return sewar.uqi(image_hr,image_sr)


