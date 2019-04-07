from evaluate import *
import vdsr_predict as vdsr
import srcnn_predict as srcnn

# Paths
path_model_results = "Model_Results/"
path_scaling_results = "Scaling_Results/"
path_images = "Images/"


# Models and predict functions
models = {
    'vdsr' : vdsr.predict,
    'srcnn' : srcnn.predict
}

# Function Definition:
# Arguments : LR
# Return Type : SR
# LR and SR are RGB images


# Metrics evaluated for each model
metrics = {
    'psnr' : psnr,
    'ssim' : ssim,
    'mse' : mse,
    'uqi' : uqi
}

# Function Definition:
# Arguments : SR, HR
# Return Type : float
# SR and HR are RGB images


# Image names used for comparison
images = ['image1','image2','image3']

scaling_factor = 4
scaling_factors = [2, 4, 8]
scaling_model = 'vdsr'

hr_dim = [512,512]