# Paths
path_model_results = "Model_Results/"
path_scaling_results = "Scaling_Results/"
path_images = "Images/"

# TO BE IMPORTED FROM INDIVIDUAL MODEL SCRIPTS
def bogus(x,y=None):
    return x

# Models and predict functions
models = {
    'vdsr' : bogus,
    'srcnn' : bogus
}

# Function Definition:
# Arguments : LR
# Return Type : SR
# LR and SR are RGB images


# Metrics evaluated for each model
metrics = {
    'psnr' : bogus,
    'ssim' : bogus
}

# Function Definition:
# Arguments : SR, HR
# Return Type : float
# SR and HR are RGB images


# Image names used for comparison
images = ['image1','image2','image3','image4','image5','image6']

scaling_factor = 4
scaling_factors = [2, 4, 8, 10]
scaling_model = 'vdsr'