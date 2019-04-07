import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import pickle
from scipy.misc import imsave
import cv2

def colorize(y, ycrcb): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycrcb[:,:,1]
    img[:,:,2] = ycrcb[:,:,2]
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    return img

def predict(img_bgr):
    model = torch.load('model_vdsr.pth', map_location=lambda storage, loc: storage)["model"]
    im_b_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    im_b_y = im_b_ycrcb[:,:,0].astype(float)
    im_input = im_b_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    model = model.cpu()
    out = model(im_input)
    out = out.cpu()
    im_h_y = out.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h = colorize(im_h_y[0,:,:], im_b_ycrcb)
    return im_h


