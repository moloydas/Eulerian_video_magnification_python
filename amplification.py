#amplification.py
import scipy.fftpack
import numpy as np

def convert_rgb_2_yiq(img_rgb):
    img_rgb = img_rgb.astype(np.float)
    rgb2yiq = np.array([[0.299,      0.587,        0.114],
                        [0.59590059, -0.27455667, -0.32134392],
                        [0.21153661, -0.52273617, 0.31119955]]).T
    return np.dot(img_rgb, rgb2yiq.copy())

def convert_yiq_2_rgb(img_yiq):
    rgb2yiq = np.array([[0.299,      0.587,        0.114],
                        [0.59590059, -0.27455667, -0.32134392],
                        [0.21153661, -0.52273617, 0.31119955]]).T
    yiq2rgb = np.linalg.inv(rgb2yiq)
    return np.dot(img_yiq, yiq2rgb.copy())

def color_amplification(filt_img_signal, alpha, chrome_attenuation):
    filt_img_signal = convert_rgb_2_yiq(filt_img_signal)
    filt_img_signal[:,:,:,0] *= alpha * chrome_attenuation
    filt_img_signal[:,:,:,1] *= (alpha * chrome_attenuation)
    filt_img_signal[:,:,:,2] *= alpha #* chrome_attenuation
    filt_img_signal = convert_yiq_2_rgb(filt_img_signal)
    return filt_img_signal